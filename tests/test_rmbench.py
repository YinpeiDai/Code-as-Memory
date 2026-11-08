"""Offline RMBench adapter and extraction checks (no simulator, no GPU)."""

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
import pytest

from cam.benchmarks import rmbench, rmbench_data
from cam.benchmarks.base import check_frames, normalize_action, source_hash
from cam.prompts import describe_interface

SOURCE = Path(__file__).resolve().parents[1] / "third_party/rmbench"
TASK = "put_back_block"
pytestmark = pytest.mark.skipif(
    not (SOURCE / "envs/_base_task.py").is_file(), reason="third_party/rmbench missing"
)


def adapter():
    return rmbench.RMBench(SOURCE)


def pose(position, forward, left):
    """A camera-to-world OpenGL pose as SAPIEN builds it (x forward, y left, z up)."""
    forward, left = np.asarray(forward, float), np.asarray(left, float)
    forward, left = forward / np.linalg.norm(forward), left / np.linalg.norm(left)
    up = np.cross(forward, left)
    model = np.eye(4)
    # SAPIEN entity pose columns are forward/left/up; its OpenGL model matrix is
    # (right, up, back) = (-left, up, -forward).
    model[:3, :3] = np.stack([-left, up, -forward], axis=1)
    model[:3, 3] = position
    return model


def native_obs(seed=0):
    rng = np.random.default_rng(seed)
    cameras = {}
    for native, _ in rmbench.CAMERAS.values():
        gl = pose(rng.normal(size=3), [0, 1, -0.5], [-1, 0, 0])
        cameras[native] = {
            "rgb": rng.integers(0, 255, (240, 320, 3), dtype=np.uint8),
            "intrinsic_cv": np.array([[257.3, 0, 160], [0, 257.3, 120], [0, 0, 1]], np.float32),
            "extrinsic_cv": np.linalg.inv(gl @ np.diag([1, -1, -1, 1]))[:3].astype(np.float32),
            "cam2world_gl": gl.astype(np.float32),
            "depth": np.ones((240, 320), np.float32),  # forwarded only with depth=True
        }
    return {
        "observation": cameras,
        "pointcloud": [],
        "joint_action": {
            "left_arm": [0.1] * 6,
            "left_gripper": 1.0,
            "right_arm": [0.2] * 6,
            "right_gripper": 0.5,
            "vector": np.zeros(14),
        },
        "endpose": {
            "left_endpose": [-0.3, -0.3, 0.94, 0.7, 0, 0, 0.71],
            "left_gripper": 1.0,
            "right_endpose": [0.3, -0.3, 0.94, 0.7, 0, 0, 0.71],
            "right_gripper": 0.5,
        },
        "third_view_rgb": np.zeros((240, 320, 3), np.uint8),
    }


def test_adapter_imports_without_simulator_and_declares_interface():
    a = adapter()
    assert a.name == "rmbench" and a.limits[TASK] == 500 and a.limits["blocks_ranking_try"] == 3500
    interface = a.interface(TASK, False)
    sensors = interface["sensors"]
    assert {"head_rgb", "front_rgb", "left_wrist_rgb", "right_wrist_rgb"} <= set(sensors)
    assert not any("depth" in name or "seg" in name for name in sensors)
    spec = interface["action"]
    assert len(spec["names"]) == 16 and spec["native_max_steps"] == 500
    assert spec["names"][7] == "left_gripper" and spec["names"][15] == "right_gripper"
    values = normalize_action(spec, [0.0] * 7 + [2.0] + [0.0] * 7 + [-1.0])
    assert values[7] == 1.0 and values[15] == 0.0
    compile(interface["starter"], "starter", "eval")
    assert "| `left_ee_pose` |" in describe_interface(interface)
    assert len(a.interface(TASK, True)["sensors"]) == len(sensors) + 4
    assert a.goal(TASK).startswith("There are four mats")


def test_frames_are_allowlisted_rebuilds():
    a = adapter()
    frame = rmbench.frame(native_obs(), 3)
    check_frames([frame], a.interface(TASK, False)["sensors"])
    assert frame["step"] == 3 and frame["phase"] == "exec"
    assert frame["left_gripper"].shape == () and frame["right_joints"].tolist() == [0.2] * 6
    assert not any("depth" in key or "third_view" in key or "pointcloud" in key for key in frame)


def test_demo_extrinsics_match_live_definition():
    obs = native_obs(1)
    for camera in obs["observation"].values():
        derived = rmbench_data.extrinsic(camera["cam2world_gl"][None])[0]
        assert np.abs(derived - camera["extrinsic_cv"]).max() < 1e-5


def test_opt_in_depth_units_invalid_values_and_alignment():
    obs = native_obs()
    for camera in obs["observation"].values():
        camera["depth"][:] = 1250.4  # native units are millimetres, not metres
        camera["depth"][0, :5] = [np.nan, np.inf, -10, 0, 40000]
    rgb = rmbench.frame(obs, 0)
    rgbd = rmbench.frame(obs, 0, depth=True)
    check_frames([rgbd], adapter().interface(TASK, True)["sensors"])
    for name in rmbench.CAMERAS:
        assert np.array_equal(rgb[name + "_rgb"], rgbd[name + "_rgb"])
        d = rgbd[name + "_depth"]
        assert d.dtype == np.int16 and d[1, 1] == 1250
        assert not d[0, :5].any()
    obs["observation"]["head_camera"]["depth"] = np.ones((120, 160))
    with pytest.raises(ValueError, match="depth shape"):
        rmbench.frame(obs, 0, depth=True)


def test_training_selection_is_first_n_by_episode():
    rows = [{"episode": e, "seed": e + 10} for e in (5, 1, 3, 0)]
    assert [r["episode"] for r in adapter().select_training(rows, 2)] == [0, 1]
    assert len(adapter().select_training(rows, None)) == 4


def test_pinned_metadata_has_no_train_test_overlap():
    root = rmbench.METADATA
    tasks = sorted(p.name[len("record_dataset_") : -len("_metadata.json")] for p in (root / "train").glob("*.json"))
    assert tasks, "no pinned training metadata"
    for task in tasks:
        train = adapter().records(task, "train")
        assert [r["episode"] for r in train] == list(range(len(train)))
        assert max(r["seed"] for r in train) < 100000
        if (root / "test" / f"record_dataset_{task}_metadata.json").exists():
            value = json.loads((root / "test" / f"record_dataset_{task}_metadata.json").read_text())
            test = value["records"]
            assert [r["episode"] for r in test] == list(range(100))
            assert min(r["seed"] for r in test) >= 100000
            assert not {r["seed"] for r in train} & {r["seed"] for r in test}
            assert value["provenance"]["embodiment"] == "aloha-agilex"
            assert value["provenance"]["task_config"] == "demo_clean"


def encoder():
    spec = importlib.util.spec_from_file_location("decoder", SOURCE / "data/decode_image_bit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.images_encoding


def write_episode(path, length, goal, rng):
    encode = encoder()
    with h5py.File(path, "w") as h5:
        string = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("data_format_version", data="v1.0", dtype=string)
        h5.create_dataset("instructions", data=json.dumps([goal]), dtype=string)
        h5.create_group("additional_info").create_dataset("frequency", data=np.int32(15))
        for arm in ("left", "right"):
            for key, width in (("arm_joint_states", 6), ("ee_joint_states", 1), ("ee_poses", 7)):
                values = rng.normal(size=(length + 1, width)).astype(np.float32)
                h5.create_dataset(f"state/{arm}_{key}", data=values[:-1])
                h5.create_dataset(f"action/{arm}_{key}", data=values[1:])
        images = {}
        for name, (_, group) in rmbench.CAMERAS.items():
            # Smooth images so JPEG loss is small and a channel swap is detectable.
            base = np.zeros((length, 240, 320, 3), np.uint8)
            base[..., 0] = 200
            base[..., 1] = 60 + 20 * np.arange(length)[:, None, None]
            base[..., 2] = 20
            images[name] = base
            bits, width = encode(base)
            vision = h5.create_group(f"vision/{group}")
            vision.create_dataset("colors", data=bits, dtype=f"S{width}")
            vision.create_dataset("shape", data=np.array([240, 320, 3], np.int32))
            vision.create_dataset(
                "intrinsic_matrix", data=np.repeat(np.eye(3, dtype=np.float32)[None], length, 0)
            )
            gl = pose(rng.normal(size=3), [0, 1, -0.4], [-1, 0, 0]).astype(np.float32)
            vision.create_dataset("extrinsics_matrix", data=np.repeat(gl[None], length, 0))
        return images


def raw_release(tmp_path, seeds, lengths):
    goal = adapter().goal(TASK)
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    directory = raw / "demo_clean" / TASK / "aloha_agilex"
    (directory / "data").mkdir(parents=True)
    (directory / "seed.txt").write_text("".join(f"{s} " for s in seeds))
    images = {}
    for i, length in enumerate(lengths):
        images[i] = write_episode(directory / "data" / f"episode_{i:07d}.hdf5", length, goal, rng)
    archive = raw / "hf/data" / TASK / "demo_clean.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as z:
        z.write(directory / "seed.txt", f"{TASK}/aloha_agilex/seed.txt")
        for i in range(len(seeds)):
            name = f"episode_{i:07d}.hdf5"
            z.write(directory / "data" / name, f"{TASK}/aloha_agilex/data/{name}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return raw, digest, images


def pin(tmp_path, monkeypatch, seeds, digest, test_seeds):
    metadata = tmp_path / "metadata"
    (metadata / "train").mkdir(parents=True, exist_ok=True)
    (metadata / "test").mkdir(exist_ok=True)
    (metadata / "train" / f"record_dataset_{TASK}_metadata.json").write_text(
        json.dumps(
            {
                "records": [{"episode": i, "seed": s} for i, s in enumerate(seeds)],
                "provenance": {"repo": "TianxingChen/RMBench", "revision": "test", "zip_sha256": digest},
            }
        )
    )
    (metadata / "test" / f"record_dataset_{TASK}_metadata.json").write_text(
        json.dumps({"records": [{"episode": i, "seed": s} for i, s in enumerate(test_seeds)]})
    )
    monkeypatch.setattr(rmbench, "METADATA", metadata)


def test_synthetic_release_extracts_and_validates(tmp_path, monkeypatch):
    seeds = [3, 1, 7, 146]
    raw, digest, images = raw_release(tmp_path, seeds, [5, 4, 6, 3])
    pin(tmp_path, monkeypatch, seeds, digest, [100000 + i for i in range(100)])
    out = tmp_path / "out"
    manifest = rmbench_data.extract(
        raw, out, "rmbench", 2, [TASK], False, True, 1, SOURCE
    )
    assert manifest["benchmark_hash"] == source_hash(adapter())
    item = manifest["tasks"][TASK]
    chosen = [row["episode"] for row in item["episodes"]]
    assert len(chosen) == 2 and chosen == sorted(chosen)
    assert all(row["seed"] == seeds[row["episode"]] for row in item["episodes"])
    assert (out / item["episodes"][0]["video"]).stat().st_size > 0
    rmbench_data.validate_dataset(out)
    with h5py.File(out / item["file"], "r") as h5:
        group = h5[f"episode_{chosen[0]}"]
        assert set(group) == set(manifest["fields"]["frame"])
        rgb = group["head_rgb"][:]
        original = images[chosen[0]]["head"]
        assert np.abs(rgb.astype(int) - original).mean() < 3  # JPEG at source, RGB order kept
        assert np.abs(rgb[..., ::-1].astype(int) - original).mean() > 50
        assert group["left_gripper"].shape == (len(rgb),)
        assert not group["is_demo"][:].any()
    manifest_path = out / "manifest.json"
    tampered = json.loads(manifest_path.read_text())
    tampered["tasks"][TASK]["episodes"][0]["seed"] = 100000
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        rmbench_data.validate_dataset(out)


def test_extraction_rejects_changed_raw_or_overlap(tmp_path, monkeypatch):
    seeds = [3, 1, 7]
    raw, digest, _ = raw_release(tmp_path, seeds, [3, 3, 3])
    pin(tmp_path, monkeypatch, seeds, "0" * 64, [100000])
    with pytest.raises(ValueError, match="sha256"):
        rmbench_data.extract(raw, tmp_path / "a", "rmbench", 2, [TASK], False, False, 1, SOURCE)
    pin(tmp_path, monkeypatch, seeds, digest, [7])
    with pytest.raises(ValueError, match="overlap"):
        rmbench_data.extract(raw, tmp_path / "b", "rmbench", 2, [TASK], False, False, 1, SOURCE)
    pin(tmp_path, monkeypatch, seeds, digest, [100000])
    path = raw / "demo_clean" / TASK / "aloha_agilex/data/episode_0000001.hdf5"
    with h5py.File(path, "a") as h5:
        h5["state/left_ee_poses"][0, 0] += 1.0  # same size, different bytes
    with pytest.raises(ValueError, match="CRC|differs"):
        rmbench_data.extract(raw, tmp_path / "c", "rmbench", "all", [TASK], False, False, 1, SOURCE)


def test_repeated_extraction_is_byte_identical(tmp_path, monkeypatch):
    seeds = [3, 1, 7, 146]
    raw, digest, _ = raw_release(tmp_path, seeds, [3, 4, 3, 2])
    pin(tmp_path, monkeypatch, seeds, digest, [100000])
    hashes = [
        rmbench_data.extract(raw, tmp_path / name, "rmbench", 2, [TASK], False, False, 1, SOURCE)["tasks"][TASK]["sha256"]
        for name in ("pilot", "full")
    ]
    assert hashes[0] == hashes[1]


def test_changed_seed_rejected_before_native_import(tmp_path, monkeypatch):
    a = adapter()
    records = a.records(TASK, "train")
    (tmp_path / f"record_dataset_{TASK}_metadata.json").write_text(
        json.dumps({"records": [{**records[0], "seed": -1}]})
    )
    monkeypatch.setattr(a, "import_native", lambda task: pytest.fail("simulator imported before seed validation"))
    with pytest.raises(ValueError, match="seed disagrees"):
        a.make_environment(tmp_path, TASK, records[0]["episode"], "train", False)
