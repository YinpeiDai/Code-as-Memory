"""Public RMBench adapter checks (no simulator, no downloaded demos)."""

import json
from pathlib import Path

import numpy as np
import pytest

from cam.interface.rmbench import adapter as rmbench
from cam.interface.base import check_frames, normalize_action, source_hash
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


def test_changed_seed_rejected_before_native_import(tmp_path, monkeypatch):
    a = adapter()
    records = a.records(TASK, "train")
    (tmp_path / f"record_dataset_{TASK}_metadata.json").write_text(
        json.dumps({"records": [{**records[0], "seed": -1}]})
    )
    monkeypatch.setattr(a, "import_native", lambda task: pytest.fail("simulator imported before seed validation"))
    with pytest.raises(ValueError, match="seed disagrees"):
        a.make_environment(tmp_path, TASK, records[0]["episode"], "train", False)


def test_published_rgbd_configs_prepare_from_manifest_without_reading_h5(tmp_path, monkeypatch):
    from cam.harness import prepare
    from cam.io import read, write

    a = adapter()
    demo = tmp_path / "h5/demo.h5"
    demo.parent.mkdir()
    # Deliberately not HDF5: preparation must trust the published manifest and
    # leave content inspection to the learner, rather than regenerate or audit.
    demo.write_bytes(b"synthetic placeholder, not a demonstration dataset")
    write(tmp_path / "manifest.json", {
        "schema": "cam-dataset/1", "benchmark": "rmbench", "depth": True,
        "benchmark_hash": source_hash(a), "raw_data": None,
        "tasks": {TASK: {"file": "h5/demo.h5", "sha256": "0" * 64,
                         "episodes": a.records(TASK, "train")[:3]}},
    })
    original_open = Path.open

    def no_h5_open(path, *args, **kwargs):
        if path.suffix == ".h5":
            raise AssertionError("Preparation must not open or hash demonstration H5s")
        return original_open(path, *args, **kwargs)

    def no_validation(*args, **kwargs):
        raise AssertionError("Published demonstrations do not need revalidation")

    monkeypatch.setattr(Path, "open", no_h5_open)
    monkeypatch.setattr(prepare, "validate_dataset", no_validation)
    for name in ("demos_claude_code.json", "demos_codex.json"):
        config = read(SOURCE.parents[1] / "configs/rmbench" / name)
        _, cfg, tasks, _, selected = prepare.preparation_inputs(tmp_path, config)
        assert cfg.depth and cfg.mode == "demos" and cfg.budget == 0
        assert tasks == [TASK] and selected["tasks"][TASK]["sha256"] == "0" * 64


def test_raw_extraction_directs_users_to_published_data_without_creating_output(tmp_path):
    from cam.data.extract import extract

    output = tmp_path / "unused"
    with pytest.raises(NotImplementedError, match="published RGB-D dataset"):
        extract(tmp_path / "missing_raw", output, benchmark="rmbench")
    assert not output.exists()
