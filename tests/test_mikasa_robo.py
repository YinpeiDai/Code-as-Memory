"""Offline MIKASA-Robo adapter, pinned metadata and RLDS extraction tests (no simulator)."""

import csv
import hashlib
import io
import json
from pathlib import Path
import struct

import h5py
import numpy as np
import pytest
from PIL import Image

import cam.benchmarks.mikasa_robo as mikasa
import cam.benchmarks.mikasa_robo_data as data
from cam.benchmarks.base import check_frames, normalize_action
from cam.benchmarks.registry import get_adapter
from cam.config import RunConfig
from cam.io import read, write
from cam.prompts import render

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/mikasa_robo"
CONFIG = ROOT / "configs/mikasa_robo/demos_claude_code.json"
needs_source = pytest.mark.skipif(
    not (SOURCE / "mikasa_robo_vla_envs.csv").is_file(), reason="MIKASA-Robo submodule missing"
)


@pytest.fixture
def adapter():
    if not (SOURCE / "mikasa_robo_suite/vla").is_dir():
        pytest.skip("MIKASA-Robo submodule missing")
    return get_adapter("mikasa_robo")


@needs_source
def test_pinned_metadata_covers_canonical_tasks_without_overlap(adapter):
    with (SOURCE / "mikasa_robo_vla_envs.csv").open(newline="") as stream:
        rows = {row["Name"]: row for row in csv.DictReader(stream, delimiter=";")}
    assert adapter.tasks() == sorted(rows) and len(rows) == 90
    release = read(mikasa.METADATA / "release.json")
    folders = {path.split("/")[0] for path in release["files"]}
    assert folders == {mikasa.dataset_name(task) for task in rows}
    assert mikasa.dataset_name("RememberShapeAndColor3x2-Long-VLA-v0") == (
        "remember_shape_and_color_3x2_long_vla_v0"
    )
    assert mikasa.dataset_name("BatteriesCheckerHard-3-VLA-v0") == "batteries_checker_hard_3_vla_v0"
    for task in rows:
        train, test = adapter.records(task, "train"), adapter.records(task, "test")
        assert [r["episode"] for r in train] == list(range(len(train))) and len(train) >= 25
        assert [(r["episode"], r["seed"]) for r in test] == [
            (i, 4242424242 + i) for i in range(50)
        ]
        assert not {r["seed"] for r in train} & {r["seed"] for r in test}
        assert all("difficulty" not in r for r in train + test)
        meta = {split: adapter.metadata(task, split) for split in ("train", "test")}
        assert meta["train"]["max_episode_steps"] == meta["test"]["max_episode_steps"]
        assert meta["train"]["provenance"]["revision"] == release["revision"]
    assert adapter.max_episode_steps("FindImposterShapeAndColor5x3-VLA-v0") == 40
    assert max(adapter.max_episode_steps(t) for t in rows) == 2160


@needs_source
def test_config_covers_native_horizons(adapter):
    config = RunConfig.from_dict(read(CONFIG))
    assert config.benchmark == "mikasa_robo" and not config.depth and not config.builtin_demo
    assert config.max_decisions == 40
    assert config.max_action_steps >= max(adapter.max_episode_steps(t) for t in adapter.tasks())


def native_obs(task_cue=False):
    rng = np.random.default_rng(0)
    obs = {
        "rgb": rng.integers(0, 256, (1, 128, 128, 6), dtype=np.uint8),
        "proprio": rng.normal(size=(1, 7)).astype(np.float32),
    }
    if task_cue:
        obs["task_cue"] = np.ones((1, 1), np.float16)
    return obs


@needs_source
def test_interface_frames_are_allowlisted(adapter):
    interface = adapter.interface("ShellGameColorLampTouch-VLA-v0", False)
    assert list(interface["sensors"]) == ["base_rgb", "wrist_rgb", "tcp", "gripper_opening"]
    assert interface["video"] == ["base_rgb", "wrist_rgb"]
    obs = native_obs(task_cue=True)
    frame = mikasa.frame(obs, 3)
    check_frames([frame], interface["sensors"])
    assert frame["step"] == 3 and frame["phase"] == "exec" and "task_cue" not in frame
    assert np.array_equal(frame["base_rgb"], obs["rgb"][0, ..., :3])
    assert np.array_equal(frame["wrist_rgb"], obs["rgb"][0, ..., 3:])
    assert np.array_equal(np.concatenate([frame["tcp"], frame["gripper_opening"]]), obs["proprio"][0])
    with pytest.raises(ValueError):
        mikasa.frame({**obs, "oracle_info": np.zeros((1, 1))}, 0)
    with pytest.raises(ValueError):
        mikasa.frame({**obs, "proprio": obs["proprio"].astype(np.float64)}, 0)
    with pytest.raises(ValueError):
        adapter.interface("ShellGameTouch-VLA-v0", True)
    spec = interface["action"]
    assert spec["names"] == ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    assert normalize_action(spec, [2, -2, 0.5, 0, 0, 0, 3]).tolist() == [1, -1, 0.5, 0, 0, 0, 1]
    assert "Native step limit for this task: 30." in spec["description"]
    assert mikasa.instruction({"language_instruction": ["Rotate by 5 degrees."]}) == (
        "Rotate by 5 degrees."
    )
    with pytest.raises(ValueError):
        mikasa.instruction({"language_instruction": ["a", "b"]})


class FakeEnv:
    """Stands in for NativeEnvironment: scripted native step outcomes."""

    def __init__(self, outcomes, horizon):
        self.outcomes, self.horizon, self.steps = list(outcomes), horizon, 0

    def reset(self):
        self.steps = 0
        return native_obs(task_cue=True), {"language_instruction": ["Touch the cup."], "oracle_info": 1}

    def step(self, values):
        self.steps += 1
        terminated, truncated, success = self.outcomes.pop(0)
        return native_obs(), terminated, truncated, success


@needs_source
def test_status_mapping_follows_native_signals(adapter):
    start = adapter.begin(FakeEnv([], 3), builtin_demo=False, depth=False)
    assert start.goal == "Touch the cup." and [f["step"] for f in start.frames] == [0]
    cases = [
        ([(False, False, False), (False, False, True)], ["running", "success"]),
        ([(False, False, False), (True, False, False)], ["running", "fail"]),
        ([(False, False, False), (False, True, False)], ["running", "timeout"]),
        ([(False, False, False)] * 3, ["running", "running", "timeout"]),
        ([(True, True, True)], ["success"]),
    ]
    for outcomes, statuses in cases:
        env = FakeEnv(outcomes, 3)
        results = [adapter.advance(env, [0] * 7, {}, depth=False) for _ in statuses]
        assert [r.status for r in results] == statuses
        assert [r.frames[0]["step"] for r in results] == list(range(1, len(statuses) + 1))
        assert all(r.metrics is None and r.private_error is None for r in results)


@needs_source
def test_prompt_uses_guide_and_declared_interface(adapter):
    config = RunConfig.from_dict(read(CONFIG))
    task = "ShellGameTouch-VLA-v0"
    records = adapter.records(task, "train")[:2]
    text = render(config, task, records, records, adapter, adapter.interface(task, False))
    assert "## MIKASA-Robo data and simulator protocol" in text
    assert "| `gripper_opening` | float32 | 1 |" in text
    assert "robot.step(dx, dy, dz, drx, dry, drz, gripper)" in text
    assert "4242424" not in text and "task_cue" not in text and "oracle" not in text


# ---------------------------------------------------------------- synthetic RLDS


def _varint(value):
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number, payload):
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _feature(value):
    if isinstance(value, list):  # BytesList
        return _field(1, b"".join(_field(1, v) for v in value))
    value = np.asarray(value)
    if value.dtype.kind == "f":
        return _field(2, _field(1, value.astype("<f4").tobytes()))
    packed = b"".join(_varint(int(v) & (2**64 - 1)) for v in value.reshape(-1))
    return _field(3, _field(1, packed))


def example(features):
    entries = b"".join(
        _field(1, _field(1, key.encode()) + _field(2, _feature(value)))
        for key, value in features.items()
    )
    return _field(1, entries)


def tfrecord(payloads):
    out = bytearray()
    for payload in payloads:
        out += struct.pack("<Q", len(payload)) + b"\0" * 4 + payload + b"\0" * 4
    return bytes(out)


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def git_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


TASK = "SyntheticMemory-VLA-v0"
FOLDER = "synthetic_memory_vla_v0/1.0.0"


def episode_arrays(k, length):
    rng = np.random.default_rng(k)
    return {
        "base": rng.integers(0, 256, (length, 128, 128, 3), dtype=np.uint8),
        "wrist": rng.integers(0, 256, (length, 128, 128, 3), dtype=np.uint8),
        "proprio": rng.normal(size=(length, 7)).astype(np.float32),
        "action": rng.uniform(-1, 1, (length, 7)).astype(np.float32),
    }


def episode_example(k, length, goal="Remember the cup.", extra=None):
    arrays = episode_arrays(k, length)
    features = {
        "steps/observation/image": [png(a) for a in arrays["base"]],
        "steps/observation/wrist_image": [png(a) for a in arrays["wrist"]],
        "steps/observation/proprio": arrays["proprio"].reshape(-1),
        "steps/action": arrays["action"].reshape(-1),
        "steps/language_instruction": [goal.encode()] * length,
        "steps/is_first": (np.arange(length) == 0).astype(np.int64),
        "steps/is_last": (np.arange(length) == length - 1).astype(np.int64),
        "steps/is_terminal": (np.arange(length) == length - 1).astype(np.int64),
        "steps/reward": np.linspace(0, 1, length, dtype=np.float32),
        "steps/discount": np.ones(length, np.float32),
        "episode_metadata/file_path": [
            f"/somewhere/data_npz/{TASK}/train_data_{k}.npz".encode()
        ],
        **(extra or {}),
    }
    return example(features)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Pinned metadata + a 6-episode, 2-shard release for one synthetic task."""
    count, lengths = 6, [3, 5, 2, 4, 6, 3]
    seeds = [100 + 7 * k for k in range(count)]
    metadata, raw = tmp_path / "metadata", tmp_path / "raw"
    folder = raw / FOLDER
    folder.mkdir(parents=True)
    order = [4, 1, 5, 0, 3, 2]  # shuffled like TFDS
    shards = {
        "mikasa_dataset-train.tfrecord-00000-of-00002": order[:3],
        "mikasa_dataset-train.tfrecord-00001-of-00002": order[3:],
    }
    files = {}
    for name, ids in shards.items():
        blob = tfrecord([episode_example(k, lengths[k]) for k in ids])
        (folder / name).write_bytes(blob)
        files[f"{FOLDER}/{name}"] = {"size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    jsons = {
        "metadata.json": {
            "env_id": TASK,
            "num_episodes": count,
            "episode_lengths": lengths,
            "success_once": [True] * count,
            "reward_sums": [1.0] * count,
            "episode_seeds": seeds,
        },
        "dataset_info.json": {
            "splits": [{"name": "train", "shardLengths": ["3", "3"]}],
            "version": "1.0.0",
        },
        "features.json": {"synthetic": True},
    }
    for name, value in jsons.items():
        blob = json.dumps(value).encode()
        (folder / name).write_bytes(blob)
        files[f"{FOLDER}/{name}"] = {"size": len(blob), "git_sha1": git_sha1(blob)}
    write(metadata / "release.json", {"repo": "test/rlds", "repo_type": "dataset", "revision": "r1", "files": files})
    common = {"env_id": TASK, "max_episode_steps": 30, "noop_cue_phase": True}
    write(
        metadata / "train" / f"record_dataset_{TASK}_metadata.json",
        {**common, "records": [{"task": TASK, "episode": k, "seed": s} for k, s in enumerate(seeds)], "provenance": {"revision": "r1"}},
    )
    write(
        metadata / "test" / f"record_dataset_{TASK}_metadata.json",
        {**common, "records": [{"task": TASK, "episode": i, "seed": 4242424242 + i} for i in range(50)]},
    )
    monkeypatch.setattr(mikasa, "METADATA", metadata)
    monkeypatch.setattr(data, "METADATA", metadata)
    monkeypatch.setattr(data, "FEATURES_SHA1", files[f"{FOLDER}/features.json"]["git_sha1"])
    if not (SOURCE / "mikasa_robo_suite/vla").is_dir():
        pytest.skip("MIKASA-Robo submodule missing")
    return {"raw": raw, "folder": folder, "metadata": metadata, "lengths": lengths, "seeds": seeds}


def test_synthetic_rlds_extraction_is_lossless_and_allowlisted(synthetic, tmp_path):
    output = tmp_path / "out"
    manifest = data.extract(synthetic["raw"], output, "mikasa_robo", 4, [TASK], False, True, 1)
    assert manifest["demonstration_schema"] == "cam-demonstrations/2" and not manifest["depth"]
    assert manifest["release"]["revision"] == "r1"
    item = manifest["tasks"][TASK]
    chosen = [row["episode"] for row in item["episodes"]]
    assert len(chosen) == 4 and chosen == sorted(chosen)
    assert manifest == data.validate_dataset(output)
    with h5py.File(output / item["file"], "r") as h5:
        assert h5.attrs["schema"] == "cam-demonstrations/2" and h5.attrs["task"] == TASK
        for row in item["episodes"]:
            k, group = row["episode"], h5[f"episode_{row['episode']}"]
            arrays = episode_arrays(k, synthetic["lengths"][k])
            assert set(group) == {"step", "is_demo", "base_rgb", "wrist_rgb", "tcp", "gripper_opening", "action"}
            assert np.array_equal(group["base_rgb"][:], arrays["base"])
            assert np.array_equal(group["wrist_rgb"][:], arrays["wrist"])
            assert np.array_equal(group["tcp"][:], arrays["proprio"][:, :6])
            assert np.array_equal(group["gripper_opening"][:], arrays["proprio"][:, 6:])
            assert np.array_equal(group["action"][:], arrays["action"])
            assert not group["is_demo"][:].any() and row["seed"] == synthetic["seeds"][k]
            assert group.attrs["goal"] == "Remember the cup." and "zero command" in group.attrs["action_alignment"]
            assert (output / row["video"]).is_file()
    audit = read(output / "audit" / f"{TASK}.json")
    assert sorted(r["episode"] for r in audit["all_training_episodes"]) == list(range(6))
    text = json.dumps(manifest)
    assert "reward" not in json.dumps(manifest["fields"]) and "train_data_" not in text


def test_extraction_rejects_tampering_and_overlap(synthetic, tmp_path):
    shard = synthetic["folder"] / "mikasa_dataset-train.tfrecord-00000-of-00002"
    original = shard.read_bytes()
    shard.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        data.extract(synthetic["raw"], tmp_path / "a", "mikasa_robo", 2, [TASK], False, False, 1)
    shard.write_bytes(original)
    path = synthetic["metadata"] / "train" / f"record_dataset_{TASK}_metadata.json"
    pinned = read(path)
    pinned["records"][0]["seed"] = 4242424242
    write(path, pinned)
    with pytest.raises(ValueError, match="overlap"):
        data.extract(synthetic["raw"], tmp_path / "b", "mikasa_robo", 2, [TASK], False, False, 1)


def test_parse_rejects_unexpected_or_privileged_features(synthetic):
    train = {k: {"episode": k} for k in range(6)}
    item = data.parse(episode_example(1, 5), TASK, train)
    assert item["episode"] == 1 and item["frames"] == 5
    with pytest.raises(ValueError, match="Unexpected RLDS features"):
        data.parse(episode_example(1, 5, extra={"steps/observation/oracle": np.zeros(5, np.float32)}), TASK, train)
    with pytest.raises(ValueError, match="not a pinned"):
        data.parse(episode_example(9, 5), TASK, train)


@needs_source
def test_historical_contract_stays_private(adapter):
    blink = adapter.diagnostic_task_contract("BlinkCountButtonPressEasy-VLA-v0")
    assert blink.startswith("## Task contract\n")
    assert "- Each press must be followed by a release and lift before the next one." in blink
    assert "Success is produced only when the black submit button is pressed." in blink
    assert "risen at least 0.085 m above its height when the press registered" in blink
    assert "customize" not in blink.lower() and "REQUIRED_LIFT_HEIGHT" not in blink
    remember = adapter.diagnostic_task_contract("RememberColor9-VLA-v0")
    assert "within 0.05 m of the centre of the cube" in remember and "0.2 rad/s" in remember
    imposter = adapter.diagnostic_task_contract("FindImposterColor3-VLA-v0")
    assert "Episode flow" not in imposter and "GOAL_THRESH" not in imposter  # names class parameters
    chains = mikasa.task_classes(adapter.source)
    assert set(chains) >= set(adapter.tasks())
    for task in adapter.tasks():
        guide = adapter.diagnostic_task_contract(task)
        names = set().union(*(c.names for c in chains[task]))
        if guide:
            assert not set(mikasa.PARAMETER.findall(guide)) & names, task
            assert "How to customize" not in guide and "4242" not in guide and "{" not in guide
    config = RunConfig.from_dict(read(CONFIG))
    task = "BlinkCountButtonPressEasy-VLA-v0"
    records = adapter.records(task, "train")[:2]
    text = render(config, task, records, records, adapter, adapter.interface(task, False))
    assert blink not in text and "Official success test" not in text
    assert "0.085" not in text and "0.0251" not in text
    assert (ROOT / "src/cam/prompts/core.md").read_text() in text


@needs_source
def test_success_tests_refuse_changed_evaluation_code(adapter):
    chain = mikasa.task_classes(adapter.source)["RememberColor9-VLA-v0"]
    tests = read(mikasa.METADATA / "success_tests.json")["classes"]
    assert mikasa.success_test(chain, tests).startswith("Official success test")
    stale = {k: {**v, "sha256": "0" * 64} for k, v in tests.items()}
    with pytest.raises(ValueError, match="re-derive"):
        mikasa.success_test(chain, stale)
    assert mikasa.constant(chain, "GOAL_THRESH") == 0.05


@needs_source
def test_every_task_has_pinned_success_test(adapter):
    for task in adapter.tasks():
        guide = adapter.diagnostic_task_contract(task)
        assert guide and "Official success test (from the environment's evaluation code)" in guide, task
    timed = adapter.diagnostic_task_contract("TimedTransferEasy-VLA-v0")
    assert "from 95 to 105 (inclusive)" in timed and "DELAY_STEPS" not in timed
