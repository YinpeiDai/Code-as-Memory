"""The shared harness with a synthetic, non-RoboMME benchmark interface."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from cam.interface.base import EpisodeStart, StepResult, action, check_frames, sensor
from cam.config import RunConfig
from cam.data import demonstrations
from cam.io import read, sha256, write

TASK = "Toy-Task-v0"
TRAIN = [{"task": TASK, "episode": i, "seed": 10 + i} for i in range(6)]
TEST = [{"task": TASK, "episode": i, "seed": 900 + i} for i in range(2)]


class ToyEnv:
    def __init__(self, leak=False):
        self.leak, self.t = leak, 0

    def frame(self, phase="exec"):
        value = {
            "phase": phase,
            "step": self.t,
            "cam_rgb": np.full((8, 8, 3), self.t, np.uint8),
            "state": np.array([self.t, 0, 0], np.float32),
        }
        if self.leak:
            value["object_pose"] = np.zeros(7)
        return value

    def close(self):
        pass


class Toy:
    name = "toy"
    package = "toy"
    guide = "robomme/guide.md"
    describe_interface = True

    def __init__(self, source=None, leak=False):
        self.source = source
        self.leak = leak
        self.sent = []

    def tasks(self):
        return [TASK]

    def records(self, task, split):
        return TRAIN if split == "train" else TEST

    def action_spec(self, task):
        return action(
            ("push", "unused"),
            "toy push",
            low=[-1.0, None],
            high=[1.0, None],
            ignored=("unused",),
        )

    def interface(self, task, depth):
        return {
            "sensors": {
                "cam_rgb": sensor("uint8", (8, 8, 3), "toy camera"),
                "state": sensor("float32", (3,), "toy state"),
            },
            "action": self.action_spec(task),
            "video": ["cam_rgb"],
            "starter": "robot.step(0.0, 0.0)",
        }

    def sensor_summary(self, task, depth):
        return "toy RGB and state"

    def select_training(self, records, count):
        return sorted(records, key=lambda r: r["episode"])[: count or len(records)]

    def make_environment(self, metadata, task, episode, split, depth):
        return ToyEnv(self.leak)

    def begin(self, env, *, builtin_demo, depth):
        return EpisodeStart("push twice", [env.frame("demo"), env.frame()], {})

    def advance(self, env, value, context, *, depth):
        self.sent.append(np.asarray(value).tolist())
        env.t += 1
        status = "success" if env.t >= 2 else "running"
        return StepResult([env.frame()], status, env.t, metrics={"progress": env.t / 2})


@pytest.fixture
def toy(tmp_path, monkeypatch):
    from cam.interface import registry as registry
    from cam.harness import prepare

    source = tmp_path / "toy_source"
    source.mkdir()
    (source / "sim.py").write_text("# pinned toy simulator\n")
    adapter = Toy(source)
    monkeypatch.setitem(
        registry.BENCHMARKS, "toy", {"status": "implemented", "data": "cam.data.demonstrations"}
    )
    monkeypatch.setattr(prepare, "get_adapter", lambda name, source=None: adapter)
    monkeypatch.setattr(prepare, "simulator_python", lambda adapter: None)
    return adapter, source


def make_dataset(root, adapter, length=4):
    from cam.interface.base import source_hash

    declared = demonstrations.declaration(
        {
            "cam_rgb": {"dtype": "uint8", "shape": [8, 8, 3], "description": "toy camera"},
            "action": {"dtype": "float32", "shape": [2], "description": "toy action"},
        },
        {"cam_K": {"dtype": "float64", "shape": [3, 3], "description": "intrinsics"}},
    )
    selected = demonstrations.random_select(TRAIN, 3, "toy", TASK)
    (root / "h5").mkdir(parents=True)
    path = root / "h5" / f"record_dataset_{TASK}.h5"
    with demonstrations.create(path, "toy", TASK, declared, False, "synthetic") as h5:
        for row in selected:
            episode = demonstrations.Episode(
                h5,
                declared,
                row["episode"],
                length,
                {"goal": "push twice", "difficulty": None, "action_alignment": "a", "step_semantics": "s"},
                {"cam_K": np.eye(3)},
            )
            episode.append(
                {
                    "cam_rgb": np.zeros((length, 8, 8, 3), np.uint8),
                    "action": np.ones((length, 2), np.float32),
                    "is_demo": np.arange(length) == 0,
                }
            )
            episode.close()
    write(
        root / "manifest.json",
        {
            "schema": "cam-dataset/1",
            "demonstration_schema": demonstrations.SCHEMA,
            "benchmark": "toy",
            "benchmark_hash": source_hash(adapter),
            "raw_data": "synthetic",
            "depth": False,
            "fields": declared,
            "tasks": {
                TASK: {
                    "file": path.relative_to(root).as_posix(),
                    "sha256": sha256(path),
                    "episodes": [{**row, "frames": length} for row in selected],
                }
            },
        },
    )
    return root


def test_random_selection_is_seeded_and_sorted():
    first = demonstrations.random_select(TRAIN, 3, "toy", TASK)
    assert first == demonstrations.random_select(list(reversed(TRAIN)), 3, "toy", TASK)
    assert [r["episode"] for r in first] == sorted(r["episode"] for r in first)
    assert len({r["episode"] for r in first}) == 3


def test_v2_dataset_validation_rejects_undeclared_fields(tmp_path, toy):
    adapter, _ = toy
    root = make_dataset(tmp_path / "data", adapter)
    demonstrations.validate_dataset(root)
    path = root / "h5" / f"record_dataset_{TASK}.h5"
    with h5py.File(path, "a") as h5:
        next(iter(h5.values())).create_dataset("object_pose", data=np.zeros((4, 7)))
    with pytest.raises(ValueError):
        demonstrations.validate_dataset(root, verify_hashes=False)


def test_generic_prepare_prompt_and_trajectory(tmp_path, toy, monkeypatch):
    from cam.harness.prepare import prepare, starter
    from cam.runtime import trajectory

    adapter, _ = toy
    root = make_dataset(tmp_path / "data", adapter)
    run = tmp_path / "run"
    prepare(run, root, RunConfig(benchmark="toy", mode="demos", depth=False))
    plan = read(run / "run.json")
    interface = plan["interfaces"][TASK]
    assert interface["action"]["names"] == ["push", "unused"]
    workspace = run / "tasks" / TASK / "workspace"
    assert (workspace / "tools/policy.py").read_text() == starter(interface)
    prompt = (run / "tasks" / TASK / "prompt.md").read_text()
    assert "Declared observation and action interface" in prompt
    assert "`robot.step(push, unused)`" in prompt and "toy RGB and state" in prompt
    monkeypatch.setattr(trajectory, "ROOT", run / "tasks" / TASK / "demonstrations")
    episode = trajectory.training_episodes(TASK)[0]
    with trajectory.Trajectory(TASK, episode) as traj:
        assert traj.T == 4 and traj.exec_start == 1 and traj.goal == "push twice"
        assert set(traj.fields) == {"cam_rgb", "action", "is_demo", "step"}
        assert traj.static["cam_K"].shape == (3, 3)
        assert traj.action(2).tolist() == [1.0, 1.0]
        assert traj.get("cam_rgb").shape == (4, 8, 8, 3)


def test_frames_must_match_declared_sensors():
    adapter = Toy()
    sensors = adapter.interface(TASK, False)["sensors"]
    check_frames([ToyEnv().frame()], sensors)
    with pytest.raises(ValueError, match="extra"):
        check_frames([ToyEnv(leak=True).frame()], sensors)
    bad = ToyEnv().frame()
    bad["state"] = bad["state"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        check_frames([bad], sensors)


POLICY = '''def perceive_decide(obs, goal, memory, last):
    """Push with a large value; the harness must clip it and zero the ignored part."""
    return {"memory": {}, "act": {"tool": "act_push", "args": {}}}


def act_push(robot):
    """Push until the episode ends."""
    for _ in range(5):
        robot.step(push=5.0, unused=3.0)
    return {"done": False}
'''


def policy_program(folder):
    (folder / "tools").mkdir(parents=True)
    (folder / "task.json").write_text(json.dumps({"task": TASK, "runtime": "python", "memory": []}))
    (folder / "policy.md").write_text("toy\n")
    (folder / "tools/policy.py").write_text(POLICY)
    return folder


@pytest.mark.isolation
def test_broker_generic_actions_metrics_and_leak(tmp_path):
    from cam.harness.broker import execute
    from cam.harness.prepare import make_runtime

    runtime = make_runtime(tmp_path / "runtime")
    program = policy_program(tmp_path / "program")
    config = {
        "depth": False,
        "builtin_demo": True,
        "max_decisions": 5,
        "max_action_steps": 50,
        "policy_timeout": 120,
    }
    adapter = Toy()
    result = execute(adapter, tmp_path, TASK, 0, "test", program, runtime, tmp_path / "ep", config)
    assert result["status"] == "success" and result["steps"] == 2, result
    assert adapter.sent == [[1.0, 0.0], [1.0, 0.0]]
    assert result["native_metrics"] == {"progress": 1.0}
    assert np.load(tmp_path / "ep/actions.npy").shape == (2, 2)
    with h5py.File(tmp_path / "ep/observations.h5") as h5:
        assert set(h5) == {"cam_rgb", "state", "step", "phase"} and len(h5["step"]) == 4
    train = execute(adapter, tmp_path, TASK, 0, "train", program, runtime, tmp_path / "train", config)
    assert "native_metrics" not in train
    assert read(tmp_path / "train/private_native_metrics.json") == {"progress": 1.0}
    leaked = execute(Toy(leak=True), tmp_path, TASK, 0, "test", program, runtime, tmp_path / "leak", config)
    assert leaked["status"] == "harness_error" and not leaked["scorable"]


@pytest.mark.isolation
@pytest.mark.parametrize("recording", [{"recording": "results"}, {"evaluation_recording": "results"}])
def test_results_recording_preserves_execution_without_large_artifacts(tmp_path, recording):
    from cam.harness.broker import execute
    from cam.harness.prepare import make_runtime

    runtime = make_runtime(tmp_path / "runtime")
    program = policy_program(tmp_path / "program")
    config = {"depth": False, "builtin_demo": True, "max_decisions": 5,
              "max_action_steps": 50, "policy_timeout": 120}
    full_adapter, light_adapter = Toy(), Toy()
    full = execute(full_adapter, tmp_path, TASK, 0, "test", program, runtime, tmp_path / "full", config)
    light = execute(light_adapter, tmp_path, TASK, 0, "test", program, runtime,
                    tmp_path / "light", {**config, **recording})
    for key in ("status", "success", "scorable", "steps", "step_calls", "decisions", "native_metrics", "program_hash"):
        assert full[key] == light[key]
    assert full_adapter.sent == light_adapter.sent
    assert light["observation_frames"] == full["video_frames"] == 4
    assert light["video_frames"] == 0
    assert not any(p.suffix in (".h5", ".mp4", ".npy", ".strace") for p in (tmp_path / "light").iterdir())
    assert (tmp_path / "light/result.json").is_file()
    assert (tmp_path / "light/decisions.jsonl").is_file()
    assert light["recording"] == "results"
    if "evaluation_recording" in recording:
        train = execute(Toy(), tmp_path, TASK, 0, "train", program, runtime,
                        tmp_path / "train", {**config, **recording})
        assert train["status"] == "success" and train["recording"] == "full"
        assert train["video_frames"] == 4
        assert (tmp_path / "train/observations.h5").is_file()
        assert "native_metrics" not in train


@pytest.mark.isolation
def test_preflight_uses_declared_interface(tmp_path):
    from cam.harness.prepare import make_runtime
    from cam.harness.preflight import transport_preflight

    runtime = make_runtime(tmp_path / "runtime")
    report = transport_preflight(runtime, interface=Toy().interface(TASK, False), timeout=120)
    assert report["passed"], report
    # Tiny sensors cannot reach 256 MiB within the 100000-frame header cap.
    assert report["frames"] == 100000 or report["aggregate_frame_message_bytes"] > 256 * 1024**2


def test_stratified_selection_is_proportional_and_seeded():
    rows = [{"episode": i, "variant": "a" if i < 34 else "b" if i < 67 else "c"} for i in range(100)]
    picked = demonstrations.stratified_select(rows, 25, "toy", TASK, lambda r: r["variant"])
    counts = {v: sum(r["variant"] == v for r in picked) for v in "abc"}
    assert counts == {"a": 9, "b": 8, "c": 8}
    assert picked == demonstrations.stratified_select(list(reversed(rows)), 25, "toy", TASK, lambda r: r["variant"])
    assert demonstrations.allocate({"x": 80, "y": 20}, 25) == {"x": 20, "y": 5}
    assert demonstrations.allocate({"x": 99, "y": 1}, 25) == {"x": 24, "y": 1}


def test_static_calibration_is_not_inferred_from_episode_length(tmp_path, toy, monkeypatch):
    from cam.runtime import trajectory

    adapter, _ = toy
    root = make_dataset(tmp_path / "data", adapter, length=3)
    monkeypatch.setattr(trajectory, "ROOT", root / "h5")
    with trajectory.Trajectory(TASK, trajectory.training_episodes(TASK)[0]) as traj:
        assert "cam_K" in traj.static and "cam_K" not in traj.fields
        assert np.array_equal(traj.get("cam_K", 1), np.eye(3))
        assert traj.description["action"]["description"] == "toy action"
        assert traj.get("action", 1).shape == (2,)


def test_scalar_sensors_keep_shape_across_wire():
    from cam.runtime.observations import Frame
    from cam.runtime.wire import encoded

    frame = Frame(encoded([{"phase": "exec", "step": 0, "gripper": np.array(0.5)}])[0])
    assert frame.gripper.shape == () and float(frame.gripper) == 0.5
    assert not frame.gripper.flags.writeable


@pytest.mark.parametrize("bad_setting", ["builtin_demo", "training_seed_count"])
def test_invalid_settings_rejected_before_creating_run(tmp_path, toy, bad_setting):
    from cam.harness.prepare import prepare

    adapter, _ = toy
    adapter.supports_builtin_demo = False
    dataset = make_dataset(tmp_path / "data", adapter)
    options = {"builtin_demo": False, "training_seed_count": 3}
    options[bad_setting] = True if bad_setting == "builtin_demo" else 4
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="reset video|training seeds"):
        prepare(run, dataset, RunConfig(benchmark="toy", mode="demos", depth=False, **options))
    assert not run.exists()


def test_missing_test_metadata_rejected_before_creating_run(tmp_path, toy, monkeypatch):
    from cam.harness.prepare import prepare

    adapter, _ = toy
    dataset = make_dataset(tmp_path / "data", adapter)
    original = adapter.records

    def records(task, split):
        if split == "test":
            raise FileNotFoundError("not pinned yet")
        return original(task, split)

    monkeypatch.setattr(adapter, "records", records)
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="pinned test metadata is missing"):
        prepare(run, dataset, RunConfig(benchmark="toy", mode="demos", depth=False))
    assert not run.exists()


def test_simulator_build_recipe_is_copied_and_verified(tmp_path, toy, monkeypatch):
    import sys
    from cam.harness import prepare as preparation

    adapter, _ = toy
    dataset = make_dataset(tmp_path / "data", adapter)
    recipe = tmp_path / "uv.lock"
    recipe.write_text('version = 1\n')
    monkeypatch.setattr(preparation, "simulator_python", lambda adapter: Path(sys.executable))
    monkeypatch.setattr(preparation, "simulator_build_files", lambda adapter: [recipe])
    run = tmp_path / "run"
    preparation.prepare(run, dataset, RunConfig(benchmark="toy", mode="demos", depth=False))
    pinned = run / "inputs/simulator_build/uv.lock"
    assert pinned.read_text() == recipe.read_text()
    recipe.write_text('version = 2\n')
    preparation.validate_run(run)
    pinned.write_text('version = 3\n')
    with pytest.raises(ValueError, match="Pinned input changed"):
        preparation.validate_run(run)


def test_pool_subset_preserves_values_and_exposes_only_assigned_groups(tmp_path, toy):
    from cam.data.subset import subset
    from cam.harness.prepare import prepare

    adapter, _ = toy
    pool = make_dataset(tmp_path / "pool", adapter)
    original = read(pool / "manifest.json")
    source_file = pool / original["tasks"][TASK]["file"]
    before = sha256(source_file)
    output = tmp_path / "assigned"
    selected = subset(pool, output, count=2)
    assert selected["selection"]["actual_counts"] == {TASK: 2}
    assert selected["selection"]["source_manifest_sha256"] == sha256(pool / "manifest.json")
    item = selected["tasks"][TASK]
    keys = {f"episode_{r['episode']}" for r in item["episodes"]}
    with h5py.File(source_file) as src, h5py.File(output / item["file"]) as dst:
        assert set(dst) == keys and len(dst) < len(src)
        for key in keys:
            for field in dst[key]:
                np.testing.assert_array_equal(dst[key][field][()], src[key][field][()])
                assert isinstance(dst[key].get(field, getlink=True), h5py.HardLink)
    run = tmp_path / "run"
    prepare(run, output, RunConfig(benchmark="toy", mode="demos", depth=False, training_seed_count=2))
    with h5py.File(run / "tasks" / TASK / "demonstrations" / source_file.name) as public:
        assert set(public) == keys
    assert sha256(source_file) == before
    resumed = subset(pool, output, 2, resume=True)
    assert resumed["tasks"] == selected["tasks"]


def test_pool_subset_reports_shortfalls_and_refuses_tampering(tmp_path, toy):
    from cam.data.subset import subset

    pool = make_dataset(tmp_path / "pool", toy[0])
    output = tmp_path / "assigned"
    with pytest.raises(ValueError, match="only 3 available"):
        subset(pool, output, count=4)
    assert not output.exists()
    selected = subset(pool, output, 4, allow_fewer=True)
    assert selected["selection"]["actual_counts"] == {TASK: 3}
    with pytest.raises(FileExistsError):
        subset(pool, output, 4, allow_fewer=True)
    (output / selected["tasks"][TASK]["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Completed subset changed"):
        subset(pool, output, 4, allow_fewer=True, resume=True)


def test_disjoint_interactions_use_new_identities_once_and_preserve_demo_scope(tmp_path, toy):
    from cam.core.budget import Ledger, next_episode
    from cam.harness.prepare import prepare, training_schedule

    adapter, _ = toy
    dataset = make_dataset(tmp_path / "data", adapter)
    demos = read(dataset / "manifest.json")["tasks"][TASK]["episodes"]
    config = RunConfig(benchmark="toy", mode="both", both_protocol="v2", budget=3,
                       training_seed_count=3, training_seed_policy="disjoint_once", depth=False)
    schedule = training_schedule(adapter, TASK, demos, config)
    assert schedule == training_schedule(adapter, TASK, list(reversed(demos)), config)
    assert {r["seed"] for r in schedule}.isdisjoint(r["seed"] for r in demos + TEST)
    root = tmp_path / "run"
    prepare(root, dataset, config)
    task = root / "tasks" / TASK
    cfg = read(task / "state/config.json")
    ledger = Ledger(cfg["ledger"])
    assert cfg["train_records"] == schedule
    with h5py.File(next((task / "demonstrations").glob("*.h5"))) as h5:
        assert set(h5) == {f"episode_{r['episode']}" for r in demos}
    assert "disjoint from the assigned demonstrations; no repeats" in (task / "prompt.md").read_text()
    for i, row in enumerate(schedule):
        assert next_episode(cfg, i) == row["episode"]
        with pytest.raises(ValueError, match="next assigned"):
            ledger.reserve(cfg, demos[0]["episode"])
        attempt = ledger.reserve(cfg, row["episode"])
        ledger.update(attempt, steps=0, step_calls=0, status="harness_error" if i == 0 else "fail")
    assert len({r["episode"] for r in ledger.rows()}) == 3
    assert next_episode(cfg, 3) is None
    with pytest.raises(ValueError, match="exhausted"):
        ledger.reserve(cfg, schedule[0]["episode"])


def test_disjoint_schedule_rejects_shortage_and_filters_seed_aliases(tmp_path, toy, monkeypatch):
    from cam.harness.prepare import prepare, training_schedule

    adapter, _ = toy
    dataset = make_dataset(tmp_path / "data", adapter)
    demos = read(dataset / "manifest.json")["tasks"][TASK]["episodes"]
    config = RunConfig(benchmark="toy", mode="both", both_protocol="v2", budget=4,
                       training_seed_count=4, training_seed_policy="disjoint_once", depth=False)
    with pytest.raises(ValueError, match="only 3 disjoint"):
        prepare(tmp_path / "rejected", dataset, config)
    assert not (tmp_path / "rejected").exists()
    # New episode IDs are insufficient if their underlying seeds repeat demo/test seeds.
    from dataclasses import replace
    alias_rows = [{**r, "episode": r["episode"] + 100} for r in TRAIN] + [{"episode": 200, "seed": TEST[0]["seed"]}]
    monkeypatch.setattr(adapter, "records", lambda task, split: alias_rows if split == "train" else TEST)
    selected = training_schedule(adapter, TASK, demos, replace(config, budget=3, training_seed_count=3))
    assert {r["seed"] for r in selected}.isdisjoint(r["seed"] for r in demos + TEST)


def test_explicit_subset_copies_only_recorded_episode_seed_pairs(tmp_path, toy, monkeypatch):
    from cam.data import subset as module
    adapter, _ = toy
    source = make_dataset(tmp_path / 'pool', adapter)
    original = read(source / 'manifest.json')
    rows = original['tasks'][TASK]['episodes']
    selection = {'schema': 'cam-demo-selection/1', 'benchmark': 'toy', 'count': 2,
                 'tasks': {TASK: [{k: r[k] for k in ('episode', 'seed')} for r in rows[-2:]]}}
    path = tmp_path / 'selected_25.json'
    write(path, selection)
    real_validate = module.validate_dataset
    def validate_new_only(root, **kwargs):
        assert Path(root) != source
        return real_validate(root, **kwargs)
    monkeypatch.setattr(module, 'validate_dataset', validate_new_only)
    output = tmp_path / 'assigned'
    result = module.subset(source, output, count=2, selection=path, validate_source=False)
    assert result['tasks'][TASK]['episodes'] == rows[-2:]
    with h5py.File(output / original['tasks'][TASK]['file']) as h:
        assert set(h) == {f"episode_{r['episode']}" for r in rows[-2:]}
    assert sha256(source / original['tasks'][TASK]['file']) == original['tasks'][TASK]['sha256']
    selection['tasks'][TASK][0]['seed'] += 999
    write(path, selection)
    with pytest.raises(ValueError, match='episode/seed'):
        module.subset(source, tmp_path / 'bad', 2, selection=path, validate_source=False)
