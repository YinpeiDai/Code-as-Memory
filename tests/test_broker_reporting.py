import h5py
import numpy as np
import pytest
from cam.interface.robomme.observations import observations, KEYS
from cam.interface.robomme.adapter import RoboMME
from cam.core.history import freeze
from cam.harness.broker import execute
from cam.io import read, write
from cam.reporting.report import build, matched_efficiency, metrics
from cam.reporting.site import build_site


class Adapter(RoboMME):
    name = "synthetic"
    observations = staticmethod(observations)

    def __init__(self, failure=False):
        self.failure = failure

    def make_environment(self, *args):
        if self.failure:
            raise RuntimeError("Synthetic GPU initialization failure")
        return Environment()


class Environment:
    def obs(self):
        frame = {
            "front_rgb": np.zeros((16, 16, 3), np.uint8),
            "wrist_rgb": np.zeros((16, 16, 3), np.uint8),
            "front_extrinsic": np.zeros((3, 4)),
            "wrist_extrinsic": np.zeros((3, 4)),
            "tcp": np.zeros(6),
            "joints": np.zeros(7),
            "fingers": np.zeros(2),
        }
        result = {KEYS[k]: [v] for k, v in frame.items()}
        result.update(
            front_depth_list=[np.full((16, 16), 1234, np.int16)],
            wrist_depth_list=[np.full((16, 16), 400, np.int16)],
        )
        return result

    def reset(self):
        return self.obs(), {
            "execution_total_control_steps": 0,
            "task_goal": "Synthetic observation test",
            "front_camera_intrinsic": np.eye(3),
            "wrist_camera_intrinsic": np.eye(3),
        }

    def step(self, action):
        return (
            self.obs(),
            0,
            True,
            False,
            {
                "execution_total_control_steps": 1,
                "status": "success",
                "hidden_object_pose": "secret",
            },
        )

    def close(self):
        pass


@pytest.mark.isolation
def test_broker_saves_observations_and_native_score(run_factory, tmp_path):
    root = run_factory()
    plan = read(root / "run.json")
    task = root / "tasks/BinFill"
    result = execute(
        Adapter(),
        tmp_path,
        "BinFill",
        0,
        "train",
        task / "program",
        root / "runtime",
        tmp_path / "episode",
        plan["config"],
    )
    assert (
        result["status"] == "success"
        and result["steps"] == 1
        and result["api_calls"] == 0
    )
    assert result["video_frames"] == 2
    with h5py.File(tmp_path / "episode/observations.h5") as h5:
        assert h5["front_depth"][1, 0, 0] == 1234
        assert "hidden_object_pose" not in h5
    bad = execute(
        Adapter(True),
        tmp_path,
        "BinFill",
        0,
        "train",
        task / "program",
        root / "runtime",
        tmp_path / "failed",
        plan["config"],
    )
    assert bad["status"] == "harness_error" and not bad["scorable"]
    assert "GPU" not in bad["error"]


@pytest.mark.isolation
@pytest.mark.parametrize("split", ["train", "test"])
def test_cpu_backend_is_forwarded_and_recorded_for_both_splits(run_factory, tmp_path, split):
    class CPUAdapter(Adapter):
        name = "mikasa_robo_v2"

        def make_environment(self, metadata, task, episode, actual_split, depth, *, physics_backend):
            assert actual_split == split and physics_backend == "cpu"
            env = Environment()
            env.physics_backend = physics_backend
            return env

    root = run_factory()
    config = {**read(root / "run.json")["config"], "physics_backend": "cpu"}
    result = execute(CPUAdapter(), tmp_path, "BinFill", 0, split,
                     root / "tasks/BinFill/program", root / "runtime", tmp_path / "episode", config)
    assert result["status"] == "success"
    assert result["physics_backend"] == "cpu"


def test_exact_rates_partial_denominators_and_paired_efficiency():
    rows = [
        {
            "task": "A",
            "episode": 0,
            "success": True,
            "scorable": True,
            "steps": 10,
            "decisions": 1,
            "status": "success",
        },
        {
            "task": "A",
            "episode": 1,
            "success": False,
            "scorable": False,
            "steps": 0,
            "decisions": 0,
            "status": "harness_error",
        },
    ]
    m = metrics(rows, 3)
    assert (
        m["success_rate"] is None
        and m["observed_success_rate"] == 1
        and not m["complete"]
    )
    assert m["infrastructure_errors"] == 1
    other = [{**rows[0], "steps": 30}, {**rows[0], "episode": 2, "steps": 999}]
    p = matched_efficiency({"R001": rows, "R005": other})[0]
    assert (p["matched_successes"], p["left_mean_steps"], p["right_mean_steps"]) == (
        1,
        10,
        30,
    )


@pytest.mark.isolation
@pytest.mark.parametrize("query", ["step", "dry_run"])
def test_tool_timeout_ends_episode_without_reusing_pending_reply(run_factory, tmp_path, monkeypatch, query):
    import time
    from cam.harness import broker

    class SlowAdapter(Adapter):
        def advance(self, *args, **kwargs):
            time.sleep(0.2)
            result = super().advance(*args, **kwargs)
            result.status = "running"
            return result

        def dry_run(self, *args):
            time.sleep(0.2)
            return {"reachable": True}

    root = run_factory(max_decisions=3)
    task = root / "tasks/BinFill"
    if query == "dry_run":
        policy = task / "program/tools/policy.py"
        # Use an explicit six-component kinematic query.
        policy.write_text('''def perceive_decide(obs, goal, memory, last):
    """Select a query."""
    return {"memory": {}, "act": {"tool": "act_query", "args": {}}}
def act_query(robot):
    """Wait for a slow kinematic reply."""
    return robot.dry_run(0, 0, 0, 0, 0, 0)
''')
    original = broker.command

    def short_tool_limit(program, runtime, argv, **kwargs):
        return original(program, runtime, ["python3.11", "-c",
            "from cam.core.tools import TIME_LIMIT; TIME_LIMIT['act']=0.05; "
            "from cam.runtime.policy_worker import main; main()"], **kwargs)

    monkeypatch.setattr(broker, "command", short_tool_limit)
    result = execute(SlowAdapter(), tmp_path, "BinFill", 0, "test", task / "program",
                     root / "runtime", tmp_path / "timed_out", read(root / "run.json")["config"])
    assert result["status"] == "policy_timeout" and result["scorable"]
    assert result["decisions"] == 1
    assert result["step_calls"] + result["dry_run_calls"] == 1


@pytest.mark.isolation
@pytest.mark.parametrize("failure", [True, False])
def test_wall_timer_does_not_relabel_simulator_failure(run_factory, tmp_path, monkeypatch, failure):
    from cam.harness import broker

    timers = []
    class Timer:
        def __init__(self, seconds, callback):
            self.callback = callback
            timers.append(self)
        def start(self): pass
        def cancel(self): pass
        def join(self): pass

    class ExpiringAdapter(Adapter):
        def advance(self, *args, **kwargs):
            timers[0].callback()
            if failure:
                raise RuntimeError("synthetic simulator failure at deadline")
            result = super().advance(*args, **kwargs)
            result.status = "running"
            return result

    root = run_factory()
    monkeypatch.setattr(broker.threading, "Timer", Timer)
    folder = tmp_path / "deadline"
    result = execute(ExpiringAdapter(), tmp_path, "BinFill", 0, "test",
                     root / "tasks/BinFill/program", root / "runtime", folder,
                     read(root / "run.json")["config"])
    assert result["status"] == ("harness_error" if failure else "policy_timeout")
    assert result["scorable"] is not failure
    if failure:
        assert "synthetic simulator failure" in (folder / "private_broker_error.txt").read_text()


def test_broker_initialization_failure_consumes_attempt(run_factory, monkeypatch):
    from cam.harness import broker
    from cam.core.budget import Ledger

    root = run_factory("interaction", 1)
    monkeypatch.setattr(broker, "get_adapter", lambda *args: Adapter(True))
    result = broker.run_episode(root, "BinFill", 0, "train")
    cfg = read(root / "tasks/BinFill/state/config.json")
    rows = Ledger(cfg["ledger"]).rows()
    assert len(rows) == 1 and rows[0]["status"] == "harness_error"
    assert result["attempt"] == 1 and not result["scorable"]


def test_report_and_site_from_new_run_only(run_factory, dataset, tmp_path):
    root = run_factory()
    task = root / "tasks/BinFill"
    frozen = freeze(read(task / "state/config.json"))
    plan = read(root / "run.json")
    for i, difficulty in enumerate(("easy", "medium", "hard")):
        write(
            task / "rollouts/test/R000" / f"ep{i:03d}/result.json",
            {
                "task": "BinFill",
                "episode": i,
                "difficulty": difficulty,
                "split": "test",
                "maximum_rollouts": 0,
                "program_hash": frozen["program_hash"],
                "benchmark_hash": plan["benchmark_hash"],
                "success": i != 2,
                "scorable": True,
                "status": "success" if i != 2 else "fail",
                "steps": 10 + i,
                "decisions": 1,
                "api_calls": 0,
                "api_cost_usd": 0,
            },
        )
    result = build(root)
    assert result["settings"][0]["success_rate"] == 2 / 3
    assert result["settings"][0]["expected"] == 3
    assert result["cost"]["reported_cost_usd"] is None
    build_site([root], [dataset[0]], tmp_path / "site")
    exported = read(tmp_path / "site/data.json")
    assert exported["runs"][0]["settings"][0]["successes"] == 2
    paths = read(tmp_path / "site/.artifacts.json").values()
    assert not any(
        "ledger" in path or "private_" in path or "/raw/" in path for path in paths
    )
