"""Held-out work requires a finished frozen policy; scored failures stay scored."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from cam.core.budget import program_hash
from cam.io import read, write

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rmbench_evaluation", ROOT / "scripts/queue_rmbench_evaluation.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def prepared(tmp_path):
    rows = {}
    for task in ("First", "Second"):
        root = tmp_path / task
        folder = root / "tasks" / task
        write(root / "run.json", {"config": {"benchmark": "rmbench", "depth": True, "mode": "demos", "budget": 0},
                                  "test_episodes": {task: [0, 1]}})
        write(folder / "logs/completion.json", {"status": "finished"})
        policy = folder / "policy"
        write(policy / "task.json", {})
        (policy / "policy.md").write_text("Fixture policy")
        (policy / "tools").mkdir()
        (policy / "tools/policy.py").write_text("x = 1\n")
        write(policy / "freeze.json", {"training": {"attempts": 0}, "program_hash": program_hash(policy)})
        rows[task] = {"task": task, "benchmark": "rmbench", "run": str(root), "status": "finished"}
    return {"status": "finished", "tasks": rows}


@pytest.mark.parametrize("fault", [None, "running", "missing_task", "unfinished", "changed_policy", "training", "rgb", "test_count"])
def test_all_task_barrier(tmp_path, fault):
    state = prepared(tmp_path)
    root = Path(state["tasks"]["Second"]["run"])
    checks = evaluation.load_checks(ROOT / "scripts/rmbench_evaluation_checks.py")
    if fault == "running":
        state["status"] = "running"
    elif fault == "missing_task":
        state["tasks"].pop("Second")
    elif fault == "unfinished":
        write(root / "tasks/Second/logs/completion.json", {"status": "failed"})
    elif fault == "changed_policy":
        (root / "tasks/Second/policy/tools/policy.py").write_text("x = 2\n")
    elif fault == "training":
        path = root / "tasks/Second/policy/freeze.json"
        frozen = read(path)
        frozen["training"]["attempts"] = 1
        write(path, frozen)
    elif fault in ("rgb", "test_count"):
        plan = read(root / "run.json")
        if fault == "rgb":
            plan["config"]["depth"] = False
        else:
            plan["test_episodes"]["Second"].pop()
        write(root / "run.json", plan)
    if fault:
        with pytest.raises(ValueError):
            evaluation.ready_policies(state, {"First": 2, "Second": 2}, checks)
    else:
        roots, frozen = evaluation.ready_policies(state, {"First": 2, "Second": 2}, checks)
        assert len(roots) == len(frozen) == 2


def test_final_rate_requires_every_assigned_test():
    expected = {"First": 100, "Second": 100}
    results = {"First": {"scored": 100, "successes": 80, "complete": True, "infrastructure_episodes": []}}
    partial = evaluation.aggregate(results, expected)
    assert partial["scored"] == 100 and partial["expected"] == 200
    assert partial["success_rate"] is None and not partial["complete"]
    results["Second"] = {"scored": 99, "successes": 90, "complete": False, "infrastructure_episodes": [7]}
    assert evaluation.aggregate(results, expected)["unresolved_infrastructure"] == 1
    results["Second"].update(scored=100, complete=True, infrastructure_episodes=[])
    final = evaluation.aggregate(results, expected)
    assert final["complete"] and final["success_rate"] == 170 / 200


@pytest.mark.parametrize("peer_status", ["waiting_data", "preparing", "learning", "failed"])
def test_per_task_admission_does_not_wait_for_unfinished_peers(tmp_path, peer_status):
    state = prepared(tmp_path)
    state["status"] = "running"
    state["tasks"]["Second"]["status"] = peer_status
    # Even a pre-existing freeze does not override an unfinished learner status.
    checks = evaluation.load_checks(ROOT / "scripts/rmbench_evaluation_checks.py")
    roots, frozen = evaluation.ready_policies(state, {"First": 2, "Second": 2}, checks, per_task=True)
    assert set(roots) == set(frozen) == {"First"}


def test_per_task_admission_still_rejects_unfinished_completion(tmp_path):
    state = prepared(tmp_path)
    state["status"] = "running"
    state["tasks"]["Second"]["status"] = "learning"
    root = Path(state["tasks"]["First"]["run"])
    write(root / "tasks/First/logs/completion.json", {"status": "failed"})
    checks = evaluation.load_checks(ROOT / "scripts/rmbench_evaluation_checks.py")
    with pytest.raises(ValueError, match="unfinished"):
        evaluation.ready_policies(state, {"First": 2, "Second": 2}, checks, per_task=True)


def test_per_task_admission_still_rejects_changed_frozen_policy(tmp_path):
    state = prepared(tmp_path)
    state["status"] = "running"
    state["tasks"]["Second"]["status"] = "learning"
    root = Path(state["tasks"]["First"]["run"])
    (root / "tasks/First/policy/tools/policy.py").write_text("x = 2\n")
    checks = evaluation.load_checks(ROOT / "scripts/rmbench_evaluation_checks.py")
    with pytest.raises(ValueError, match="changed"):
        evaluation.ready_policies(state, {"First": 2, "Second": 2}, checks, per_task=True)


def test_strict_gpu_admission_waits_for_mikasa_and_other_compute(tmp_path, monkeypatch):
    checks = evaluation.load_checks(ROOT / "scripts/rmbench_evaluation_checks.py")
    monkeypatch.setenv("CAM_RGBD_LOCK_DIR", str(tmp_path / "locks"))
    blockers = iter([[123], [], []])
    monkeypatch.setattr(checks, "queued_mikasa", lambda: next(blockers))
    processes = iter(["456, python\n", ""])
    monkeypatch.setattr(checks.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=next(processes)))
    sleeps = []
    monkeypatch.setattr(checks.time, "sleep", lambda n: sleeps.append(n))
    with checks.gpu_lease(tmp_path, 0, "evaluation", allow_sharing=False):
        assert len(sleeps) == 2


def test_evaluation_rejects_shared_gpu_admission(tmp_path):
    checks = evaluation.load_checks(ROOT / "scripts/rmbench_evaluation_checks.py")
    with pytest.raises(ValueError, match="exclusive GPU"):
        with checks.gpu_lease(tmp_path, 0, "evaluation", allow_sharing=True):
            pytest.fail("Shared GPU was admitted")
