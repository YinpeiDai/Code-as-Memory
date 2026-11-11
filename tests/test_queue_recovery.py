import threading
import pytest
from cam.core.budget import Ledger
from cam.harness.jobs import gpu_slots, run_queue
from cam.harness.recover import recover_learning, recover_test
from cam.io import read, write


def test_sixteen_slots_share_one_queue(tmp_path):
    barrier = threading.Barrier(16)
    lock = threading.Lock()
    counts = {0: 0, 1: 0}
    maxima = counts.copy()

    def worker(job, gpu, folder):
        with lock:
            counts[gpu] += 1
            maxima[gpu] = max(maxima[gpu], counts[gpu])
        barrier.wait(timeout=10)
        with lock:
            counts[gpu] -= 1
        return {"status": "finished"}

    result = run_queue(
        tmp_path, "test", [(str(i),) for i in range(16)], gpu_slots([0, 1], 8), worker
    )
    assert result["status"] == "finished" and maxima == {0: 8, 1: 8}


def test_recovery_preserves_training_budget_and_scored_tests(run_factory):
    root = run_factory("interaction", 2)
    folder = root / "tasks/BinFill"
    cfg = read(folder / "state/config.json")
    ledger = Ledger(cfg["ledger"])
    ledger.reserve(cfg, 0)
    recover_learning(root, "BinFill", "worker interrupted in synthetic check")
    assert len(ledger.rows()) == 1 and ledger.rows()[0]["status"] == "interrupted"
    ledger.reserve(cfg, 3)
    path = folder / "rollouts/test/R001/ep000/result.json"
    write(path, {"scorable": True, "status": "fail"})
    with pytest.raises(ValueError, match="cannot be retried"):
        recover_test(root, "BinFill", 0, 1, "test")
    write(path, {"scorable": False, "status": "harness_error"})
    result = recover_test(root, "BinFill", 0, 1, "synthetic infrastructure repair")
    assert not path.exists() and result["archived"]


@pytest.mark.parametrize("crash", ["legacy_row", "new_row", "before_folder", "before_status_save"])
def test_sanity_retries_crashed_workers_without_replacing_scored_failures(run_factory, tmp_path, monkeypatch, crash):
    from pathlib import Path
    from cam.core.history import freeze
    from cam.harness import sanity

    root = run_factory()
    task = root / "tasks/BinFill"
    freeze(read(task / "state/config.json"))
    folder = task / "rollouts/test/R000/ep000"
    if crash != "before_folder":
        folder.mkdir(parents=True)
        (folder / "policy_stderr.txt").write_text("original worker crash")
    scored_path = task / "rollouts/test/R000/ep001/result.json"
    scored = {"episode": 1, "status": "fail", "success": False, "scorable": True, "step_calls": 2}
    write(scored_path, scored)
    original = scored_path.read_bytes()
    crashed = {"episode": 0, "status": "worker_error", "success": False, "runnable": False}
    if crash == "new_row":
        crashed.update(scorable=False, result=str(folder / "result.json"))
    results = [] if crash == "before_status_save" else [crashed]
    results.append({**scored, "runnable": True, "result": str(scored_path)})
    batch = tmp_path / "sanity/batch"
    key = "robomme/demo_only_3/BinFill"
    write(batch / "status.json", {
        "schema": "cam-solution-check/2", "purpose": "published_policy_execution_check",
        "status": "needs_attention", "expected": 2, "expected_policies": 1,
        "episodes_per_policy": 2, "tasks": {key: {
            "benchmark": "robomme", "setting": "demo_only_3", "task": "BinFill",
            "status": "failed", "episodes": [0, 1], "results": results, "run": str(root),
        }},
    })
    calls = []
    def rerun(run, argv, *args, **kwargs):
        calls.append(argv)
        assert argv[3] == "0" and not folder.exists()
        write(folder / "result.json", {**scored, "episode": 0})
        return 0
    monkeypatch.setattr(sanity, "subprocess_job", rerun)
    monkeypatch.setattr(sanity, "free_mib", lambda gpu: 100000)
    monkeypatch.setattr(sanity.time, "sleep", lambda seconds: None)
    kwargs = dict(resume=batch, episodes=2, gpus=(0,), workers_per_gpu=1)
    sanity.check_solutions(tmp_path / "unused", batch.parent, **kwargs)
    assert not calls
    result = sanity.check_solutions(tmp_path / "unused", batch.parent, retry_infrastructure=True, **kwargs)
    assert result["status"] == "complete" and result["passed"] == 2
    assert len(calls) == 1 and scored_path.read_bytes() == original
    row = read(batch / "status.json")["tasks"][key]
    receipt = row["infrastructure_retries"][0]
    archived = Path(receipt["archived"])
    assert (archived / "recovery.json").exists()
    if crash != "before_folder":
        assert (archived / "policy_stderr.txt").read_text() == "original worker crash"
