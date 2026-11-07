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
