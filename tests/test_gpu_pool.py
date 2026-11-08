"""Host GPU leases serialize simulators while coding sessions remain concurrent."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import threading
import time

from cam.harness.gpu import simulator_slot


def test_gpu_pool_excludes_reserved_gpu_and_serializes_each_device(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_SIMULATOR_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("CAM_SIMULATOR_GPUS", "0,1")
    with (tmp_path / "gpu_1.lock").open("a") as reserved:
        fcntl.flock(reserved, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with simulator_slot(1) as gpu:
            assert gpu == 0
    active, maximum = {0: 0, 1: 0}, {0: 0, 1: 0}
    guard = threading.Lock()

    def worker(index):
        with simulator_slot(index % 2) as gpu:
            with guard:
                active[gpu] += 1
                maximum[gpu] = max(maximum[gpu], active[gpu])
            time.sleep(0.03)
            with guard:
                active[gpu] -= 1
            return gpu

    with ThreadPoolExecutor(max_workers=8) as pool:
        devices = list(pool.map(worker, range(8)))
    assert set(devices) == {0, 1}
    assert maximum == {0: 1, 1: 1}
