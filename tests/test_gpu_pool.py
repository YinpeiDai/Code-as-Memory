"""Host GPU leases serialize simulators while coding sessions remain concurrent."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import threading
import time

from cam.harness.gpu import simulator_slot


def test_memory_admission_waits_and_staggers_shared_launches(tmp_path, monkeypatch):
    from cam.harness import gpu

    monkeypatch.setenv("CAM_GPU_ADMISSION_DIR", str(tmp_path))
    monkeypatch.setenv("CAM_GPU_MIN_FREE_MIB", "8192")
    monkeypatch.setenv("CAM_GPU_LAUNCH_SPACING_SECONDS", "5")
    clock, sleeps = [100.0], []
    available = iter([1024, 4096, 16384, 16384, 16384])
    monkeypatch.setattr(gpu, "free_mib", lambda device: next(available))
    monkeypatch.setattr(gpu.time, "time", lambda: clock[0])

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(gpu.time, "sleep", sleep)
    gpu.wait_for_admission(0)
    assert sleeps == [5, 5]
    # A second process sees the persisted launch time; another GPU remains independent.
    gpu.wait_for_admission(0)
    assert sleeps == [5, 5, 5]
    gpu.wait_for_admission(1)
    assert sleeps == [5, 5, 5]
    with (tmp_path / "gpu_0.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_unconfigured_admission_needs_no_gpu_probe(monkeypatch):
    from cam.harness import gpu

    monkeypatch.delenv("CAM_GPU_ADMISSION_DIR", raising=False)
    monkeypatch.setattr(gpu, "free_mib", lambda device: (_ for _ in ()).throw(AssertionError("unexpected probe")))
    gpu.wait_for_admission(0)


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
