"""Host-only GPU leases shared by coding sessions and data preparation."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import math
import subprocess
import time


def free_mib(gpu):
    probe = subprocess.run([
        "nvidia-smi", "--id=" + str(gpu), "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=15, check=True)
    return int(probe.stdout.strip())


def wait_for_admission(gpu):
    """Stagger process launches and wait for free memory before consuming an episode.

    The campaign supplies this host-only directory to both training and evaluation.
    The admission lock is released before execution; episode workers stay concurrent.
    """
    directory = os.environ.get("CAM_GPU_ADMISSION_DIR")
    if not directory:
        return
    minimum = float(os.environ.get("CAM_GPU_MIN_FREE_MIB", "8192"))
    spacing = float(os.environ.get("CAM_GPU_LAUNCH_SPACING_SECONDS", "5"))
    if not all(math.isfinite(v) and v >= 0 for v in (minimum, spacing)):
        raise ValueError("GPU admission thresholds must be finite and nonnegative")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"gpu_{gpu}.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        stamp = root / f"gpu_{gpu}.last_launch"
        if stamp.exists():
            delay = max(0, min(spacing, float(stamp.read_text()) + spacing - time.time()))
            if delay:
                time.sleep(delay)
        while free_mib(gpu) < minimum:
            time.sleep(5)
        stamp.write_text(str(time.time()))


@contextmanager
def simulator_slot(preferred):
    """Wait before simulator initialization; waiting consumes no training attempt.

    The batch supervisor supplies this private directory and GPU allowlist.
    Coding sandboxes receive neither the environment nor the lock directory.
    """
    directory = os.environ.get("CAM_SIMULATOR_LOCK_DIR")
    if not directory:
        yield preferred
        return
    gpus = [int(v) for v in os.environ.get("CAM_SIMULATOR_GPUS", str(preferred)).split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("Invalid host simulator GPU allowlist")
    gpus.sort(key=lambda gpu: gpu != preferred)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    leases = [(gpu, (root / f"gpu_{gpu}.lock").open("a")) for gpu in gpus]
    try:
        while True:
            for gpu, lease in leases:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    yield gpu
                finally:
                    fcntl.flock(lease, fcntl.LOCK_UN)
                return
            time.sleep(0.1)
    finally:
        for _, lease in leases:
            lease.close()
