"""Host-only GPU leases shared by coding sessions and data preparation."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time


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
