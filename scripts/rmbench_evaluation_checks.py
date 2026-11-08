"""Pinned checks for demo-only RMBench evaluation and exclusive GPU admission.

Kept separate from benchmark-specific experiment launchers so a queued job can
snapshot the checks without importing another benchmark's development pipeline.
"""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())



def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)



def alive(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False



def queued_mikasa():
    """Submitted replay jobs can initialize GPUs even between native workers."""
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "python" in args and "cam.benchmarks.mikasa_robo_v2_data" in args:
            if " worker " in args or ("extract" in args and "gpus=" in args):
                found.append(int(process.name))
    return found



@contextmanager
def gpu_lease(job, gpu, phase, allow_sharing=False):
    """Coordinate with the existing RGB-D collection queue and GPU users."""
    if allow_sharing:
        raise ValueError("RMBench evaluation requires exclusive GPU admission")
    lock_root = Path(os.environ.get("CAM_RGBD_LOCK_DIR", "/tmp/cam-rmbench-rgbd-locks"))
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / f"gpu_{gpu}.lock").open("a") as lock:
        while True:
            blockers = queued_mikasa()
            try:
                if blockers:
                    raise BlockingIOError
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                write(job / "status.json", {"status": "waiting_for_gpu", "phase": phase,
                      "gpu": gpu, "blocking_mikasa_pids": blockers, "updated": time.time()})
                time.sleep(20)
                continue
            result = subprocess.run(["nvidia-smi", "-i", str(gpu),
                "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True)
            if result.stdout.strip():
                fcntl.flock(lock, fcntl.LOCK_UN)
                write(job / "status.json", {"status": "waiting_for_gpu", "phase": phase,
                      "gpu": gpu, "blocking_processes": result.stdout.splitlines(), "updated": time.time()})
                time.sleep(20)
                continue
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            return



def frozen_policy(root, task):
    """Fail closed before any held-out work if demo-only learning is unfinished."""
    root = Path(root)
    config = read(root / "run.json")["config"]
    folder = root / "tasks" / task
    completion = read(folder / "logs/completion.json")
    frozen = read(folder / "policy/freeze.json")
    if config["mode"] != "demos" or config["budget"] != 0 or completion["status"] != "finished":
        raise ValueError(f"Demo-only development is unfinished: {task}")
    if frozen["training"]["attempts"] != 0:
        raise ValueError(f"Unexpected training interactions: {task}")
    # The harness performs its own exact checkpoint/program integrity validation.
    from cam.core.budget import program_hash
    if program_hash(folder / "policy") != frozen["program_hash"]:
        raise ValueError(f"Frozen policy changed: {task}")
    return frozen



def score_summary(root, task):
    root = Path(root)
    expected = read(root / "run.json")["test_episodes"][task]
    frozen = read(root / "tasks" / task / "policy/freeze.json")
    records = []
    infrastructure = []
    for episode in expected:
        folder = root / "tasks" / task / "rollouts/test/R000" / f"ep{episode:03d}"
        if not folder.exists():
            continue
        if not (folder / "result.json").exists():
            infrastructure.append(episode)
            continue
        result = read(folder / "result.json")
        if result["program_hash"] != frozen["program_hash"]:
            raise ValueError(f"Tested policy differs from frozen policy: {task}/{episode}")
        if result["episode"] != episode or result["task"] != task or result["split"] != "test":
            raise ValueError(f"Test identity differs from assignment: {task}/{episode}")
        if not result["scorable"]:
            infrastructure.append(episode)
        else:
            records.append(result)
    successes = sum(bool(r["success"]) for r in records)
    return {"task": task, "run": str(root), "expected": len(expected),
            "scored": len(records), "successes": successes,
            "complete": len(records) == len(expected),
            "tsr": successes / len(expected) if len(records) == len(expected) else None,
            "infrastructure_episodes": infrastructure}
