"""Host-only interpreter wrapper: bounded native replay workers on multiple GPUs.

The extraction snapshot stays unchanged. Metadata-only Python invocations pass
straight through; simulator replay processes acquire one recorded GPU slot.
"""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def validate(config):
    gpus = config["gpus"]
    workers = config["workers_per_gpu"]
    if not gpus or len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError("Replay GPUs must be distinct nonnegative integers")
    if type(workers) is not int or workers < 1 or config["min_free_memory_mib"] < 1:
        raise ValueError("Replay workers and free-memory floor must be positive")


@contextmanager
def slot(config):
    validate(config)
    folder = Path(config["locks"])
    folder.mkdir(parents=True, exist_ok=True)
    slots = [(g, n) for n in range(config["workers_per_gpu"]) for g in config["gpus"]]
    while True:
        for gpu, number in slots:
            with (folder / f"gpu_{gpu}_{number}.lock").open("a") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
                free_mib = int(result.stdout.strip())
                if free_mib < config["min_free_memory_mib"]:
                    continue
                try:
                    yield gpu, number, free_mib
                finally:
                    fcntl.flock(lease, fcntl.LOCK_UN)
                return
        time.sleep(2)


def dispatch(config, argv, job_path):
    """Keep the lease while the native child runs; archive its original evidence."""
    archive = Path(config["archive"]) / job_path.name.removesuffix(".job.json")
    with slot(config) as (gpu, number, free_mib):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        record = {"gpu": gpu, "slot": number, "free_mib_at_start": free_mib,
                  "job": str(job_path), "dispatcher_pid": os.getpid(), "started": time.time(),
                  "command": [config["python"], *argv]}
        with subprocess.Popen(record["command"], env=env) as proc:
            record["native_pid"] = proc.pid
            write(archive / "dispatch.json", record)
            code = proc.wait()
        record.update(returncode=code, finished=time.time())
        write(archive / "dispatch.json", record)
        for suffix in (".job.json", ".result.json", ".log"):
            source = job_path.with_name(job_path.name.removesuffix(".job.json") + suffix)
            if source.exists():
                shutil.copy2(source, archive / source.name)
        return code


def main(config_path):
    config = json.loads(Path(config_path).read_text())
    validate(config)
    argv = sys.argv[1:]
    if "cam.interface.robomemarena.data" not in argv:
        # Environment/version capture does not initialize a simulator.
        os.execv(config["python"], [config["python"], *argv])
    job = Path(argv[-1]).resolve()
    raise SystemExit(dispatch(config, argv, job))
