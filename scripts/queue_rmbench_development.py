"""Queue RGB-D demo-only development after the capture producer validates its pool.

The experiment/config and batch launcher are copied at queue time. The batch
pins the current harness and simulator source when data become ready, then uses
cam prepare for fresh per-task runs. No held-out evaluation is launched.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def pool_ready(producer, pool):
    """The producer removes its partial manifest only after full validation."""
    from cam.io import read

    state = read(producer / "status.json")["status"]
    ready = (pool / "manifest.json").is_file() and not (pool / "manifest.partial.json").exists()
    if state not in ("assembling", "complete"):
        raise RuntimeError(f"RMBench producer requires attention: {state}")
    return ready


def start(args):
    from cam.config import RunConfig
    from cam.io import read, sha256, write

    producer = args.producer.resolve()
    spec = read(args.plan)
    if set(spec["benchmarks"]) != {"rmbench"}:
        raise ValueError("This queue requires one RMBench experiment")
    item = spec["benchmarks"]["rmbench"]
    config = read(ROOT / item["config"])
    cfg = RunConfig.from_dict(config)
    if (cfg.benchmark, cfg.mode, cfg.budget, cfg.depth) != ("rmbench", "demos", 0, True):
        raise ValueError("Queue requires RGB-D demo-only with zero training rollouts")
    capture = read(producer / "plan.json")
    pool = Path(capture["pool"]).resolve()
    if [str((ROOT / p).resolve()) for p in item["datasets"]] != [str(pool)]:
        raise ValueError("Experiment must use this producer's full pool")
    pool_ready(producer, pool)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    job = ROOT / "outputs/rmbench_development_queue" / tag
    job.mkdir(parents=True)
    write(job / "config.json", config)
    item["config"] = str(job / "config.json")
    item["datasets"] = [str(pool)]
    item["raw"] = str((ROOT / item["raw"]).resolve())
    write(job / "experiment.json", spec)
    shutil.copy2(ROOT / "scripts/develop_batch.py", job / "develop_batch.py")
    shutil.copy2(Path(__file__), job / "driver.py")
    plan = {"created": time.time(), "root": str(ROOT), "producer": str(producer),
            "pool": str(pool), "workers": args.workers, "gpus": [0, 1],
            "python": sys.executable, "producer_plan_sha256": sha256(producer / "plan.json"),
            "pinned_files": {name: sha256(job / name) for name in
                             ("config.json", "experiment.json", "develop_batch.py", "driver.py")}}
    write(job / "plan.json", plan)
    (job / "README.md").write_text(
        f"# Queued RMBench RGB-D development: {tag}\n\n"
        "All ten tasks, target 50 demos each; actual assignment is 496 successful captures "
        "(49 battery_try, 49 blocks_ranking_try, 48 observe_and_pickup, 50 each elsewhere). "
        f"Claude Code / Opus 5.5, shared demo-only prompt, {args.workers} coding jobs, zero training rollouts.\n\n"
        f"Wait for `{producer}` to publish and validate `{pool}`. The capture producer removes "
        "manifest.partial.json after validation, before creating the separate historical 25-demo subset; "
        "this experiment does not wait for that subset. The experiment config and launcher are "
        "pinned here. Current harness and simulator source are pinned by the new development batch "
        "at launch; every task gets a fresh cam prepare run. No held-out tests are launched.\n\n"
        "See status.json for readiness or errors and batch.json for the launched batch. "
        "Failures are preserved and never automatically retried.\n"
    )
    write(job / "status.json", {"status": "waiting_for_validated_pool", "updated": time.time()})
    with (job / "supervisor.log").open("ab") as log:
        proc = subprocess.Popen([sys.executable, str(job / "driver.py"), "_worker", str(job)],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=ROOT)
    write(job / "process.json", {"pid": proc.pid, "started": time.time()})
    return {"queue": str(job), "pid": proc.pid, "status": "waiting_for_validated_pool"}


def worker(job):
    from cam.io import read, sha256, write

    plan = read(job / "plan.json")
    with (job / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, digest in plan["pinned_files"].items():
            if sha256(job / name) != digest:
                raise ValueError(f"Queued input changed: {name}")
        producer, pool = Path(plan["producer"]), Path(plan["pool"])
        while True:
            if sha256(producer / "plan.json") != plan["producer_plan_sha256"]:
                raise ValueError("Capture producer plan changed")
            if pool_ready(producer, pool):
                break
            write(job / "status.json", {"status": "waiting_for_validated_pool", "updated": time.time(),
                                       "producer": str(producer), "pool": str(pool)})
            time.sleep(30)
        write(job / "status.json", {"status": "launching", "updated": time.time()})
        spec = importlib.util.spec_from_file_location("queued_develop_batch", job / "develop_batch.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        launcher.ROOT = Path(plan["root"])
        result = launcher.start(argparse.Namespace(plan=job / "experiment.json",
                                                  workers=plan["workers"], gpus=plan["gpus"]))
        write(job / "batch.json", result)
        write(job / "status.json", {"status": "launched", "updated": time.time(), **result})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("start")
    p.add_argument("plan", type=Path)
    p.add_argument("--producer", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    commands.add_parser("_worker").add_argument("job", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        ROOT = Path(json.loads((args.job / "plan.json").read_text())["root"])
    sys.path.insert(0, str(ROOT / "src"))
    sys.dont_write_bytecode = True
    if args.command == "start":
        if args.workers < 1:
            parser.error("Use a positive worker count")
        print(json.dumps(start(args), indent=2))
    else:
        try:
            worker(args.job)
        except Exception:
            from cam.io import write
            write(args.job / "status.json", {"status": "needs_attention", "updated": time.time(),
                                             "error": traceback.format_exc()})
            raise
