"""Snapshot and queue successful-only RMBench RGB-D expert regeneration.

  .venv/bin/python scripts/prepare_rmbench_rgbd.py start --gpus 0 1 --workers 10 --allow-gpu-sharing
  .venv/bin/python scripts/prepare_rmbench_rgbd.py status JOB

500 released training identities; no test trajectories or coding client.
GPU sharing is opt-in; ten workers across two GPUs means exactly five per GPU.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def launch(job):
    from cam.io import write

    with (job / "supervisor.log").open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, str(job / "driver.py"), "_worker", str(job)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=job,
        )
    write(job / "process.json", {"pid": process.pid, "started": time.time()})
    print(job)
    print(f"Supervisor PID {process.pid}")


def start(args):
    import cam
    from cam.benchmarks.base import pin_source, source_hash
    from cam.benchmarks.rmbench import RMBench
    from cam.benchmarks.rmbench_rgbd_data import gpu_allocation
    from cam.harness.prepare import environment_snapshot, simulator_snapshot
    from cam.benchmarks.registry import simulator_python
    from cam.io import copy_tree, tree_hash, write

    gpus = args.gpus if args.gpus is not None else [args.gpu if args.gpu is not None else 1]
    allocation = gpu_allocation(args.workers, gpus)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    job = (ROOT / "outputs/rmbench_rgbd_preparation" / tag).resolve()
    data = (ROOT / "data/rmbench_rgbd" / tag).resolve()
    job.mkdir(parents=True)
    snapshot = job / "inputs/checkout"
    snapshot.mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, snapshot / name)
    copy_tree(Path(cam.__file__).parent, snapshot / "src/cam")
    adapter = RMBench()
    source = snapshot / "third_party/rmbench"
    assets = pin_source(adapter, source)
    if source_hash(RMBench(source)) != source_hash(adapter):
        raise ValueError("Benchmark snapshot hash mismatch")
    write(job / "inputs/environment.json", environment_snapshot())
    write(job / "inputs/simulator_environment.json", simulator_snapshot(simulator_python(adapter)))
    plan = {"created": time.time(), "tag": tag, "gpus": gpus, "workers": args.workers,
            "workers_by_gpu": allocation, "allow_gpu_sharing": args.allow_gpu_sharing,
            "reuse_captures": str(args.reuse_captures.resolve()) if args.reuse_captures else None,
            "simulator_python": str(simulator_python(adapter).absolute()),
            "supersedes": str(args.supersedes.resolve()) if args.supersedes else None,
            "source": str(source), "source_hash": source_hash(adapter), "asset_hashes": assets,
            "harness_hash": tree_hash(snapshot / "src/cam"),
            "raw": str((ROOT / "data/raw/rmbench").resolve()),
            "pool": str(data / "pool"), "initial": str(data / "initial_25"),
            "expected_by_task": {t: len(adapter.records(t, "train")) for t in adapter.tasks()},
            "method": "native expert regeneration on released training seeds; original dense paths are unavailable"}
    write(job / "plan.json", plan)
    shutil.copy2(Path(__file__), job / "driver.py")
    (job / "README.md").write_text(
        f"# RMBench RGB-D training dataset preparation\n\n"
        f"Created {tag}. The release supplies 50 training identities per task, 500 in total. "
        "The dense original trajectories are absent; the official expert is regenerated at those same seeds. "
        "Sampled state rows are never sent as action commands. Original files remain unchanged.\n\n"
        "Four aligned 320x240 RGB-D cameras; int16 optical-axis depth in millimetres (0 invalid), "
        "calibration, end-effector poses, joint targets and gripper state. RGB is rendered with depth, "
        "not borrowed from the original JPEG recordings. Native successful collections alone enter the pool; "
        "all failures and native trajectory paths stay in private jobs/. Counts are never padded.\n\n"
        f"GPUs {gpus}, {allocation} workers by GPU, {args.workers} total. "
        f"Sharing other GPU workloads is {'explicitly enabled with a free-memory check' if args.allow_gpu_sharing else 'disabled; waits for idle GPUs'}. "
        "One native successful RGB-D pilot is required before bulk dispatch and reused in the pool. "
        "Verified jobs start while later source files are checked. Completed prior captures may be carried "
        "forward with identity/hash validation and explicit provenance; neither successes nor failures are rerun. "
        "All-task pool followed by separate deterministic up-to-25-demo subsets. No learning or test evaluation.\n\n"
        f"Pool: `{data / 'pool'}`\n\nSubset: `{data / 'initial_25'}`\n\n"
        "Progress: status.json here and in the pool; supervisor.log; per-episode jobs/*/ep*/result.json. "
        "Final counts: pool/status.json and pool/manifest.json.\n"
    )
    write(job / "status.json", {"status": "queued", "updated": time.time(), "pool": plan["pool"]})
    launch(job)


def worker(job):
    sys.path.insert(0, str(job / "inputs/checkout/src"))
    from cam.benchmarks.rmbench_rgbd_data import extract
    from cam.data.subset import subset
    from cam.io import read, tree_hash, write

    plan = read(job / "plan.json")
    # The snapshot deliberately omits virtualenvs; use the pinned host interpreter.
    os.environ["CAM_RMBENCH_PYTHON"] = plan["simulator_python"]
    with (job / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if tree_hash(job / "inputs/checkout/src/cam") != plan["harness_hash"]:
                raise ValueError("Pinned preparation code changed")
            write(job / "status.json", {"status": "preparing", "pool": plan["pool"], "updated": time.time()})
            manifest = extract(plan["raw"], plan["pool"], count="all", workers=plan["workers"],
                               benchmark_source=plan["source"], gpus=plan["gpus"],
                               allow_gpu_sharing=plan["allow_gpu_sharing"], reuse=plan["reuse_captures"],
                               resume=Path(plan["pool"]).exists())
            available = [t for t, item in manifest["tasks"].items() if item["episodes"]]
            if available:
                subset(plan["pool"], plan["initial"], count=25, tasks=available, allow_fewer=True,
                       resume=Path(plan["initial"]).exists())
            counts = {t: len(item["episodes"]) for t, item in manifest["tasks"].items()}
            write(job / "status.json", {"status": "complete", "counts": counts, "total": sum(counts.values()),
                  "expected": sum(plan["expected_by_task"].values()), "updated": time.time()})
        except Exception:
            write(job / "status.json", {"status": "needs_attention", "error": traceback.format_exc(), "updated": time.time()})
            raise


def main():
    import json
    if sys.argv[1:2] == ["_worker"]:
        sys.path.insert(0, str(Path(sys.argv[2]).resolve() / "inputs/checkout/src"))
    from cam.io import read

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start_parser = sub.add_parser("start")
    device = start_parser.add_mutually_exclusive_group()
    device.add_argument("--gpu", type=int)
    device.add_argument("--gpus", type=int, nargs="+")
    start_parser.add_argument("--workers", type=int, default=4)
    start_parser.add_argument("--allow-gpu-sharing", action="store_true")
    start_parser.add_argument("--reuse-captures", type=Path)
    start_parser.add_argument("--supersedes", type=Path)
    for name in ("status", "resume", "_worker"):
        sub.add_parser(name).add_argument("job", type=Path)
    args = parser.parse_args()
    if args.command == "start":
        start(args)
    elif args.command == "_worker":
        worker(args.job.resolve())
    elif args.command == "resume":
        job = args.job.resolve()
        if (job / "process.json").exists():
            pid = read(job / "process.json")["pid"]
            if Path(f"/proc/{pid}").exists():
                raise SystemExit("Supervisor still exists; do not launch duplicate workers")
        launch(job)
    else:
        job = args.job.resolve()
        plan = read(job / "plan.json")
        status = {"job": str(job), "supervisor": read(job / "status.json"), "plan": plan}
        path = Path(plan["pool"]) / "status.json"
        if path.exists():
            status["collection"] = read(path)
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
