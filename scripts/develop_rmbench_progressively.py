"""Assemble immutable RMBench captures on local disk and develop ready tasks.

Two local assembly slots feed eight demo-only coding slots. All files retain
hash/schema/identity checks; five seeded random episodes per task receive extra
RGB-D inspection. Original captures, pool assembly and old schedules are retained.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def inspect_samples(dataset, task, item, count=5):
    import h5py
    import numpy as np
    from PIL import Image, ImageOps, ImageDraw
    from cam.benchmarks.rmbench import CAMERAS
    from cam.io import write

    seed = int(hashlib.sha256(f"rmbench/rgbd-inspection/1/{task}".encode()).hexdigest()[:16], 16)
    chosen = random.Random(seed).sample(sorted(r["episode"] for r in item["episodes"]), min(count, len(item["episodes"])))
    rows, tiles = [], []
    with h5py.File(dataset / item["file"], "r") as h5:
        for episode in chosen:
            group = h5[f"episode_{episode}"]
            length = len(group["step"])
            indices = sorted({0, length // 2, length - 1})
            for index in indices:
                row = {"episode": episode, "frame": index, "cameras": {}}
                strip = Image.new("RGB", (160 * len(CAMERAS) * 2, 140))
                draw = ImageDraw.Draw(strip)
                for c, camera in enumerate(CAMERAS):
                    rgb = group[camera + "_rgb"][index]
                    depth = group[camera + "_depth"][index]
                    if rgb.shape[:2] != depth.shape or not (depth >= 0).all() or not (depth > 0).any():
                        raise ValueError(f"Invalid sampled RGB-D: {task}/{episode}/{index}/{camera}")
                    for name in group:
                        if group[name].dtype.kind == "f" and not np.isfinite(group[name][index]).all():
                            raise ValueError(f"Nonfinite sampled field: {task}/{episode}/{name}")
                    valid = depth[depth > 0]
                    lo, hi = np.percentile(valid, [2, 98])
                    gray = np.clip((depth.astype(float) - lo) * 255 / max(hi - lo, 1), 0, 255).astype('uint8')
                    gray[depth == 0] = 0
                    color = ImageOps.colorize(Image.fromarray(gray), "black", "cyan")
                    strip.paste(Image.fromarray(rgb).resize((160, 120)), (c * 320, 20))
                    strip.paste(color.resize((160, 120)), (c * 320 + 160, 20))
                    draw.text((c * 320 + 3, 3), f"ep{episode} f{index} {camera}: RGB / depth", fill='white')
                    row["cameras"][camera] = {"valid_depth_fraction": float((depth > 0).mean()),
                                              "depth_mm_p02_p98": [float(lo), float(hi)]}
                rows.append(row)
                tiles.append(strip)
    preview = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)))
    y = 0
    for tile in tiles:
        preview.paste(tile, (0, y))
        y += tile.height
    preview.save(dataset / "sample_inspection.png")
    result = {"task": task, "seed": seed, "selected_episodes": chosen,
              "rule": "seeded random distinct successful episodes; first/middle/last frames, all four cameras",
              "simulator_replays": 0, "frames": rows, "passed": True}
    write(dataset / "sample_inspection.json", result)
    return result


def materialize_task(job, plan, task, update=lambda **changes: None):
    from cam.benchmarks import rmbench_rgbd_data as capture
    from cam.benchmarks.rmbench import RMBench
    from cam.io import read, write

    destination = job / "datasets" / task
    destination.mkdir(parents=True)
    pool = Path(plan["capture_pool"])
    records = RMBench(plan["benchmark_source"]).records(task, "train")
    copied = []
    for index, record in enumerate(records):
        ep = int(record["episode"])
        original = pool / "jobs" / task / f"ep{ep:05d}"
        source_job = read(original / "job.json")
        result = read(original / "result.json")
        if any(int(row[k]) != int(record[k]) for row in (source_job, result) for k in ("episode", "seed")):
            raise ValueError(f"Capture identity changed: {task}/{ep}")
        if source_job["task"] != task or result["protocol"] != capture.PROTOCOL:
            raise ValueError(f"Capture protocol or task changed: {task}/{ep}")
        staged = destination / "jobs" / task / f"ep{ep:05d}"
        write(staged / "job.json", source_job)
        write(staged / "result.json", result)
        if result["success"]:
            if result["status"] != "success":
                raise ValueError("Successful flag differs from capture outcome")
            source, target = original / "observations.h5", staged / "observations.h5"
            before = source.stat()
            digest = hashlib.sha256()
            with source.open('rb') as src, target.open('xb') as dst:
                for block in iter(lambda: src.read(8 * 1024**2), b''):
                    digest.update(block)
                    dst.write(block)
            after = source.stat()
            if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                    or digest.hexdigest() != result["observations_sha256"]):
                raise ValueError(f"Capture changed during local staging: {task}/{ep}")
            copied.append({"episode": ep, "source": str(source), "sha256": digest.hexdigest(), "bytes": before.st_size})
        update(status="staging_captures", staged_identities=index + 1, total_identities=len(records))
    update(status="assembling_data")
    item = capture.assemble(destination, task, records)
    expected = plan["demo_counts"][task]
    if len(item["episodes"]) != expected:
        raise ValueError(f"{task}: expected {expected} successful demos, found {len(item['episodes'])}")
    manifest = deepcopy(read(job / "capture_manifest.json"))
    manifest.update(created=time.time(), requested_tasks=[task], tasks={task: item})
    manifest["materialization"] = {"source_pool": str(pool), "source_capture_job": plan["capture_job"],
                                   "scheduler": str(job), "copied_capture_hashes": copied,
                                   "rule": "unchanged successful episode groups, locally assembled; no new simulator execution"}
    write(destination / "manifest.json", manifest)
    update(status="validating_data")
    capture.validate_dataset(destination)
    sampled = inspect_samples(destination, task, item)
    write(destination / "ready.json", {"ready": True, "count": expected, "h5_sha256": item["sha256"],
                                       "all_capture_hashes_checked": True, "all_schema_identity_checks": True,
                                       "sampled_episodes": sampled["selected_episodes"]})
    # Delete only redundant local staging copies after successful validation.
    # Source captures and all result/job receipts remain intact.
    for entry in copied:
        (destination / "jobs" / task / f"ep{entry['episode']:05d}" / "observations.h5").unlink()
    return destination


def retire_waiter(path, expected_status, successor):
    from cam.io import read, write

    state = read(path / "status.json")
    if state["status"] != expected_status:
        raise ValueError(f"Cannot retire an active or unexpected scheduler: {path}: {state['status']}")
    pid = read(path / "process.json")["pid"]
    proc = Path('/proc') / str(pid)
    if proc.exists():
        argv = (proc / 'cmdline').read_bytes().split(b'\0')
        if b'_worker' not in argv or Path(argv[argv.index(b'_worker') + 1].decode()).resolve() != path.resolve():
            raise ValueError("Scheduler process identity differs")
        os.kill(pid, signal.SIGSTOP)
        state = read(path / "status.json")
        if state["status"] != expected_status:
            os.kill(pid, signal.SIGCONT)
            raise ValueError("Scheduler advanced before handover")
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
    write(path / "status_before_progressive_handover.json", state)
    write(path / "status.json", {"status": "superseded", "successor": str(successor), "updated": time.time()})
    write(path / "successor.json", {"job": str(successor), "reason": "Start each validated task independently using local SSD assembly"})


def start(args):
    import cam
    from cam.benchmarks.base import source_hash
    from cam.benchmarks.registry import get_adapter, simulator_python
    from cam.harness.prepare import simulator_build_files
    from cam.io import read, write, sha256, tree_hash, copy_tree

    previous = args.previous.resolve()
    old_plan = read(previous / "plan.json")
    capture_job = Path(old_plan["producer"])
    capture_plan = read(capture_job / "plan.json")
    if read(capture_job / "status.json")["completed"] != 500:
        raise ValueError("All capture attempts must already be complete")
    experiment = read(previous / "experiment.json")
    item = experiment["benchmarks"]["rmbench"]
    config = read(item["config"])
    if (config["mode"], config["budget"], config["depth"]) != ("demos", 0, True):
        raise ValueError("Expected RGB-D demo-only development")
    adapter = get_adapter("rmbench", capture_plan["source"])
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    job = (ROOT / "outputs/rmbench_progressive_development" / tag).resolve()
    checkout = job / "inputs/checkout"
    checkout.mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, checkout / name)
    copy_tree(Path(cam.__file__).parent, checkout / "src/cam")
    source = checkout / "third_party/rmbench"
    copy_tree(adapter.source, source, adapter.pin_ignore)
    if source_hash(get_adapter("rmbench", source)) != source_hash(adapter):
        raise ValueError("Benchmark source changed while pinning")
    for recipe in simulator_build_files(adapter):
        target = checkout / "envs/rmbench" / recipe.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(recipe, target)
    pool = Path(capture_plan["pool"])
    manifest = read(pool / ("manifest.partial.json" if (pool / "manifest.partial.json").exists() else "manifest.json"))
    if manifest["benchmark_hash"] != source_hash(adapter):
        raise ValueError("Capture and benchmark source differ")
    write(job / "capture_manifest.json", manifest)
    write(job / "config.json", config)
    item["config"] = str(job / "config.json")
    item["datasets"] = [str(job / "datasets" / task) for task in item["tasks"]]
    write(job / "experiment.json", experiment)
    shutil.copy2(ROOT / "scripts/develop_batch.py", job / "boundary_checks.py")
    shutil.copy2(ROOT / "scripts/audit_run.py", job / "audit_run.py")
    shutil.copy2(Path(__file__), job / "driver.py")
    (job / "runner.py").write_text("from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).parent / 'inputs/checkout/src'))\nif __name__ == '__main__':\n from cam.cli import main\n raise SystemExit(main())\n")
    plan = {"tag": tag, "created": time.time(), "root": str(ROOT), "previous_queue": str(previous),
            "capture_job": str(capture_job), "capture_pool": str(pool), "benchmark_source": str(source),
            "benchmark_hash": source_hash(adapter), "harness_hash": tree_hash(checkout / "src/cam"),
            "tasks": item["tasks"], "workers": 8, "assembly_workers": 2, "python": sys.executable,
            "demo_counts": {t: item.get("demo_counts", {}).get(t, item["demo_count"]) for t in item["tasks"]},
            "environment": {"CAM_RMBENCH_PYTHON": str(simulator_python(adapter)),
                            "CAM_RMBENCH_ASSETS": str(next(iter(adapter.asset_links.values())).resolve())},
            "pinned_files": {name: sha256(job / name) for name in
                             ("config.json", "experiment.json", "capture_manifest.json", "driver.py", "boundary_checks.py", "audit_run.py", "runner.py")}}
    write(job / "plan.json", plan)
    tasks = {f"rmbench/{t}": {"benchmark": "rmbench", "task": t, "status": "waiting_data"} for t in plan["tasks"]}
    write(job / "status.json", {"status": "running", "workers": 8, "tasks": tasks, "updated": time.time()})
    (job / "README.md").write_text(
        f"# Progressive RMBench demo-only development: {tag}\n\n"
        "All ten tasks; 496 successful RGB-D demos (50 each except battery_try=49, blocks_ranking_try=49, observe_and_pickup=48). "
        "Claude Code / Opus 5.5, unchanged shared demo-only prompt, zero training rollouts, eight coding slots.\n\n"
        "Two local SSD assembly slots stage immutable successful capture files with SHA-256 checks, reuse the existing "
        "native-expert H5 assembly code, and run full file/schema/identity/test-disjointness validation. "
        "Five seeded random successful episodes per task receive detailed first/middle/last-frame RGB-D inspection, "
        "recorded in datasets/TASK/sample_inspection.json and .png. No simulator replay is repeated. "
        "Each task is prepared through cam prepare and starts learning as soon as its own dataset is ready; "
        "the eight active task slots include assembly/preparation. All-task completion still gates held-out evaluation.\n\n"
        f"Original captures at `{pool}` remain unchanged. The superseded NFS assembly is stopped, "
        "with its partial files and logs retained; completed local H5s are not copied to NFS automatically. "
        "Only redundant local staging H5s are removed after successful validation; original captures and receipts remain. "
        "The new local datasets under this job are self-contained, and prepared runs hard-link only their assigned H5. "
        "status.json, logs/, datasets/, and per-run READMEs retain counts, provenance and progress.\n"
    )
    if (previous / "batch.json").exists():
        raise ValueError("Previous queue already launched; preserve active learners instead")
    evaluation = read(previous / "evaluation_followup.json")
    retire_waiter(Path(evaluation["job"]), "waiting_for_development", job)
    retire_waiter(previous, "waiting_for_validated_pool", job)
    capture_state = read(capture_job / "status.json")
    if capture_state["status"] == "assembling" and capture_state["active"] == 0:
        retire_waiter(capture_job, "assembling", job)
    monitor = capture_job / "monitor"
    monitor_pid = read(monitor / "process.json")["pid"]
    monitor_proc = Path('/proc') / str(monitor_pid)
    if monitor_proc.exists():
        argv = (monitor_proc / 'cmdline').read_bytes().split(b'\0')
        if not any(token.endswith(b'/monitor_driver.py') for token in argv):
            raise ValueError("Capture monitor identity differs")
        if not any(Path(token.decode()).resolve() == capture_job for token in argv if token.startswith(b'/')):
            raise ValueError("Capture monitor belongs to another job")
        os.kill(monitor_pid, signal.SIGTERM)
    write(monitor / "status_before_local_handover.json", read(monitor / "status.json"))
    write(monitor / "status.json", {"status": "superseded", "successor": str(job), "completed": 500,
                                    "successes": 496, "updated": time.time()})
    with (job / "supervisor.log").open("ab") as log:
        proc = subprocess.Popen([sys.executable, str(job / "driver.py"), "_worker", str(job)],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=ROOT, env={**os.environ, **plan["environment"]})
    write(job / "process.json", {"pid": proc.pid, "started": time.time()})
    write(job / "batch.json", {"batch": str(job), "pid": proc.pid, "tasks": 10})
    return {"job": str(job), "pid": proc.pid}


def worker(job):
    from cam.io import read, write, sha256, tree_hash
    from cam.benchmarks.registry import get_adapter
    from cam.benchmarks.base import source_hash

    plan = read(job / "plan.json")
    if tree_hash(job / "inputs/checkout/src/cam") != plan["harness_hash"]:
        raise ValueError("Pinned harness changed")
    if source_hash(get_adapter("rmbench", plan["benchmark_source"])) != plan["benchmark_hash"]:
        raise ValueError("Pinned benchmark changed")
    for name, digest in plan["pinned_files"].items():
        if sha256(job / name) != digest:
            raise ValueError(f"Pinned input changed: {name}")
    spec = importlib.util.spec_from_file_location("boundary_checks", job / "boundary_checks.py")
    boundary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boundary)
    boundary.ROOT = Path(plan["root"])
    state = read(job / "status.json")
    guard, assembly = threading.Lock(), threading.BoundedSemaphore(plan["assembly_workers"])

    def update(task, **changes):
        with guard:
            state["tasks"][f"rmbench/{task}"].update(changes, updated=time.time())
            state["updated"] = time.time()
            write(job / "status.json", state)

    def command(label, argv):
        path = job / "logs" / f"{label}_{time.time_ns()}"
        write(path.with_suffix(".command.json"), {"argv": argv, "started": time.time()})
        with path.with_suffix(".log").open("xb") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, **plan["environment"], "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            write(path.with_suffix(".process.json"), {"pid": proc.pid, "started": time.time()})
            code = proc.wait()
        if code:
            raise RuntimeError(f"Command exited {code}: {path.with_suffix('.log')}")
        return path.with_suffix(".log")

    def task_worker(task):
        try:
            with assembly:
                update(task, status="staging_captures")
                dataset = materialize_task(job, plan, task, lambda **changes: update(task, **changes))
            update(task, status="preparing", dataset=str(dataset))
            log = command(f"prepare_{task}", [plan["python"], str(job / "runner.py"), "prepare", str(ROOT / "runs"),
                "--dataset", str(dataset), "--config", str(job / "config.json"), "--benchmark-source", plan["benchmark_source"],
                "--timetag", plan["tag"], "--tasks", task])
            root = Path(read(log)["runs"][0]["root"])
            boundary.boundary_check(root, task, plan["demo_counts"][task])
            update(task, status="learning", run=str(root))
            command(f"learn_{task}", [plan["python"], str(root / "runner.py"), "learn", str(root),
                                      "--gpus", str(plan["tasks"].index(task) % 2), "--workers-per-gpu", "1"])
            completion = read(root / "tasks" / task / "logs/completion.json")
            command(f"audit_{task}", [plan["python"], str(job / "audit_run.py"), str(root), "--task", task])
            update(task, status=completion["status"], completion=completion)
        except Exception:
            update(task, status="failed", error=traceback.format_exc())

    with ThreadPoolExecutor(max_workers=plan["workers"]) as executor:
        list(executor.map(task_worker, plan["tasks"]))
    state["status"] = "finished" if all(r["status"] == "finished" for r in state["tasks"].values()) else "needs_attention"
    state["updated"] = time.time()
    write(job / "status.json", state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('start').add_argument('previous', type=Path)
    sub.add_parser('_worker').add_argument('job', type=Path)
    args = parser.parse_args()
    if args.command == '_worker':
        ROOT = Path(json.loads((args.job / 'plan.json').read_text())['root'])
        sys.path.insert(0, str(args.job / 'inputs/checkout/src'))
    else:
        sys.path.insert(0, str(ROOT / 'src'))
    sys.dont_write_bytecode = True
    if args.command == 'start':
        print(json.dumps(start(args), indent=2))
    else:
        try:
            worker(args.job)
        except Exception:
            from cam.io import write
            state = json.loads((args.job / 'status.json').read_text())
            state.update(status='needs_attention', error=traceback.format_exc(), updated=time.time())
            write(args.job / 'status.json', state)
            raise
