"""Prepare all new-benchmark demo pools and separate initial subsets in the background.

Run with CaM's Python: scripts/prepare_demo_pools.py start --gpu 1
Inspect or resume with: scripts/prepare_demo_pools.py {status,resume} JOB
No coding client or held-out evaluation is launched. Original datasets stay intact.
MIKASA and RMBench use their published RGB-D pools directly; see docs/benchmarks/rmbench/README.md.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("robomemarena",)


def alive(pid):
    try:
        return (Path("/proc") / str(int(pid)) / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def launch(job):
    from cam.io import write

    with (job / "supervisor.log").open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, str(job / "driver.py"), "_worker", str(job)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=job,
        )
    write(job / "process.json", {"pid": process.pid, "started": time.time()})
    return {"job": str(job), "pid": process.pid, "status": "started"}


def start(args):
    import cam
    from cam.interface.base import source_hash
    from cam.interface.registry import get_adapter, simulator_python
    from cam.harness.prepare import environment_snapshot, simulator_snapshot
    from cam.io import copy_tree, tree_hash, write

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    job = (ROOT / "outputs/demo_preparation" / tag).resolve()
    data = (ROOT / "data/demo_pools" / tag).resolve()
    job.mkdir(parents=True)
    data.mkdir(parents=True)
    checkout = job / "inputs/checkout"
    checkout.mkdir(parents=True)
    shutil.copy2(ROOT / "pyproject.toml", checkout / "pyproject.toml")
    shutil.copy2(ROOT / "uv.lock", checkout / "uv.lock")
    copy_tree(Path(cam.__file__).parent, checkout / "src/cam")
    settings = {}
    for name in BENCHMARKS:
        adapter = get_adapter(name)
        suffix = "evaluation_benchmark" if name == "robomemarena" else ""
        target = checkout / "third_party" / name / suffix
        copy_tree(adapter.source, target, getattr(adapter, "pin_ignore", ()))
        # Replay needs assets; CPU extraction does not. Assets remain private and
        # are excluded from the official source fingerprint by this adapter.
        assets = {}
        if name == "robomemarena":
            for link, original in adapter.asset_links.items():
                path = target / link
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(Path(original).resolve(), target_is_directory=True)
                assets[link] = tree_hash(original)
        assert source_hash(get_adapter(name, target)) == source_hash(adapter)
        tasks = adapter.tasks()
        native_counts = {task: len(adapter.records(task, "train")) for task in tasks}
        # RoboMemArena has 100 candidate replays.
        if name == "robomemarena" and max(native_counts.values()) > args.target:
            raise ValueError("RoboMemArena count=all would exceed the requested pool target")
        count = args.target if name == "mikasa_robo" else "all"
        python = simulator_python(adapter)
        settings[name] = {
            "source": str(target), "source_hash": source_hash(adapter),
            "asset_hashes": assets, "tasks": tasks, "released_training_counts": native_counts,
            "test_episode_counts": {task: len(adapter.records(task, "test")) for task in tasks},
            "count": count, "depth": name == "robomemarena",
            "workers": args.replay_workers if name == "robomemarena" else (args.cpu_workers if name == "mikasa_robo" else 2),
            "raw": str((ROOT / "data/raw" / name / ("rlds" if name == "mikasa_robo" else "")).resolve()),
            "pool": str(data / name / "pool"), "initial": str(data / name / f"initial_{args.initial}"),
            "simulator_python": str(python) if python else None,
        }
        if name == "robomemarena":
            write(job / "inputs/simulator_environment.json", simulator_snapshot(python))
    write(job / "inputs/environment.json", environment_snapshot())
    plan = {"created": time.time(), "target": args.target, "initial": args.initial,
            "gpu": args.gpu, "python": sys.executable, "data": str(data),
            "harness_hash": tree_hash(checkout / "src/cam"), "benchmarks": settings}
    write(job / "plan.json", plan)
    shutil.copy2(Path(__file__), job / "driver.py")
    (job / "runner.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'inputs/checkout/src'))\n"
        "if __name__ == '__main__':\n"
        "    from cam.cli import main\n    raise SystemExit(main())\n"
    )
    (job / "README.md").write_text(
        f"# All-task demonstration preparation\n\nTarget: up to {args.target} distinct demos per task; "
        f"separate {args.initial}-demo initial subsets.\n\n"
        "RoboMemArena replays all 100 training identities and retains successful replays. "
        "Actual counts and shortages are recorded, never padded. All raw data is preserved.\n\n"
        "The job snapshots its extraction code and benchmark source. It saves logs and "
        "resumes completed task artifacts with integrity checks. Pool H5s are private "
        "preparation data: prepare experiments from the initial subset path, not the pool, "
        "to expose only the assigned demos. Extra preview videos are omitted.\n\n"
        f"Data root: `{data}`. Progress: `status.json` and `<benchmark>.status.json`. "
        "Inspect with `scripts/prepare_demo_pools.py status JOB`; resume with `resume JOB`. "
        "No learner or held-out test is run.\n"
    )
    return launch(job)


def worker(job):
    from cam.interface.base import source_hash
    from cam.interface.registry import get_adapter
    from cam.io import read, tree_hash, write

    lock = (job / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(job / "plan.json")
    assert tree_hash(job / "inputs/checkout/src/cam") == plan["harness_hash"]
    for name, cfg in plan["benchmarks"].items():
        assert source_hash(get_adapter(name, cfg["source"])) == cfg["source_hash"]
        for link, digest in cfg["asset_hashes"].items():
            assert tree_hash(Path(cfg["source"]) / link) == digest
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": str(plan["gpu"])}
    for name, cfg in plan["benchmarks"].items():
        if cfg["simulator_python"]:
            env[f"CAM_{name.upper()}_PYTHON"] = cfg["simulator_python"]

    def one(name, cfg):
        path = job / f"{name}.status.json"
        state = {"benchmark": name, "status": "extracting", "updated": time.time(), "pool": cfg["pool"], "initial": cfg["initial"]}
        write(path, state)

        def run(stage, arguments):
            command = [plan["python"], "-B", str(job / "runner.py"), *arguments]
            write(job / f"{name}.{stage}.command.json", {"argv": command, "gpu": plan["gpu"]})
            with (job / f"{name}.{stage}.log").open("ab") as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                state.update(stage=stage, pid=proc.pid, updated=time.time())
                write(path, state)
                code = proc.wait()
            if code:
                raise RuntimeError(f"{stage} exited {code}; see {name}.{stage}.log")

        try:
            pool = Path(cfg["pool"])
            args = ["extract", "--benchmark", name, "--benchmark-source", cfg["source"], "--raw-data", cfg["raw"],
                    "--output", str(pool), "--count", str(cfg["count"]), "--workers", str(cfg["workers"]), "--no-videos"]
            if not cfg["depth"]:
                args.append("--rgb-only")
            if pool.exists():
                args.append("--resume")
            run("extract", args)
            manifest = read(pool / "manifest.json")
            counts = {task: len(value["episodes"]) for task, value in manifest["tasks"].items()}
            state.update(status="selecting_initial", pool_counts=counts,
                         target_shortfalls={t: n for t, n in counts.items() if n < plan["target"]})
            write(path, state)
            eligible = [task for task, n in counts.items() if n]
            if not eligible:
                raise RuntimeError("No successful demonstrations available; preserved raw data and replay diagnostics")
            initial = Path(cfg["initial"])
            args = ["subset", "--dataset", str(pool), "--output", str(initial), "--count", str(plan["initial"]),
                    "--allow-fewer", "--tasks", *eligible]
            if initial.exists():
                args.append("--resume")
            run("subset", args)
            state.update(status="complete" if len(eligible) == len(cfg["tasks"]) else "needs_attention",
                         initial_counts={task: len(value["episodes"]) for task, value in read(initial / "manifest.json")["tasks"].items()},
                         unavailable_tasks=sorted(set(cfg["tasks"]) - set(eligible)), finished=time.time())
        except Exception as exc:
            state.update(status="failed", error=str(exc), finished=time.time())
        state["updated"] = time.time()
        state.pop("pid", None)
        write(path, state)
        return name, state

    write(job / "status.json", {"status": "running", "pid": os.getpid(), "updated": time.time()})
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(one, name, cfg) for name, cfg in plan["benchmarks"].items()]
        for future in as_completed(futures):
            name, state = future.result()
            results[name] = state
            write(job / "status.json", {"status": "running", "updated": time.time(), "benchmarks": results})
    write(job / "status.json", {"status": "complete" if all(s["status"] == "complete" for s in results.values()) else "needs_attention",
                               "finished": time.time(), "benchmarks": results})


def status(job):
    from cam.io import read

    result = read(job / "status.json") if (job / "status.json").exists() else {"status": "starting"}
    plan = read(job / "plan.json")
    result["job"] = str(job)
    process = read(job / "process.json")
    result["worker_alive"] = alive(process["pid"])
    if not result["worker_alive"] and result["status"] in ("running", "starting"):
        result["status"] = "interrupted"
    result["benchmarks"] = {}
    for name, cfg in plan["benchmarks"].items():
        path = job / f"{name}.status.json"
        state = read(path) if path.exists() else {"status": "pending"}
        folder = Path(cfg["pool"])
        completed = list((folder / "audit").glob("*.json"))
        state["pool_tasks_written"] = sum(not p.name.endswith(".replay.json") for p in completed)
        state["replay_tasks_audited"] = sum(p.name.endswith(".replay.json") for p in completed)
        state["tasks_expected"] = len(cfg["tasks"])
        result["benchmarks"][name] = state
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("start")
    command.add_argument("--target", type=int, default=100)
    command.add_argument("--initial", type=int, default=25)
    command.add_argument("--gpu", type=int, default=1)
    command.add_argument("--cpu-workers", type=int, default=4)
    command.add_argument("--replay-workers", type=int, default=4)
    for name in ("status", "resume", "_worker"):
        commands.add_parser(name).add_argument("job", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((args.job.resolve() / "inputs/checkout/src") if args.command != "start" else ROOT / "src"))
    if args.command == "start":
        if min(args.target, args.initial, args.cpu_workers, args.replay_workers) < 1 or args.initial > args.target:
            parser.error("Use positive limits and initial <= target")
        result = start(args)
    elif args.command == "_worker":
        worker(args.job.resolve())
        return
    elif args.command == "resume":
        # Do not launch a second supervisor over running extraction children.
        from cam.io import read

        job = args.job.resolve()
        paths = [job / "process.json", *job.glob("*.status.json")]
        for path in paths:
            info = read(path)
            if info.get("pid") and alive(info["pid"]):
                raise RuntimeError(f"Preparation process {info['pid']} is still alive")
        result = launch(job)
    else:
        result = status(args.job.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
