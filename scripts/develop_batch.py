"""Start a pinned demo-only or demo-plus-interaction development pool.

Usage: .venv/bin/python scripts/develop_batch.py start PLAN --workers 8 --gpus 0 1
       .venv/bin/python scripts/develop_batch.py status BATCH
PLAN declares tasks, demo counts, datasets and raw-data roots. Existing plans
default to 25 demos. MIKASA v2 requires completed, separately rendered data.
Missing demos for other benchmarks may be extracted. No held-out tests.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def data_assignment(name, item, config):
    """Check budgets and complete manifests before creating or launching a batch."""
    from cam.config import RunConfig
    from cam.io import read

    cfg = RunConfig.from_dict(config)
    count = item.get("demo_count", 25)
    if type(count) is not int or count < 1:
        raise ValueError("demo_count must be a positive integer")
    if cfg.benchmark != name:
        raise ValueError("Batch benchmark differs from configuration")
    if cfg.mode == "demos":
        if cfg.budget != 0:
            raise ValueError("Demo-only batches require zero simulator attempts")
    elif (cfg.mode, cfg.both_protocol, cfg.training_seed_policy) != (
        "both", "v2", "disjoint_once"
    ) or cfg.budget < 1 or cfg.budget != cfg.training_seed_count:
        raise ValueError("Batch requires demo-only or v2 with one distinct training seed per allowed attempt")
    datasets = {}
    for path in item["datasets"]:
        root = (ROOT / path).resolve()
        if not (root / "manifest.json").is_file():
            raise ValueError(f"Dataset is not finalized: {root}/manifest.json")
        manifest = read(root / "manifest.json")
        if manifest["benchmark"] != name or bool(manifest["depth"]) != cfg.depth:
            raise ValueError(f"Dataset benchmark or depth differs from configuration: {root}")
        datasets[str(root)] = manifest
    existing, missing = {}, []
    for task in item["tasks"]:
        dataset = next((d for d, m in datasets.items() if task in m["tasks"]), None)
        if dataset is None:
            missing.append(task)
            continue
        records = datasets[dataset]["tasks"][task]["episodes"]
        if len(records) != count:
            raise ValueError(f"Expected exactly {count} demos: {name}/{task}, found {len(records)}")
        existing.setdefault(dataset, []).append(task)
    if missing and (name == "mikasa_robo_v2" or item.get("require_prepared_data", False)):
        raise ValueError(f"Complete the prepared demonstrations first: {name}: {', '.join(missing)}")
    return {"demo_count": count, "existing": existing, "missing": missing}


def start(args):
    import cam
    from cam.benchmarks.base import source_hash
    from cam.benchmarks.registry import get_adapter, simulator_python
    from cam.harness.prepare import simulator_build_files
    from cam.io import copy_tree, read, tree_hash, write

    spec = read(args.plan)
    configs = {name: read(ROOT / item["config"]) for name, item in spec["benchmarks"].items()}
    assignments = {name: data_assignment(name, item, configs[name]) for name, item in spec["benchmarks"].items()}
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    batch = (ROOT / "outputs/development_batches" / tag).resolve()
    batch.mkdir(parents=True)
    data = (ROOT / "data/development_batches" / tag).resolve()
    data.mkdir(parents=True)
    checkout = batch / "inputs/checkout"
    checkout.mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, checkout / name)
    copy_tree(Path(cam.__file__).parent, checkout / "src/cam")
    shutil.copy2(ROOT / "scripts/audit_run.py", batch / "audit_run.py")
    env, benches = {}, {}
    for name, item in spec["benchmarks"].items():
        adapter = get_adapter(name)
        tasks = item["tasks"]
        if len(set(tasks)) != len(tasks) or set(tasks) - set(adapter.tasks()):
            raise ValueError(f"Invalid task selection: {name}")
        target = checkout / "third_party" / adapter.package
        if name == "robomemarena":
            target /= "evaluation_benchmark"
        if not target.exists():
            copy_tree(adapter.source, target, getattr(adapter, "pin_ignore", ()))
        assets = {}
        if name == "robomemarena":
            for link, original in adapter.asset_links.items():
                path = target / link
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(Path(original).resolve(), target_is_directory=True)
                assets[link] = tree_hash(original)
        else:
            asset = next(iter(adapter.asset_links.values()))
            env[f"CAM_{adapter.package.upper()}_ASSETS"] = str(Path(asset).resolve())
        assert source_hash(get_adapter(name, target)) == source_hash(adapter)
        env[f"CAM_{name.upper()}_PYTHON"] = str(simulator_python(adapter))
        for recipe in simulator_build_files(adapter):
            dest = checkout / "envs" / name / recipe.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(recipe, dest)
        config = configs[name]
        write(batch / "inputs" / f"{name}.json", config)
        benches[name] = {"tasks": tasks, **assignments[name], "interaction_count": config["budget"],
                         "raw": str((ROOT / item["raw"]).resolve()), "source": str(target),
                         "source_hash": source_hash(adapter), "asset_hashes": assets,
                         "config": str(batch / "inputs" / f"{name}.json"),
                         "new_data": str(data / name), "depth": config["depth"]}
    plan = {"created": time.time(), "tag": tag, "workers": args.workers, "gpus": args.gpus,
            "selection_reason": spec["selection_reason"], "benchmarks": benches,
            "environment": env, "python": sys.executable, "runs": str((ROOT / "runs").resolve()),
            "harness_hash": tree_hash(checkout / "src/cam"), "held_out_evaluation": False}
    write(batch / "plan.json", plan)
    shutil.copy2(Path(__file__), batch / "driver.py")
    (batch / "runner.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'inputs/checkout/src'))\n"
        "if __name__ == '__main__':\n    from cam.cli import main\n    raise SystemExit(main())\n"
    )
    rows = [f"| {name} | {cfg['demo_count']} | {cfg['interaction_count']} | {', '.join(cfg['tasks'])} |" for name, cfg in benches.items()]
    (batch / "README.md").write_text(
        f"# Coding development batch: {tag}\n\n"
        f"{sum(len(b['tasks']) for b in benches.values())} tasks, {args.workers} concurrent coding sessions; "
        "Exact clients/models are pinned in the per-benchmark configurations. Each task gets the "
        "demo and interaction counts below. Demo-only tasks allow no simulator attempts. "
        "For v2 tasks, interaction seeds are distinct and disjoint from demos and tests; "
        "use the remaining attempts for verification even when the solution seems ready.\n\n"
        + spec["selection_reason"] + "\n\n| Benchmark | Demos/task | Max interactions/task | Selected tasks |\n|---|---|---|---|\n" + "\n".join(rows)
        + "\n\n`plan.json` pins selection, configs, sources and data locations. "
        "`status.json` lists every task and run path; `logs/` retains exact commands and output. "
        "Each run has its own README, transcript, command audit, source history and checkpoints. "
        "The learner sees only its assigned demo H5, public API, own code and counted feedback. "
        "Source, private metadata, host files, networking, other tasks and test feedback are unavailable.\n\n"
        "Coding sessions run concurrently. A shared host lease permits one learning simulator per GPU. "
        "RoboMemArena demo preparation reserves one whole GPU while its replay workers run. "
        "Waiting for a GPU does not consume attempts. Missing demos are prepared before launching "
        "that task; data failures are reported, never silently replaced with another task. "
        "No held-out evaluation is started.\n\n"
        "Inspect: `.venv/bin/python scripts/develop_batch.py status BATCH`. "
        "Completed sessions are frozen automatically; failures retain evidence for explicit recovery.\n"
    )
    with (batch / "supervisor.log").open("ab") as log:
        proc = subprocess.Popen([sys.executable, str(batch / "driver.py"), "_worker", str(batch)],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=batch)
    write(batch / "process.json", {"pid": proc.pid, "started": time.time()})
    return {"batch": str(batch), "pid": proc.pid, "tasks": sum(len(b["tasks"]) for b in benches.values())}


def boundary_check(root, task, demo_count=25):
    from cam.harness.isolation import run
    from cam.io import read, write

    plan = read(root / "run.json")
    folder = root / "tasks" / task
    state = read(folder / "state/config.json")
    demos = read(root / "inputs/dataset.json")["tasks"][task]["episodes"]
    train = state["train_records"]
    tests = read(root / "inputs/metadata/test" / f"record_dataset_{task}_metadata.json")["records"]
    assert len(demos) == demo_count
    assert {r["seed"] for r in demos}.isdisjoint(r["seed"] for r in tests)
    prompt = (folder / "prompt.md").read_text()
    if plan["config"]["mode"] == "demos":
        interaction_count = 0
        assert plan["config"]["budget"] == 0
        assert "Zero simulator attempts are allowed" in prompt
    else:
        interaction_count = plan["config"]["training_seed_count"]
        assert len(train) == plan["config"]["budget"] == interaction_count
        assert len({r["seed"] for r in train}) == interaction_count
        assert {r["seed"] for r in train}.isdisjoint(r["seed"] for r in demos + tests)
        assert {r["episode"] for r in train}.isdisjoint(r["episode"] for r in demos)
        assert "disjoint from the assigned demonstrations; no repeats" in prompt
    assert "Task-specific contract" not in prompt and "Success criterion" not in prompt
    private = [str(root), str(root / "inputs"), str(ROOT), plan["raw_data_reference"],
               "/runtime/cam/benchmarks", "/runtime/cam/harness", "/feedback/../inputs"]
    script = """import json, os, socket, pathlib, h5py
private = PRIVATE
assert not any(pathlib.Path(p).exists() for p in private), private
assert not any(k.startswith(('AWS_', 'ANTHROPIC_', 'CAM_SIMULATOR_')) for k in os.environ)
files = list(pathlib.Path('/demonstrations').glob('*.h5'))
assert len(files) == 1
with h5py.File(files[0], 'r') as h: assert len(h) == DEMO_COUNT
s = socket.socket(); s.settimeout(1)
try: result = s.connect_ex(('192.0.2.1', 443))
except OSError: result = -1
assert result != 0
s.close()
print(json.dumps({'passed': True, 'public_demo_episodes': DEMO_COUNT, 'private_paths_hidden': True, 'network_unreachable': True}))
""".replace("PRIVATE", repr(private)).replace("DEMO_COUNT", str(demo_count))
    code, out, err = run(folder / "workspace", root / "runtime", ["python", "-c", script],
                         feedback=folder / "feedback", demonstrations=folder / "demonstrations", readonly=True)
    if code:
        raise RuntimeError("Boundary preflight failed: " + (out + err).decode(errors="replace"))
    result = json.loads(out)
    result.update(disjoint_interaction_identities=interaction_count, zero_test_overlap=True, simulator_attempts=0)
    write(root / "orchestration/boundary_preflight.json", result)


def worker(batch):
    from cam.benchmarks.base import source_hash
    from cam.benchmarks.registry import get_adapter
    from cam.io import read, tree_hash, write

    with (batch / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(batch / "plan.json")
        assert tree_hash(batch / "inputs/checkout/src/cam") == plan["harness_hash"]
        for name, cfg in plan["benchmarks"].items():
            assert source_hash(get_adapter(name, cfg["source"])) == cfg["source_hash"]
            for link, digest in cfg["asset_hashes"].items():
                assert tree_hash(Path(cfg["source"]) / link) == digest
        env = {**os.environ, **plan["environment"], "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "CAM_SIMULATOR_LOCK_DIR": str(batch / "gpu_locks"),
               "CAM_SIMULATOR_GPUS": ",".join(map(str, plan["gpus"]))}
        (batch / "gpu_locks").mkdir(exist_ok=True)
        (batch / "logs").mkdir(exist_ok=True)
        tasks = {f"{name}/{task}": {"benchmark": name, "task": task, "status": "waiting_data"}
                 for name, cfg in plan["benchmarks"].items() for task in cfg["tasks"]}
        guard, ready, preparing = threading.Lock(), queue.Queue(), threading.Event()

        def save(key=None, **changes):
            with guard:
                if key:
                    tasks[key].update(changes, updated=time.time())
                write(batch / "status.json", {"status": "running", "pid": os.getpid(),
                      "updated": time.time(), "workers": plan["workers"], "tasks": tasks})

        def command(label, argv, extra=None):
            path = batch / "logs" / f"{label}_{time.time_ns()}"
            write(path.with_suffix(".command.json"), {"argv": argv, "started": time.time()})
            with path.with_suffix(".log").open("xb") as log:
                proc = subprocess.Popen(argv, env={**env, **(extra or {})}, stdout=log, stderr=subprocess.STDOUT)
                write(path.with_suffix(".process.json"), {"pid": proc.pid, "started": time.time()})
                code = proc.wait()
            if code:
                raise RuntimeError(f"Command exited {code}: {path.with_suffix('.log')}")
            return path.with_suffix(".log")

        def prepare_group(name, cfg, dataset, group):
            for task in group:
                save(f"{name}/{task}", status="preparing", dataset=str(dataset))
            log = command(f"prepare_{name}", [plan["python"], str(batch / "runner.py"), "prepare", plan["runs"],
                          "--dataset", str(dataset), "--config", cfg["config"], "--benchmark-source", cfg["source"],
                          "--timetag", plan["tag"], "--tasks", *group])
            for row in json.loads(log.read_text())["runs"]:
                root, task = Path(row["root"]), row["task"]
                boundary_check(root, task, cfg.get("demo_count", 25))
                save(f"{name}/{task}", status="queued", run=str(root))
                ready.put((name, task, root))

        def prepare_benchmark(name, cfg):
            for dataset, group in cfg["existing"].items():
                try:
                    prepare_group(name, cfg, dataset, group)
                except Exception as exc:
                    for task in group:
                        if tasks[f"{name}/{task}"]["status"] == "preparing":
                            save(f"{name}/{task}", status="preparation_failed", error=str(exc))
            group = cfg["missing"]
            if not group:
                return
            for task in group:
                save(f"{name}/{task}", status="extracting")
            args = [plan["python"], str(batch / "runner.py"), "extract", "--benchmark", name,
                    "--benchmark-source", cfg["source"], "--raw-data", cfg["raw"], "--output", cfg["new_data"],
                    "--count", str(cfg.get("demo_count", 25)), "--workers", "4" if name == "robomemarena" else "2", "--no-videos", "--tasks", *group]
            if not cfg["depth"]:
                args.append("--rgb-only")
            try:
                if name == "robomemarena":
                    gpu = plan["gpus"][-1]
                    with (batch / "gpu_locks" / f"gpu_{gpu}.lock").open("a") as lease:
                        fcntl.flock(lease, fcntl.LOCK_EX)
                        command(f"extract_{name}", args, {"CUDA_VISIBLE_DEVICES": str(gpu)})
                else:
                    command(f"extract_{name}", args)
                prepare_group(name, cfg, cfg["new_data"], group)
            except Exception as exc:
                for task in group:
                    if tasks[f"{name}/{task}"]["status"] in ("extracting", "preparing"):
                        save(f"{name}/{task}", status="preparation_failed", error=str(exc))

        def develop(index):
            while not preparing.is_set() or not ready.empty():
                try:
                    name, task, root = ready.get(timeout=1)
                except queue.Empty:
                    continue
                key = f"{name}/{task}"
                try:
                    save(key, status="learning", worker=index)
                    gpu = plan["gpus"][index % len(plan["gpus"])]
                    command(f"learn_{name}_{task}", [plan["python"], str(root / "runner.py"), "learn", str(root),
                            "--gpus", str(gpu), "--workers-per-gpu", "1"])
                    completion = read(root / "tasks" / task / "logs/completion.json")
                    save(key, status=completion["status"], completion=completion)
                    command(f"audit_{name}_{task}", [plan["python"], str(batch / "audit_run.py"), str(root), "--task", task])
                except Exception as exc:
                    save(key, status="failed", error=str(exc))
                finally:
                    ready.task_done()

        save()
        with ThreadPoolExecutor(max_workers=plan["workers"]) as learners:
            learning = [learners.submit(develop, index) for index in range(plan["workers"])]
            try:
                with ThreadPoolExecutor(max_workers=len(plan["benchmarks"])) as preparers:
                    futures = [preparers.submit(prepare_benchmark, name, cfg) for name, cfg in plan["benchmarks"].items()]
                    for future in as_completed(futures):
                        future.result()
            finally:
                preparing.set()
            for future in learning:
                future.result()
        write(batch / "status.json", {"status": "finished" if all(t["status"] == "finished" for t in tasks.values()) else "needs_attention",
              "finished": time.time(), "workers": plan["workers"], "tasks": tasks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("start")
    p.add_argument("plan", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    for name in ("_worker", "status"):
        commands.add_parser(name).add_argument("batch", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    source = ROOT / "src" if args.command == "start" else args.batch.resolve() / "inputs/checkout/src"
    sys.path.insert(0, str(source))
    if args.command == "start":
        if args.workers < 1 or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
            parser.error("Use positive workers and distinct nonnegative GPUs")
        print(json.dumps(start(args), indent=2))
    elif args.command == "_worker":
        worker(args.batch.resolve())
    else:
        from cam.io import read
        result = read(args.batch / "status.json")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
