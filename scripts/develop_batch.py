"""Start a pinned demo-only or demo-plus-interaction development pool.

Usage: .venv/bin/python scripts/develop_batch.py start PLAN --workers 8 --gpus 0 1
       .venv/bin/python scripts/develop_batch.py status BATCH
PLAN declares tasks, demo counts, datasets and raw-data roots. Existing plans
default to 25 demos. MIKASA requires its released RGB-D demonstration package.
Missing demos for other benchmarks may be extracted. --evaluate-on-freeze opts
into a separate host evaluation queue after each task finishes and freezes.
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


def frozen_evaluation(root, task):
    """Require completed learning and an intact final checkpoint before testing."""
    from cam.core.budget import program_hash
    from cam.core.history import select_checkpoint
    from cam.io import read

    root = Path(root)
    plan = read(root / "run.json")
    folder = root / "tasks" / task
    if read(folder / "logs/completion.json")["status"] != "finished":
        raise ValueError("Evaluation requires successful learning completion")
    final = read(folder / "policy/freeze.json")
    budget = plan["config"]["budget"]
    if final["checkpoint"] != budget or final["training"]["attempts"] != budget:
        raise ValueError("Automatic evaluation requires the full assigned training budget")
    _, checkpoint = select_checkpoint(folder, budget)
    if (program_hash(folder / "policy") != final["program_hash"]
            or checkpoint["program_hash"] != final["program_hash"]):
        raise ValueError("Frozen policy and final checkpoint integrity mismatch")
    episodes = plan["test_episodes"][task]
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Evaluation requires distinct assigned held-out episodes")
    return {"budget": budget, "expected": len(episodes), "program_hash": final["program_hash"]}


def data_assignment(name, item, config):
    """Check budgets and complete manifests before creating or launching a batch."""
    from cam.config import RunConfig
    from cam.io import read

    cfg = RunConfig.from_dict(config)
    count = item.get("demo_count", 25)
    if type(count) is not int or count < 1:
        raise ValueError("demo_count must be a positive integer")
    counts = item.get("demo_counts", {})
    if not isinstance(counts, dict) or set(counts) - set(item["tasks"]):
        raise ValueError("demo_counts must map selected tasks to explicit counts")
    if any(type(value) is not int or value < 1 or value > count for value in counts.values()):
        raise ValueError("Per-task demo counts must be positive and no larger than demo_count")
    interactions = item.get("interaction_counts", {})
    if not isinstance(interactions, dict) or set(interactions) - set(item["tasks"]):
        raise ValueError("interaction_counts must map selected tasks to explicit counts")
    if interactions and (cfg.mode != "both" or cfg.training_seed_policy != "demo_cycle"):
        raise ValueError("Per-task interaction counts require the demo-seed protocol")
    if any(type(value) is not int or value < 1 or value > cfg.budget for value in interactions.values()):
        raise ValueError("Per-task interaction counts must be positive and no larger than the budget")
    if cfg.benchmark != name:
        raise ValueError("Batch benchmark differs from configuration")
    if cfg.mode == "demos":
        if cfg.budget != 0:
            raise ValueError("Demo-only batches require zero simulator attempts")
    elif (cfg.mode != "both" or cfg.both_protocol not in ("v2", "demo_first")
          or cfg.training_seed_policy not in ("disjoint_once", "demo_cycle")
          or cfg.budget < 1 or cfg.budget != cfg.training_seed_count):
        raise ValueError("Batch requires demo-only or v2 with one distinct training seed per allowed attempt")
    datasets = {}
    for path in item["datasets"]:
        root = (ROOT / path).resolve()
        if not (root / "manifest.json").is_file():
            raise ValueError(f"Dataset is not finalized: {root}/manifest.json")
        manifest = read(root / "manifest.json")
        from cam.interface.registry import benchmark_matches
        if not benchmark_matches(name, manifest["benchmark"]) or bool(manifest["depth"]) != cfg.depth:
            raise ValueError(f"Dataset benchmark or depth differs from configuration: {root}")
        datasets[str(root)] = manifest
    existing, missing = {}, []
    for task in item["tasks"]:
        dataset = next((d for d, m in datasets.items() if task in m["tasks"]), None)
        if dataset is None:
            missing.append(task)
            continue
        records = datasets[dataset]["tasks"][task]["episodes"]
        expected = counts.get(task, count)
        if len(records) != expected:
            raise ValueError(f"Expected exactly {expected} demos: {name}/{task}, found {len(records)}")
        if (cfg.mode == "both" and cfg.training_seed_policy == "demo_cycle"
                and expected != interactions.get(task, cfg.budget)):
            raise ValueError("Demo-seed batches require one attempt for every assigned demonstration")
        existing.setdefault(dataset, []).append(task)
    if missing and (name in ("mikasa_robo", "mikasa_robo_v2") or item.get("require_prepared_data", False)):
        raise ValueError(f"Complete the prepared demonstrations first: {name}: {', '.join(missing)}")
    if any(task in counts for task in missing):
        raise ValueError("Explicit per-task counts require already prepared demonstrations")
    if any(task in interactions for task in missing):
        raise ValueError("Explicit interaction counts require already prepared demonstrations")
    assignment = {"demo_count": count, "existing": existing, "missing": missing}
    if counts:
        assignment["demo_counts"] = {task: counts.get(task, count) for task in item["tasks"]}
    if interactions:
        assignment["interaction_counts"] = {task: interactions.get(task, cfg.budget) for task in item["tasks"]}
    return assignment


def task_configuration(config, assignment, task):
    """Pin a demo shortfall's attempt count without changing its learning protocol."""
    budget = assignment.get("interaction_counts", {}).get(task, config["budget"])
    if budget == config["budget"]:
        return dict(config)
    return {**config, "budget": budget, "training_seed_count": budget}


def preparation_groups(cfg):
    """Prepare tasks sharing an assigned dataset and exact pinned configuration."""
    for dataset, tasks in cfg["existing"].items():
        groups = {}
        for task in tasks:
            config = cfg.get("task_configs", {}).get(task, cfg["config"])
            groups.setdefault(config, []).append(task)
        for config, group in groups.items():
            yield dataset, group, {**cfg, "config": config}


def assign_downloaded_pool(name, item, output):
    """Materialize a selected-25 H5 during preparation, before any learner starts."""
    from cam.data.subset import selected_rows, subset
    from cam.io import read

    if item.get("demo_count", 25) != 25:
        return
    datasets = []
    for index, raw in enumerate(item["datasets"]):
        source = (ROOT / raw).resolve()
        if not (source / "manifest.json").is_file():
            datasets.append(raw)
            continue
        manifest = read(source / "manifest.json")
        tasks = [task for task in item["tasks"] if task in manifest["tasks"]]
        selection = source / "selected_25.json"
        if not selection.exists():
            selection = ROOT / "configs" / name / "datasets/demo_only_25_seeds.json"
        if not tasks or not selection.exists():
            datasets.append(raw)
            continue
        assignment = read(selection)
        for task in tasks:
            selected_rows(manifest, assignment, task, 25)
        if any(len(manifest["tasks"][task]["episodes"]) > 25 for task in tasks):
            destination = Path(output) / name / f"selected_25_{index}"
            subset(source, destination, 25, tasks, selection=selection, validate_source=False)
            datasets.append(str(destination))
        else:
            datasets.append(raw)
    item["datasets"] = datasets


def start(args):
    import cam
    from cam.interface.base import source_hash
    from cam.interface.registry import get_adapter, simulator_python
    from cam.harness.prepare import simulator_build_files
    from cam.io import copy_tree, read, tree_hash, write

    spec = read(args.plan)
    if getattr(args, "dataset", None):
        if len(spec["benchmarks"]) != 1:
            raise ValueError("--dataset requires a single-benchmark plan")
        item = next(iter(spec["benchmarks"].values()))
        item["datasets"] = [str(args.dataset.resolve())]
        item["require_prepared_data"] = True
    if getattr(args, "tasks", None):
        if len(spec["benchmarks"]) != 1:
            raise ValueError("--tasks requires a single-benchmark plan")
        item = next(iter(spec["benchmarks"].values()))
        item["tasks"] = args.tasks
        for key in ("demo_counts", "interaction_counts"):
            if key in item:
                item[key] = {task: count for task, count in item[key].items() if task in args.tasks}
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output).resolve() if getattr(args, "output", None) else None
    data = ((output / "datasets" if output else ROOT / "data/development_batches") / tag).resolve()
    for name, item in spec["benchmarks"].items():
        assign_downloaded_pool(name, item, data)
    configs = {name: read(ROOT / item["config"]) for name, item in spec["benchmarks"].items()}
    assignments = {name: data_assignment(name, item, configs[name]) for name, item in spec["benchmarks"].items()}
    batch = ((output / "batches" if output else ROOT / "outputs/development_batches") / tag).resolve()
    batch.mkdir(parents=True)
    data.mkdir(parents=True, exist_ok=True)
    checkout = batch / "inputs/checkout"
    checkout.mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, checkout / name)
    copy_tree(Path(cam.__file__).parent, checkout / "src/cam", ("legacy",))
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
        elif getattr(adapter, "asset_links", {}):
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
        task_configs = {}
        for task in tasks:
            task_config = task_configuration(config, assignments[name], task)
            if task_config != config:
                path = batch / "inputs/task_configs" / name / f"{task}.json"
                write(path, task_config)
                task_configs[task] = str(path)
        benches[name] = {"tasks": tasks, **assignments[name], "interaction_count": config["budget"],
                         "raw": str((ROOT / item.get("raw", f"data/raw/{name}")).resolve()), "source": str(target),
                         "source_hash": source_hash(adapter), "asset_hashes": assets,
                         "config": str(batch / "inputs" / f"{name}.json"),
                         "task_configs": task_configs,
                         "training_seed_policy": config.get("training_seed_policy", "demo_cycle"),
                         "new_data": str(data / name), "depth": config["depth"]}
    evaluation = {"enabled": getattr(args, "evaluate_on_freeze", False),
                  "workers_per_gpu": getattr(args, "evaluation_workers_per_gpu", 5),
                  "task_concurrency": 1}
    plan = {"created": time.time(), "tag": tag, "workers": args.workers, "gpus": args.gpus,
            "prepare_workers": getattr(args, "prepare_workers", 1),
            "selection_reason": spec["selection_reason"], "benchmarks": benches,
            "environment": env, "python": sys.executable, "runs": str((output / "dev_logs" if output else ROOT / "runs").resolve()),
            "harness_hash": tree_hash(checkout / "src/cam"),
            "held_out_evaluation": evaluation["enabled"], "evaluation": evaluation}
    write(batch / "plan.json", plan)
    shutil.copy2(Path(__file__), batch / "driver.py")
    (batch / "runner.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'inputs/checkout/src'))\n"
        "if __name__ == '__main__':\n    from cam.cli import main\n    raise SystemExit(main())\n"
    )
    rows = [f"| {name} | {cfg['demo_count']} | {cfg['interaction_count']} | {', '.join(cfg['tasks'])} |" for name, cfg in benches.items()]
    actual_counts = [f"- {name}/{task}: {count} assigned demos (target {cfg['demo_count']}), "
                     f"{cfg.get('interaction_counts', {}).get(task, cfg['interaction_count'])} training attempts."
                     for name, cfg in benches.items() for task, count in cfg.get("demo_counts", {}).items()]
    (batch / "README.md").write_text(
        f"# Coding development batch: {tag}\n\n"
        f"{sum(len(b['tasks']) for b in benches.values())} tasks, {args.workers} concurrent coding sessions; "
        "Exact clients/models are pinned in the per-benchmark configurations. Each task gets the "
        "demo and interaction counts below. Demo-only tasks allow no simulator attempts. "
        "For v2 tasks, the pinned seed policy chooses either new identities or the "
        "assigned demo identities; all exclude held-out tests. "
        "use the remaining attempts for verification even when the solution seems ready.\n\n"
        + spec["selection_reason"] + "\n\n| Benchmark | Demos/task | Max interactions/task | Selected tasks |\n|---|---|---|---|\n" + "\n".join(rows)
        + ("\n\nExplicit per-task assignments:\n\n" + "\n".join(actual_counts) if actual_counts else "")
        + "\n\n`plan.json` pins selection, configs, sources and data locations. "
        "`status.json` lists every task and run path; `logs/` retains exact commands and output. "
        "Each run has its own README, transcript, command audit, source history and checkpoints. "
        "The learner sees only its assigned demo H5, public API, own code and counted feedback. "
        "Source, private metadata, host files, networking, other tasks and test feedback are unavailable.\n\n"
        "Coding sessions run concurrently. A shared host lease permits one learning simulator per GPU. "
        "RoboMemArena demo preparation reserves one whole GPU while its replay workers run. "
        "Waiting for a GPU does not consume attempts. Missing demos are prepared before launching "
        "that task; data failures are reported, never silently replaced with another task. "
        + ("A separate host queue evaluates each completed, verified frozen policy on all held-out "
           "episodes at its final budget. One task is evaluated at a time with "
           f"{evaluation['workers_per_gpu']} episode workers per GPU ({len(args.gpus) * evaluation['workers_per_gpu']} total). "
           "Training keeps its one-simulator-per-GPU lease; evaluation workers run alongside it. "
           "Test results never enter learner feedback. Evaluation status and scored/expected counts "
           "are recorded separately from learning. Infrastructure failures retain evidence for recovery.\n\n"
           if evaluation["enabled"] else "No held-out evaluation is started.\n\n")
        + "Inspect: `.venv/bin/python scripts/develop_batch.py status BATCH`. "
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
        assert {r["seed"] for r in train}.isdisjoint(r["seed"] for r in tests)
        if plan["config"]["training_seed_policy"] == "disjoint_once":
            assert {r["seed"] for r in train}.isdisjoint(r["seed"] for r in demos)
            assert {r["episode"] for r in train}.isdisjoint(r["episode"] for r in demos)
            assert "disjoint from the assigned demonstrations; no repeats" in prompt
        else:
            assert plan["config"]["training_seed_policy"] == "demo_cycle"
            assert len(train) == len(demos)
            assert {(r["episode"], r["seed"]) for r in train} == {
                (r["episode"], r["seed"]) for r in demos
            }
    assert "Task-specific contract" not in prompt and "Success criterion" not in prompt
    private = [str(root), str(root / "inputs"), str(ROOT), plan["raw_data_reference"],
               "/runtime/cam/interface", "/runtime/cam/harness", "/feedback/../inputs"]
    script = """import json, os, socket, pathlib
private = PRIVATE
assert not any(pathlib.Path(p).exists() for p in private), private
assert not any(k.startswith(('AWS_', 'ANTHROPIC_', 'CAM_SIMULATOR_')) for k in os.environ)
files = list(pathlib.Path('/demonstrations').glob('*.h5'))
assert len(files) == 1
s = socket.socket(); s.settimeout(1)
try: result = s.connect_ex(('192.0.2.1', 443))
except OSError: result = -1
assert result != 0
s.close()
print(json.dumps({'passed': True, 'public_demo_episodes': DEMO_COUNT, 'demo_count_source': 'prepared_manifest', 'h5_opened': False, 'private_paths_hidden': True, 'network_unreachable': True}))
""".replace("PRIVATE", repr(private)).replace("DEMO_COUNT", str(demo_count))
    code, out, err = run(folder / "workspace", root / "runtime", ["python", "-c", script],
                         feedback=folder / "feedback", demonstrations=folder / "demonstrations", readonly=True)
    if code:
        raise RuntimeError("Boundary preflight failed: " + (out + err).decode(errors="replace"))
    result = json.loads(out)
    result.update(
        training_seed_policy=plan["config"]["training_seed_policy"],
        interaction_identities=interaction_count,
        disjoint_interaction_identities=(interaction_count if plan["config"]["training_seed_policy"] == "disjoint_once" else 0),
        demo_seed_identities=(interaction_count if plan["config"]["mode"] == "both" and plan["config"]["training_seed_policy"] == "demo_cycle" else 0),
        zero_test_overlap=True, simulator_attempts=0,
    )
    write(root / "orchestration/boundary_preflight.json", result)


def worker(batch):
    from cam.interface.base import source_hash
    from cam.interface.registry import get_adapter
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
        testing, learned = queue.Queue(), threading.Event()
        evaluation = plan.get("evaluation", {"enabled": False})

        def save(key=None, **changes):
            from cam.reporting.progress import write_progress
            with guard:
                if key:
                    tasks[key].update(changes, updated=time.time())
                current = {"status": "running", "pid": os.getpid(),
                           "updated": time.time(), "workers": plan["workers"], "tasks": tasks}
                write(batch / "status.json", current)
                write_progress(batch, "Development batch", current)

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
                boundary_check(root, task, cfg.get("demo_counts", {}).get(task, cfg.get("demo_count", 25)))
                save(f"{name}/{task}", status="queued", run=str(root))
                ready.put((name, task, root))

        def prepare_benchmark(name, cfg):
            def existing_group(dataset, group, group_cfg):
                try:
                    prepare_group(name, group_cfg, dataset, group)
                except Exception as exc:
                    for task in group:
                        if tasks[f"{name}/{task}"]["status"] == "preparing":
                            save(f"{name}/{task}", status="preparation_failed", error=str(exc))

            with ThreadPoolExecutor(max_workers=plan.get("prepare_workers", 1)) as pool:
                futures = [pool.submit(existing_group, dataset, group, group_cfg)
                           for dataset, group, group_cfg in preparation_groups(cfg)]
                for future in as_completed(futures):
                    future.result()
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
                    if completion["status"] == "finished" and evaluation["enabled"]:
                        try:
                            assignment = frozen_evaluation(root, task)
                            save(key, evaluation_status="queued", evaluation=assignment)
                            testing.put((name, task, root, assignment))
                        except Exception as exc:
                            save(key, evaluation_status="failed", evaluation_error=str(exc))
                    try:
                        command(f"audit_{name}_{task}", [plan["python"], str(batch / "audit_run.py"), str(root), "--task", task])
                        save(key, audit_status="finished")
                    except Exception as exc:
                        save(key, audit_status="failed", audit_error=str(exc))
                except Exception as exc:
                    save(key, status="failed", error=str(exc))
                finally:
                    ready.task_done()

        def evaluate():
            from cam.reporting.report import build

            while not learned.is_set() or not testing.empty():
                try:
                    name, task, root, assignment = testing.get(timeout=1)
                except queue.Empty:
                    continue
                key = f"{name}/{task}"
                try:
                    if frozen_evaluation(root, task) != assignment:
                        raise ValueError("Frozen assignment changed while queued")
                    save(key, evaluation_status="running", evaluation_started=time.time())
                    command(f"evaluate_{name}_{task}", [plan["python"], str(root / "runner.py"),
                            "evaluate", str(root), "--budgets", str(assignment["budget"]),
                            "--gpus", *map(str, plan["gpus"]), "--workers-per-gpu",
                            str(evaluation["workers_per_gpu"])])
                    summary = next(row for row in build(root)["settings"] if row["budget"] == assignment["budget"])
                    counts = {field: summary[field] for field in (
                        "scored", "expected", "successes", "success_rate", "infrastructure_errors")}
                    complete = counts["scored"] == counts["expected"] == assignment["expected"] and not counts["infrastructure_errors"]
                    save(key, evaluation_status="finished" if complete else "needs_attention",
                         evaluation={**assignment, **counts}, evaluation_finished=time.time())
                except Exception as exc:
                    save(key, evaluation_status="failed", evaluation_error=str(exc))
                    # Keep partial counts, including infrastructure failures, separate from learning.
                    try:
                        summary = next(row for row in build(root)["settings"] if row["budget"] == assignment["budget"])
                        save(key, evaluation={**assignment, **{field: summary[field] for field in (
                            "scored", "expected", "successes", "success_rate", "infrastructure_errors")}})
                    except Exception as report_error:
                        save(key, evaluation_report_error=str(report_error))
                finally:
                    testing.task_done()

        save()
        tester = threading.Thread(target=evaluate, name="held-out-evaluation") if evaluation["enabled"] else None
        if tester:
            tester.start()
        try:
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
        finally:
            learned.set()
            if tester:
                tester.join()
        complete = all(t["status"] == "finished" and t.get("audit_status") == "finished"
                       and (not evaluation["enabled"] or t.get("evaluation_status") == "finished")
                       for t in tasks.values())
        final = {"status": "finished" if complete else "needs_attention",
                 "finished": time.time(), "workers": plan["workers"], "tasks": tasks}
        write(batch / "status.json", final)
        from cam.reporting.progress import write_progress
        write_progress(batch, "Development batch", final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("start")
    p.add_argument("plan", type=Path)
    p.add_argument("--dataset", type=Path, help="Downloaded demonstration directory containing manifest.json")
    p.add_argument("--output", type=Path, help="Keep batch state, development logs and rollouts beneath this directory")
    p.add_argument("--tasks", nargs="+", help="Optionally select tasks from a single-benchmark plan")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prepare-workers", type=int, default=1,
                   help="Concurrent dataset preparations; use per-task datasets to start learning progressively")
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--evaluate-on-freeze", action="store_true",
                   help="Evaluate completed frozen policies in a separate host queue")
    p.add_argument("--evaluation-workers-per-gpu", type=int, default=5)
    for name in ("_worker", "status"):
        commands.add_parser(name).add_argument("batch", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    source = ROOT / "src" if args.command == "start" else args.batch.resolve() / "inputs/checkout/src"
    sys.path.insert(0, str(source))
    if args.command == "start":
        if min(args.workers, args.prepare_workers, args.evaluation_workers_per_gpu) < 1 or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
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
