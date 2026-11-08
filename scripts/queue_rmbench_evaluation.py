"""Evaluate all ten frozen RMBench RGB-D demo-only policies after development.

Supports an all-task barrier or --as-ready per-task admission. Uses each run's pinned CLI,
all assigned test episodes, and strict GPU leases (one task per GPU, <=6 workers).
Only unscorable infrastructure episodes may be archived and retried identically.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def load_checks(path):
    spec = importlib.util.spec_from_file_location("evaluation_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready_policies(status, expected, checks, per_task=False):
    """Validate only completed learners; preserve the historical all-task gate."""
    from cam.io import read

    if not per_task and status["status"] != "finished":
        raise ValueError("All development must finish before held-out evaluation")
    rows = list(status["tasks"].values())
    if len(rows) != len(expected) or {r["task"] for r in rows} != set(expected):
        raise ValueError("Development does not cover the exact assigned task set")
    roots, frozen = {}, {}
    for row in rows:
        task = row["task"]
        if per_task and row["status"] != "finished":
            continue
        if row["status"] != "finished" or row["benchmark"] != "rmbench":
            raise ValueError(f"Learner unfinished or benchmark differs: {task}")
        root = Path(row["run"])
        plan = read(root / "run.json")
        cfg = plan["config"]
        if cfg["benchmark"] != "rmbench" or not cfg["depth"]:
            raise ValueError(f"Expected RMBench RGB-D: {task}")
        if len(plan["test_episodes"][task]) != expected[task]:
            raise ValueError(f"Test episode count changed: {task}")
        frozen[task] = checks.frozen_policy(root, task)
        roots[task] = root
    return roots, frozen


def aggregate(results, expected):
    scored = sum(r["scored"] for r in results.values())
    successes = sum(r["successes"] for r in results.values())
    complete = set(results) == set(expected) and all(r["complete"] for r in results.values())
    total = sum(expected.values())
    return {"tasks": results, "expected": total, "scored": scored, "successes": successes,
            "complete": complete, "success_rate": successes / total if complete else None,
            "unresolved_infrastructure": sum(len(r["infrastructure_episodes"]) for r in results.values())}


def start(args):
    from cam.io import read, sha256, write

    development = args.development.resolve()
    experiment = read(development / "experiment.json")
    if set(experiment["benchmarks"]) != {"rmbench"}:
        raise ValueError("Expected the RMBench development queue")
    item = experiment["benchmarks"]["rmbench"]
    config = read(item["config"])
    if (config["mode"], config["budget"], config["depth"]) != ("demos", 0, True):
        raise ValueError("Expected RGB-D demo-only development")
    previous = None
    if (development / "evaluation_followup.json").exists():
        previous = read(development / "evaluation_followup.json")
        if not args.supersedes or Path(previous["job"]).resolve() != args.supersedes.resolve():
            raise FileExistsError("Use --supersedes to replace the waiting evaluation follow-up")
        if read(Path(previous["job"]) / "status.json")["status"] != "waiting_for_development":
            raise ValueError("Only a waiting evaluator can be replaced; preserve active evaluations")
    elif args.supersedes:
        raise ValueError("No matching prior evaluation follow-up")
    from cam.benchmarks.registry import get_adapter

    adapter = get_adapter("rmbench")
    if item["tasks"] != adapter.tasks():
        raise ValueError("Expected all ten RMBench tasks")
    expected = {task: len(adapter.records(task, "test")) for task in item["tasks"]}
    if len(expected) != 10 or any(n != 100 for n in expected.values()):
        raise ValueError("Expected 100 test episodes for each of ten tasks")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    job = ROOT / "outputs/rmbench_evaluation_queue" / tag
    job.mkdir(parents=True)
    shutil.copy2(Path(__file__), job / "driver.py")
    # Reuse tested, host-only freeze/score checks and strict GPU admission.
    shutil.copy2(ROOT / "scripts/rmbench_evaluation_checks.py", job / "evaluation_checks.py")
    plan = {"created": time.time(), "root": str(ROOT), "development_queue": str(development),
            "development_plan_sha256": sha256(development / "plan.json"),
            "development_experiment_sha256": sha256(development / "experiment.json"),
            "expected": expected, "gpus": args.gpus, "workers_per_gpu": args.workers_per_gpu,
            "release_policy": "per_task" if args.as_ready else "all_tasks",
            "supersedes": previous["job"] if previous else None,
            "maximum_passes": 3, "python": sys.executable,
            "pinned_files": {name: sha256(job / name) for name in ("driver.py", "evaluation_checks.py")}}
    write(job / "plan.json", plan)
    (job / "README.md").write_text(
        f"# RMBench held-out evaluation follow-up: {tag}\n\n"
        f"Follow `{development}`. "
        + ("Admit each task as soon as its own demo-only learner finishes and its policy is frozen. Other tasks may continue learning. "
           if args.as_ready else "Wait for all ten demo-only learners to finish. ")
        + "Verify the admitted task's frozen policy and zero training attempts before its first test episode. Evaluate all 100 assigned "
        "test episodes per task (1,000 total), using each run's unchanged pinned runner and R000 policy.\n\n"
        f"GPUs {args.gpus}, one task per GPU, {args.workers_per_gpu} episode workers per GPU. "
        "Strict admission waits for idle GPU compute and queued MIKASA RGB-D simulation work; "
        "leases coordinate with RMBench collection. No dataset-generation sharing exception applies.\n\n"
        "Only unscorable infrastructure episodes are archived with cam recover-test before an "
        "identical retry; at most three evaluation passes per task. Scored successes and failures "
        "are retained. Test outcomes are never sent to learners. No policy changes or new learning "
        "are triggered by tests. results.json reports scored/expected coverage and a final success "
        "rate only when all 1,000 episodes are scored; status.json and logs/ retain progress/errors. "
        "Per-run reports appear on the existing port-8915 dashboard.\n"
    )
    write(job / "status.json", {"status": "waiting_for_development", "expected": 1000,
                                "development_queue": str(development), "updated": time.time()})
    if previous:
        old = Path(previous["job"])
        pid = read(old / "process.json")["pid"]
        proc_dir = Path('/proc') / str(pid)
        if proc_dir.exists():
            argv = (proc_dir / 'cmdline').read_bytes().split(b'\0')
            if b'_worker' not in argv or Path(argv[argv.index(b'_worker') + 1].decode()).resolve() != old.resolve():
                raise ValueError("Old evaluator process identity differs")
            os.kill(pid, signal.SIGSTOP)
            old_state = read(old / "status.json")
            if old_state["status"] != "waiting_for_development":
                os.kill(pid, signal.SIGCONT)
                raise ValueError("Old evaluator advanced; preserve active work")
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
            for _ in range(50):
                try:
                    ended = (proc_dir / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z'
                except FileNotFoundError:
                    ended = True
                if ended:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("Old evaluator did not stop")
        else:
            old_state = read(old / "status.json")
        write(old / "status_before_per_task_handover.json", old_state)
        write(old / "status.json", {"status": "superseded", "successor": str(job), "updated": time.time()})
        write(old / "successor.json", {"job": str(job), "reason": "User requested evaluation as each policy finishes"})
        write(development / f"evaluation_followup_before_{tag}.json", previous)
    with (job / "supervisor.log").open("ab") as log:
        proc = subprocess.Popen([sys.executable, str(job / "driver.py"), "_worker", str(job)],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=ROOT)
    write(job / "process.json", {"pid": proc.pid, "started": time.time()})
    result = {"job": str(job), "pid": proc.pid, "expected": 1000}
    write(development / "evaluation_followup.json", result)
    return result


def progressive_worker(job, plan, checks):
    """Schedule frozen tasks independently; no test outcome enters development."""
    from cam.io import read, write, sha256

    development = Path(plan["development_queue"])
    expected = plan["expected"]
    results, frozen, roots = {}, {}, {}
    tasks = {task: {"status": "waiting_for_learner"} for task in expected}
    guard = threading.Lock()

    def update(task=None, **changes):
        with guard:
            if task:
                tasks[task].update(changes)
            write(job / "status.json", {"status": "following_development", "updated": time.time(),
                "release_policy": "per_task", "tasks": tasks,
                **{k: v for k, v in aggregate(results, expected).items() if k != "tasks"}})

    def command(label, argv, allow_failure=False):
        path = job / "logs" / f"{label}_{time.time_ns()}"
        write(path.with_suffix(".command.json"), {"argv": argv, "started": time.time()})
        with path.with_suffix(".log").open("xb") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            write(path.with_suffix(".process.json"), {"pid": proc.pid, "started": time.time()})
            code = proc.wait()
        write(path.with_suffix(".completion.json"), {"returncode": code, "finished": time.time()})
        if code and not allow_failure:
            raise RuntimeError(f"Command exited {code}: {path.with_suffix('.log')}")

    def evaluate(task, root, freeze, gpu):
        gpu_job = job / f"gpu_{gpu}"
        gpu_job.mkdir(exist_ok=True)
        update(task, status="waiting_for_gpu", gpu=gpu, run=str(root))
        with checks.gpu_lease(gpu_job, gpu, "evaluation", allow_sharing=False):
            for attempt in range(plan["maximum_passes"]):
                if checks.frozen_policy(root, task) != freeze:
                    raise ValueError(f"Frozen policy changed after admission: {task}")
                summary = checks.score_summary(root, task)
                if summary["complete"]:
                    break
                for episode in summary["infrastructure_episodes"]:
                    command(f"recover_{task}_{episode}", [plan["python"], str(root / "runner.py"),
                        "recover-test", str(root), task, str(episode), "--budget", "0", "--reason",
                        "Unscorable infrastructure episode; retry unchanged frozen policy, seed and limits under exclusive GPU scheduling."])
                update(task, status="evaluating", pass_number=attempt + 1)
                command(f"evaluate_{task}", [plan["python"], str(root / "runner.py"), "evaluate", str(root),
                    "--gpus", str(gpu), "--workers-per-gpu", str(plan["workers_per_gpu"]),
                    "--budgets", "0", "--resume"], allow_failure=True)
            return checks.score_summary(root, task)

    submitted, active = set(), {}
    update()
    with ThreadPoolExecutor(max_workers=len(plan["gpus"])) as executor:
        while True:
            for name, key in (("plan.json", "development_plan_sha256"),
                              ("experiment.json", "development_experiment_sha256")):
                if sha256(development / name) != plan[key]:
                    raise ValueError(f"Development input changed: {name}")
            batch_path = development / "batch.json"
            batch = Path(read(batch_path)["batch"]) if batch_path.exists() else development
            state = read(batch / "status.json")
            if "tasks" in state:
                ready, completed = ready_policies(state, expected, checks, per_task=True)
            else:
                ready, completed = {}, {}
            for gpu, (future, task) in list(active.items()):
                if not future.done():
                    continue
                try:
                    result = future.result()
                    with guard:
                        results[task] = result
                        write(job / "results.json", aggregate(results, expected))
                    update(task, status="finished" if result["complete"] else "needs_attention")
                except Exception:
                    update(task, status="needs_attention", error=traceback.format_exc())
                del active[gpu]
            for gpu in plan["gpus"]:
                if gpu in active:
                    continue
                task = next((t for t in expected if t in ready and t not in submitted), None)
                if task is None:
                    continue
                roots[task], frozen[task] = ready[task], completed[task]
                write(job / "frozen_policies.json", frozen)
                write(job / "development_batch.json", {"batch": str(batch), "runs": {t: str(r) for t, r in roots.items()}})
                submitted.add(task)
                active[gpu] = (executor.submit(evaluate, task, roots[task], frozen[task], gpu), task)
            stopped = not checks.alive(read(batch / "process.json")["pid"])
            if state["status"] in ("finished", "needs_attention", "superseded") or stopped:
                for task in expected:
                    if task not in ready and task not in submitted:
                        submitted.add(task)
                        update(task, status="development_incomplete")
            if len(submitted) == len(expected) and not active:
                break
            update()
            time.sleep(10)
    final = aggregate(results, expected)
    write(job / "results.json", final)
    write(job / "status.json", {"status": "finished" if final["complete"] else "needs_attention",
                                "updated": time.time(), "release_policy": "per_task", "task_status": tasks, **final})


def worker(job):
    from cam.io import read, sha256, write

    plan = read(job / "plan.json")
    development = Path(plan["development_queue"])
    checks = load_checks(job / "evaluation_checks.py")
    with (development / ".evaluation_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, digest in plan["pinned_files"].items():
            if sha256(job / name) != digest:
                raise ValueError(f"Evaluation supervisor changed: {name}")
        if plan.get("release_policy") == "per_task":
            return progressive_worker(job, plan, checks)
        while True:
            for name, key in (("plan.json", "development_plan_sha256"),
                              ("experiment.json", "development_experiment_sha256")):
                if sha256(development / name) != plan[key]:
                    raise ValueError(f"Queued development input changed: {name}")
            state = read(development / "status.json")
            if state["status"] == "needs_attention":
                raise RuntimeError("Development queue needs attention; no tests started")
            launched = development / "batch.json"
            if launched.exists():
                batch = Path(read(launched)["batch"])
                if (batch / "status.json").exists():
                    state = read(batch / "status.json")
                    if state["status"] == "finished":
                        break
                    if state["status"] == "needs_attention":
                        raise RuntimeError("Not all learners finished; no tests started")
                    if not checks.alive(read(batch / "process.json")["pid"]):
                        raise RuntimeError("Development supervisor stopped; no tests started")
            elif not checks.alive(read(development / "process.json")["pid"]):
                raise RuntimeError("Development queue stopped; no tests started")
            write(job / "status.json", {"status": "waiting_for_development", "updated": time.time(),
                                        "expected": sum(plan["expected"].values()), "development_status": state["status"]})
            time.sleep(20)
        roots, frozen = ready_policies(state, plan["expected"], checks)
        write(job / "frozen_policies.json", frozen)
        write(job / "development_batch.json", {"batch": str(batch), "runs": {t: str(r) for t, r in roots.items()}})
        pending = queue.Queue()
        for task in plan["expected"]:
            pending.put(task)
        results, tasks = {}, {task: {"status": "queued"} for task in roots}
        guard = threading.Lock()

        def update(task=None, **changes):
            with guard:
                if task:
                    tasks[task].update(changes)
                write(job / "status.json", {"status": "evaluating", "updated": time.time(),
                                            "development_batch": str(batch), "tasks": tasks,
                                            **{k: v for k, v in aggregate(results, plan["expected"]).items() if k != "tasks"}})

        def command(label, argv, allow_failure=False):
            path = job / "logs" / f"{label}_{time.time_ns()}"
            write(path.with_suffix(".command.json"), {"argv": argv, "started": time.time()})
            env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            with path.with_suffix(".log").open("xb") as log:
                proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
                write(path.with_suffix(".process.json"), {"pid": proc.pid, "started": time.time()})
                code = proc.wait()
            write(path.with_suffix(".completion.json"), {"returncode": code, "finished": time.time()})
            if code and not allow_failure:
                raise RuntimeError(f"Command exited {code}: {path.with_suffix('.log')}")
            return code

        def evaluate_gpu(gpu):
            while True:
                try:
                    task = pending.get_nowait()
                except queue.Empty:
                    return
                root = roots[task]
                update(task, status="waiting_for_gpu", gpu=gpu)
                gpu_job = job / f"gpu_{gpu}"
                gpu_job.mkdir(exist_ok=True)
                with checks.gpu_lease(gpu_job, gpu, "evaluation", allow_sharing=False):
                    for attempt in range(plan["maximum_passes"]):
                        if checks.frozen_policy(root, task) != frozen[task]:
                            raise ValueError(f"Frozen policy changed after all-task barrier: {task}")
                        summary = checks.score_summary(root, task)
                        if summary["complete"]:
                            break
                        for episode in summary["infrastructure_episodes"]:
                            command(f"recover_{task}_{episode}", [plan["python"], str(root / "runner.py"),
                                "recover-test", str(root), task, str(episode), "--budget", "0", "--reason",
                                "Unscorable infrastructure episode; retry unchanged frozen policy, seed and limits under exclusive GPU scheduling."])
                        update(task, status="evaluating", pass_number=attempt + 1)
                        command(f"evaluate_{task}", [plan["python"], str(root / "runner.py"), "evaluate", str(root),
                            "--gpus", str(gpu), "--workers-per-gpu", str(plan["workers_per_gpu"]),
                            "--budgets", "0", "--resume"], allow_failure=True)
                    result = checks.score_summary(root, task)
                    with guard:
                        results[task] = result
                        write(job / "results.json", aggregate(results, plan["expected"]))
                    update(task, status="finished" if result["complete"] else "needs_attention")
                pending.task_done()

        update()
        with ThreadPoolExecutor(max_workers=len(plan["gpus"])) as pool:
            list(pool.map(evaluate_gpu, plan["gpus"]))
        final = aggregate(results, plan["expected"])
        write(job / "results.json", final)
        write(job / "status.json", {"status": "finished" if final["complete"] else "needs_attention",
                                    "updated": time.time(), "development_batch": str(batch), **final})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("start")
    p.add_argument("development", type=Path)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--workers-per-gpu", type=int, default=6)
    p.add_argument("--as-ready", action="store_true", help="Evaluate each task after its own learner finishes and freezes")
    p.add_argument("--supersedes", type=Path, help="Replace only a still-waiting evaluation scheduler")
    commands.add_parser("_worker").add_argument("job", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        ROOT = Path(json.loads((args.job / "plan.json").read_text())["root"])
    sys.path.insert(0, str(ROOT / "src"))
    sys.dont_write_bytecode = True
    if args.command == "start":
        if (not 1 <= args.workers_per_gpu <= 6 or not args.gpus
                or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)):
            parser.error("Use distinct nonnegative GPUs and one to six workers per GPU")
        print(json.dumps(start(args), indent=2))
    else:
        try:
            worker(args.job)
        except Exception:
            from cam.io import write
            write(args.job / "status.json", {"status": "needs_attention", "updated": time.time(),
                                             "error": traceback.format_exc()})
            raise
