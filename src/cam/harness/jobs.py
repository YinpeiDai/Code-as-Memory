"""Pinned, resumable coding and evaluation queues with one shared GPU slot pool."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from cam.config import CodingConfig, default_evaluation_budgets
from cam.core.history import freeze, select_checkpoint
from cam.harness.isolation import run, terminate
from cam.harness.preflight import transport_preflight
from cam.harness.prepare import STARTER, pinned_environment, starter, validate_run
from cam.io import read, sha256, write
from cam.providers.clients import invocation, unexpected_tool
from cam.providers.usage import summarize_sessions


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def preflight(root, full_data=False):
    """Stream a long synthetic reset for every distinct declared task interface."""
    root = Path(root)
    plan = validate_run(root, full_data=full_data)
    path = root / "preflight.json"
    interfaces = plan.get("interfaces") or {None: None}
    key = json.dumps(sorted(interfaces.items(), key=lambda item: str(item[0])), sort_keys=True)
    digest = __import__("hashlib").sha256(key.encode()).hexdigest()
    with locked(root / "locks/preflight.lock"):
        if path.exists():
            result = read(path)
            if (
                result["runtime_hash"] == plan["runtime_hash"]
                and result.get("interfaces_sha256", digest) == digest
                and result["passed"]
            ):
                return result
        checks, seen = {}, {}
        for task, interface in interfaces.items():
            signature = json.dumps(interface, sort_keys=True)
            if signature not in seen:
                seen[signature] = transport_preflight(
                    root / "runtime",
                    parent=root,
                    include_depth=plan["config"]["depth"],
                    timeout=120,
                    interface=interface,
                    python=plan.get("policy_python"),
                )
            checks[str(task)] = seen[signature]
        result = {
            "passed": all(check["passed"] for check in checks.values()),
            "runtime_hash": plan["runtime_hash"],
            "interfaces_sha256": digest,
            "checks": checks,
        }
        write(path, result)
        if not result["passed"]:
            raise RuntimeError("Transport preflight failed; inspect " + str(path))
    return result


def learn_task(root, task, gpu, resume=False):
    root = Path(root).resolve()
    plan = validate_run(root)
    task_root = root / "tasks" / task
    if task not in plan["tasks"] or (task_root / "policy").exists():
        raise ValueError("Task is unassigned or already frozen")
    coding = CodingConfig(**plan["config"]["coding"])
    logs = task_root / "logs"
    with locked(task_root / "state/learning.lock"):
        previous = sorted(logs.glob("session_*"))
        if previous and not resume:
            raise ValueError(
                "Prior coding logs exist. Use --resume to retain them and continue the same budget."
            )
        session = logs / f"session_{len(previous):03d}"
        session.mkdir()
        prompt = (task_root / "prompt.md").read_text()
        if previous:
            prompt += (
                "\nContinue your existing unfinished workspace. Read usage first; the attempt budget "
                "and allowed data have not changed. Inspect your logs and feedback, then finish offline "
                "if the simulator budget is exhausted. All earlier sessions are retained.\n"
            )
        (session / "prompt.md").write_text(prompt)
        call = invocation(coding, root, task, gpu, session)
        if coding.agent == "codex":
            from cam.providers.probe import probe_codex

            probe_folder = session / "offline_probe"
            probe_folder.mkdir()
            probe_codex(call, probe_folder)
        write(
            session / "launch.json",
            {
                "agent": coding.agent,
                "model": coding.model,
                "provider": coding.provider,
                "gpu": gpu,
                "started": time.time(),
                "prompt_sha256": sha256(session / "prompt.md"),
                "argv": call.argv,
                "cli_version": subprocess.check_output(
                    [call.argv[0], "--version"], text=True
                ).strip(),
            },
        )
        proc = None
        status, detail, code = "failed", None, None
        started = time.monotonic()
        try:
            if coding.agent == "claude_code":
                from cam.providers.probe import probe_cam

                probe_cam(session / "mcp.json", session / "mcp_probe")
            with (
                (session / "events.jsonl").open("wb") as output,
                (session / "stderr.txt").open("wb") as errors,
            ):
                proc = subprocess.Popen(
                    call.argv,
                    cwd=call.cwd,
                    env=call.environment,
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=errors,
                    start_new_session=True,
                )
                write(
                    session / "process.json", {"pid": proc.pid, "started": time.time()}
                )
                proc.stdin.write(prompt.encode())
                proc.stdin.close()
                with (session / "events.jsonl").open() as reader:

                    def consume():
                        while True:
                            offset = reader.tell()
                            line = reader.readline()
                            if not line.endswith("\n"):
                                reader.seek(offset)
                                return
                            try:
                                entry = json.loads(line)
                            except ValueError:
                                continue
                            violation = unexpected_tool(coding.agent, entry)
                            if violation:
                                raise RuntimeError(violation)

                    last_save = 0
                    while proc.poll() is None:
                        consume()
                        if time.monotonic() - last_save > 10:
                            summarize_sessions(logs, coding.agent)
                            last_save = time.monotonic()
                        if time.monotonic() - started > coding.timeout:
                            raise TimeoutError(
                                "Coding session wall-clock limit exceeded"
                            )
                        time.sleep(0.5)
                    code = proc.wait()
                    consume()
            if code:
                raise RuntimeError(
                    f"Coding client exited {code}; inspect {session / 'stderr.txt'}"
                )
            # Model-generated Python is never imported into the trusted launcher.
            from cam.harness.service import Service

            service = Service(root, task, gpu)
            try:
                service.sync("final")
                service.archive("final")
                valid, out, err = run(
                    service.public,
                    service.runtime,
                    ["python3.11", "-m", "cam.runtime.policy_worker", "--validate"],
                    timeout=120,
                    readonly=True,
                )
                if valid:
                    raise ValueError(
                        "Final isolated validation failed: "
                        + (out + err).decode(errors="replace")[-4000:]
                    )
                blank = (
                    starter(plan["interfaces"][task])
                    if "interfaces" in plan
                    else STARTER
                )
                policy_file = task_root / "program/tools/policy.py"
                if policy_file.exists() and policy_file.read_text() == blank:
                    raise ValueError("Agent left the starter policy unchanged")
                freeze(service.cfg)
            finally:
                service.lock.close()
            status = "finished"
        except Exception as exc:
            detail = str(exc)
        finally:
            if proc is not None:
                terminate(proc)
            usage = summarize_sessions(logs, coding.agent)
            result = {
                "status": status,
                "error": detail,
                "returncode": code,
                "seconds": time.monotonic() - started,
                "finished": time.time(),
                "usage": usage,
            }
            write(session / "completion.json", result)
            write(logs / "completion.json", result)
        return result


def run_queue(root, name, jobs, slots, worker, resume=False):
    """Slots are held for a complete subprocess; no accidental per-mode oversubscription."""
    root = Path(root)
    if not slots:
        raise ValueError("At least one worker slot is required")
    with locked(root / "locks" / (name + ".lock")):
        folder = root / "queues" / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "status.json").exists() and not resume:
            raise ValueError(
                "Queue exists; use --resume to continue without overwriting results"
            )
        if (folder / "status.json").exists():
            old = read(folder / "status.json")
            write(folder / f"status_before_{time.time_ns()}.json", old)
        state = {
            "pid": os.getpid(),
            "started": time.time(),
            "slots": slots,
            "jobs": {job[0]: {"status": "pending"} for job in jobs},
        }
        if len(state["jobs"]) != len(jobs):
            raise ValueError("Duplicate queue jobs")
        mutex, pending, stop = threading.Lock(), queue.Queue(), threading.Event()
        for job in jobs:
            pending.put(job)

        def save():
            from cam.reporting.progress import write_progress
            state["updated"] = time.time()
            write(folder / "status.json", state)
            write_progress(folder, name.capitalize(), state)

        save()

        def slot(gpu):
            while not stop.is_set():
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                key = job[0]
                with mutex:
                    state["jobs"][key] = {
                        "status": "running",
                        "gpu": gpu,
                        "started": time.time(),
                    }
                    save()
                try:
                    result = worker(job, gpu, folder)
                except Exception as exc:
                    result = {"status": "failed", "error": str(exc)}
                if result.get("stop_queue") or result["status"] == "failed":
                    stop.set()
                with mutex:
                    state["jobs"][key].update(result, finished=time.time())
                    save()

        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            list(pool.map(slot, slots))
        state["finished"] = time.time()
        state["status"] = "needs_attention" if stop.is_set() else "finished"
        save()
        return state


def subprocess_job(root, argv, log, gpu, timeout, python=None):
    env = pinned_environment(root)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    with log.open("xb") as stream:
        proc = subprocess.Popen(
            [python or sys.executable, str(root / "runner.py"), *argv],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout)
        except BaseException:
            terminate(proc)
            raise


def learn(root, gpus=(0, 1), workers_per_gpu=None, tasks=None, resume=False, *, workers=8):
    root = Path(root).resolve()
    plan = validate_run(root)
    preflight(root)
    tasks = tasks or plan["tasks"]
    if set(tasks) - set(plan["tasks"]) or len(set(tasks)) != len(tasks):
        raise ValueError("Unassigned or duplicate tasks")
    slots = gpu_slots(gpus, workers_per_gpu) if workers_per_gpu is not None else development_slots(gpus, workers)
    jobs = [
        (task,)
        for task in tasks
        if not (root / "tasks" / task / "policy/freeze.json").exists()
    ]

    def worker(job, gpu, folder):
        task = job[0]
        argv = ["_learn-task", str(root), task, "--gpu", str(gpu)] + (
            ["--resume"] if resume else []
        )
        code = subprocess_job(
            root,
            argv,
            folder / f"{task}_{time.time_ns()}.log",
            gpu,
            plan["config"]["coding"]["timeout"] + 120,
        )
        path = root / "tasks" / task / "logs/completion.json"
        return (
            read(path)
            if path.exists()
            else {"status": "failed", "error": f"Worker exit {code}"}
        )

    return run_queue(root, "learning", jobs, slots, worker, resume)


def gpu_slots(gpus, workers_per_gpu):
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0 or workers_per_gpu < 1:
        raise ValueError(
            "Choose distinct nonnegative GPUs and positive workers per GPU"
        )
    return [gpu for gpu in gpus for _ in range(workers_per_gpu)]


def development_slots(gpus, workers=8):
    gpu_slots(gpus, 1)
    if workers < 1:
        raise ValueError("Choose a positive total number of development workers")
    return [gpus[index % len(gpus)] for index in range(workers)]


def evaluate(
    root, budgets=None, gpus=(0, 1), workers_per_gpu=5, tasks=None, resume=False
):
    root = Path(root).resolve()
    plan = validate_run(root)
    preflight(root)
    tasks = tasks or plan["tasks"]
    budgets = (
        budgets
        if budgets is not None
        else default_evaluation_budgets(plan["config"])
    )
    if (
        len(set(budgets)) != len(budgets)
        or not budgets
        or min(budgets) < 0
        or max(budgets) > plan["config"]["budget"]
        or set(tasks) - set(plan["tasks"])
    ):
        raise ValueError("Choose valid tasks and budgets within the development limit")
    existing = (
        read(root / "evaluation_plan.json")
        if (root / "evaluation_plan.json").exists()
        else {"budgets": []}
    )
    write(
        root / "evaluation_plan.json",
        {
            "budgets": sorted(set(existing["budgets"]) | set(budgets)),
            "expected_tasks": plan["tasks"],
            "test_episodes": plan["test_episodes"],
        },
    )
    jobs = []
    for task in tasks:
        for maximum in budgets:
            _, frozen = select_checkpoint(root / "tasks" / task, maximum)
            for episode in plan["test_episodes"][task]:
                key = f"{task}_R{maximum:03d}_ep{episode:03d}"
                folder = (
                    root
                    / "tasks"
                    / task
                    / "rollouts/test"
                    / f"R{maximum:03d}"
                    / f"ep{episode:03d}"
                )
                if folder.exists():
                    path = folder / "result.json"
                    if not path.exists():
                        raise ValueError(
                            f"Incomplete episode retained at {folder}; inspect before retrying"
                        )
                    result = read(path)
                    if result["program_hash"] != frozen["program_hash"]:
                        raise ValueError(
                            "Existing test result has a different policy hash"
                        )
                    if not result["scorable"]:
                        raise ValueError(
                            f"Infrastructure error at {folder}; use cam recover-test after diagnosis"
                        )
                    if resume:
                        continue
                    raise FileExistsError("Evaluation results exist; use --resume")
                jobs.append((key, task, maximum, episode, folder))

    def worker(job, gpu, queue_folder):
        key, task, maximum, episode, folder = job
        code = subprocess_job(
            root,
            [
                "_episode",
                str(root),
                task,
                str(episode),
                "--split",
                "test",
                "--budget",
                str(maximum),
            ],
            queue_folder / f"{key}_{time.time_ns()}.log",
            gpu,
            plan["config"]["policy_timeout"] + 600,
            python=plan.get("simulator_python"),
        )
        path = folder / "result.json"
        if code or not path.exists():
            return {
                "status": "failed",
                "error": f"Worker exit {code}; inspect logs",
                "stop_queue": True,
            }
        result = read(path)
        return {
            "status": "finished" if result["scorable"] else "failed",
            "outcome": result["status"],
            "stop_queue": not result["scorable"],
        }

    state = run_queue(
        root, "evaluation", jobs, gpu_slots(gpus, workers_per_gpu), worker, resume
    )
    from cam.reporting.report import build

    build(root)
    return state
