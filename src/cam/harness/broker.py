"""One trusted simulator episode; the generated policy sees sensors only."""

import ctypes
from pathlib import Path
import selectors
import signal
import subprocess
import threading
import time
import traceback

import numpy as np

from cam.benchmarks.registry import get_adapter
from cam.core.budget import Ledger, program_hash
from cam.core.history import select_checkpoint
from cam.harness.capture import Capture
from cam.harness.isolation import command, limits, terminate
from cam.harness.preflight import stream_initial
from cam.io import read, write
from cam.runtime.wire import observation_fields, receive, send, encoded

TERMINAL = ("success", "fail", "timeout", "error")


class PolicyError(Exception):
    pass


def policy_limits():
    limits()
    ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0)


def execute(
    adapter,
    metadata,
    task,
    episode,
    split,
    program,
    runtime,
    folder,
    config,
    progress=None,
):
    """Same execution and recording path for training and held-out testing."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.time()
    env = worker = capture = stderr = timer = None
    expired = threading.Event()
    status, steps, calls, rows, actions = "harness_error", 0, 0, [], []
    goal = error = None
    stage = "environment_creation"
    try:
        env = adapter.make_environment(metadata, task, episode, split, config["depth"])
        stage = "environment_reset"
        initial = adapter.begin(
            env, builtin_demo=config["builtin_demo"], depth=config["depth"]
        )
        goal, frames, context = initial.goal, initial.frames, initial.context
        capture = Capture(folder)
        capture.append(frames)
        stage = "worker_initialization"
        stderr = (folder / "policy_stderr.txt").open("wb")
        worker = subprocess.Popen(
            [
                "/usr/bin/strace",
                "-f",
                "-qq",
                "-e",
                "trace=%file,%network,%process",
                "-o",
                str(folder / "private_policy_access.strace"),
                *command(
                    program,
                    runtime,
                    ["python3.11", "-m", "cam.runtime.policy_worker"],
                    readonly=True,
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=policy_limits,
            bufsize=0,
        )

        def expire():
            expired.set()
            terminate(worker)

        timer = threading.Timer(config["policy_timeout"], expire)
        timer.daemon = True
        timer.start()
        with selectors.DefaultSelector() as selector:
            selector.register(worker.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=30):
                raise TimeoutError("Isolated worker readiness timeout")
        ready = receive(worker.stdout, 2 * 1024**2)
        if ready != {"kind": "ready"}:
            if ready.get("status") == "policy_error":
                raise PolicyError(ready.get("error"))
            raise ValueError("Worker readiness protocol mismatch")
        stage = "initial_observation_transfer"
        stream_initial(
            worker.stdin,
            goal,
            (encoded([f])[0] for f in frames),
            len(frames),
            limits={key: config[key] for key in ("max_decisions", "max_action_steps")},
            action_spec=adapter.action_spec(task),
        )
        del initial, frames
        stage, status = "policy_execution", "running"
        with (
            (folder / "actions.jsonl").open("w") as action_log,
            (folder / "decisions.jsonl").open("w") as decision_log,
        ):
            import json

            while True:
                try:
                    message = receive(worker.stdout, 2 * 1024**2)
                except (ValueError, BrokenPipeError) as exc:
                    raise PolicyError(
                        "Policy worker closed its channel: " + str(exc)
                    ) from exc
                kind = message.get("kind")
                if kind == "decision":
                    if len(rows) >= config["max_decisions"]:
                        raise PolicyError("Decision limit violated")
                    rows.append(message["row"])
                    decision_log.write(json.dumps(message) + "\n")
                    decision_log.flush()
                    continue
                if kind == "outcome":
                    decision_log.write(json.dumps(message) + "\n")
                    decision_log.flush()
                    continue
                if kind == "done":
                    if status not in TERMINAL:
                        status = message.get("status", "policy_error")
                        if status not in ("decision_limit", "policy_error"):
                            raise PolicyError(
                                "A policy cannot declare simulator success"
                            )
                    error = message.get("error")
                    break
                if kind != "step" or status in TERMINAL:
                    raise PolicyError("Unsupported simulator request")
                action = np.asarray(message.get("action"), dtype=float)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise PolicyError("Invalid action")
                action[6] = (
                    np.clip(action[6], -1, 1)
                    if adapter.action_spec(task)["gripper"]
                    else 0
                )
                calls += 1
                actions.append(action.tolist())
                action_log.write(
                    json.dumps(
                        {"call": calls, "action": action.tolist(), "time": time.time()}
                    )
                    + "\n"
                )
                action_log.flush()
                stage = "simulator_step"
                result = adapter.advance(env, action, context, depth=config["depth"])
                steps, status, frames = (
                    max(steps, result.steps),
                    result.status,
                    result.frames,
                )
                if result.private_error:
                    write(folder / "private_native_error.json", result.private_error)
                if calls % 20 == 0 or status in TERMINAL:
                    write(
                        folder / "progress.json",
                        {
                            "status": status,
                            "steps": steps,
                            "step_calls": calls,
                            "updated": time.time(),
                        },
                    )
                    if progress:
                        progress(steps, calls)
                capture.append(frames)
                stage = "policy_execution"
                try:
                    send(
                        worker.stdin,
                        {
                            "frames": encoded(frames),
                            "status": status,
                            "motion_error": result.motion_error,
                        },
                    )
                except BrokenPipeError as exc:
                    if status in TERMINAL:
                        break
                    raise PolicyError(
                        "Policy exited while waiting for observations"
                    ) from exc
    except PolicyError as exc:
        # A terminal native outcome remains authoritative if the worker closes afterward.
        if status not in TERMINAL:
            status, error = "policy_error", str(exc)
    except BaseException:
        status, error = "harness_error", traceback.format_exc()
        (folder / "private_broker_error.txt").write_text(error)
    finally:
        if timer:
            timer.cancel()
            timer.join()
        if expired.is_set() and status not in TERMINAL:
            status, error = "policy_timeout", "Policy wall-clock limit exceeded"
        if worker:
            terminate(worker)
            worker.stdin.close()
            worker.stdout.close()
        if stderr:
            stderr.close()
        for resource in (env, capture):
            if resource is not None:
                try:
                    resource.close()
                except BaseException:
                    status, error = "harness_error", traceback.format_exc()
                    (folder / "private_cleanup_error.txt").write_text(error)
        result = {
            "schema": "cam-episode/1",
            "benchmark": adapter.name,
            "task": task,
            "episode": episode,
            "split": split,
            "status": status,
            "success": status == "success",
            "scorable": status != "harness_error",
            "steps": steps,
            "step_calls": calls,
            "decisions": len(rows),
            "goal": goal,
            "seconds": time.time() - started,
            "stage": stage,
            "error": "Infrastructure error; details retained privately."
            if status == "harness_error"
            else error,
            "api_calls": 0,
            "api_cost_usd": 0,
            "observation_fields": list(observation_fields(config["depth"])),
            "video_frames": capture.n if capture else 0,
            "program_hash": program_hash(program),
        }
        np.save(folder / "actions.npy", np.asarray(actions, dtype=float).reshape(-1, 7))
        write(folder / "result.json", result)
    return result


def run_episode(root, task, episode, split, maximum=0):
    from cam.harness.prepare import validate_run

    root = Path(root).resolve()
    plan = validate_run(root)
    if task not in plan["tasks"] or split not in ("train", "test"):
        raise ValueError("Unassigned task or split")
    cfg = read(root / "tasks" / task / "state/config.json")
    task_root = root / "tasks" / task
    ledger = Ledger(cfg["ledger"])
    attempt = None
    if split == "train":
        attempt = ledger.reserve(
            cfg, episode
        )  # Before native environment creation, even if it fails.
        folder = task_root / "rollouts/train" / f"R{attempt:03d}"
        program = task_root / "checkpoints" / f"R{attempt - 1:03d}"
        selected = attempt - 1
    else:
        if (
            episode not in plan["test_episodes"][task]
            or not 0 <= maximum <= plan["config"]["budget"]
        ):
            raise ValueError("Unassigned test episode or checkpoint budget")
        program, frozen = select_checkpoint(task_root, maximum)
        selected = frozen["checkpoint"]
        folder = task_root / "rollouts/test" / f"R{maximum:03d}" / f"ep{episode:03d}"
    adapter = get_adapter(
        plan["config"]["benchmark"],
        root / "inputs/benchmark" / plan["benchmark_package"],
    )
    result = execute(
        adapter,
        root / "inputs/metadata" / split,
        task,
        episode,
        split,
        program,
        root / "runtime",
        folder,
        plan["config"],
        progress=(
            lambda steps, calls: ledger.update(attempt, steps=steps, step_calls=calls)
        )
        if attempt
        else None,
    )
    result.update(
        mode=plan["config"]["mode"],
        coding=plan["config"]["coding"],
        maximum_rollouts=maximum if split == "test" else plan["config"]["budget"],
        selected_checkpoint=selected,
        attempt=attempt,
        benchmark_hash=plan["benchmark_hash"],
        harness_hash=plan["harness_hash"],
    )
    records = read(
        root / "inputs/metadata" / split / f"record_dataset_{task}_metadata.json"
    )["records"]
    identity = next(row for row in records if int(row["episode"]) == episode)
    # Privileged seed metadata is stored separately and never copied into training feedback.
    write(folder / "private_metadata.json", identity)
    result["difficulty"] = identity["difficulty"]
    write(folder / "result.json", result)
    if attempt:
        ledger.update(
            attempt,
            steps=result["steps"],
            step_calls=result["step_calls"],
            status=result["status"],
            error=result["error"],
        )
    return result
