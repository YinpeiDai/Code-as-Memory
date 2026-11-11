"""Complete episode checks for every published policy, with no large recordings."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import queue
import threading
import time
import traceback
import subprocess

from cam.harness.jobs import gpu_slots, locked, subprocess_job
from cam.harness.solutions import load_solutions, prepare_solutions
from cam.interface.registry import get_adapter, simulator_python
from cam.io import read, write


def executable_result(result):
    """Task failure is runnable; policy and infrastructure errors fail the check."""
    return bool(result.get("scorable") and result.get("step_calls", 0) > 0
                and result.get("status") in ("success", "fail", "timeout", "decision_limit"))


def free_mib(gpu):
    probe = subprocess.run([
        "nvidia-smi", "--id=" + str(gpu), "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=15, check=True)
    return int(probe.stdout.strip())


def check_solutions(solutions, output, benchmarks=None, tasks=None, gpus=(0, 1),
                    workers_per_gpu=10, min_free_gib=8, resume=None, episodes=5,
                    retry_infrastructure=False):
    slots = gpu_slots(gpus, workers_per_gpu)
    if min_free_gib < 0 or episodes < 1:
        raise ValueError("Use a nonnegative memory threshold and positive episode count")
    solutions, output = Path(solutions).resolve(), Path(output).resolve()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    batch = Path(resume).resolve() if resume else output / tag
    if retry_infrastructure and not resume:
        raise ValueError("Infrastructure recovery requires --resume BATCH")
    if resume:
        state = read(batch / "status.json")
        if state.get("schema") != "cam-solution-check/2" or state["episodes_per_policy"] != episodes:
            raise ValueError("Resume the same check schema and episode count")
    else:
        rows, checked = {}, set()
        for folder in sorted(solutions.glob("*/*")):
            if not folder.is_dir() or (benchmarks and folder.parent.name not in benchmarks):
                continue
            spec, names, _ = load_solutions(folder)
            selected = [t for t in names if tasks is None or t in tasks]
            if not selected:
                continue
            adapter = get_adapter(spec["benchmark"])
            if spec["benchmark"] not in checked:
                simulator_python(adapter)
                for asset in getattr(adapter, "asset_links", {}).values():
                    if not Path(asset).is_dir():
                        raise FileNotFoundError(f"{adapter.name} simulator assets missing: {asset}")
                checked.add(spec["benchmark"])
            catalog = batch / "catalogs" / folder.parent.name / (folder.name + ".json")
            for task in selected:
                ids = [int(r["episode"]) for r in adapter.records(task, "test")][:episodes]
                if len(ids) != episodes:
                    raise ValueError(f"{task} has fewer than {episodes} assigned tests")
                key = "/".join((folder.parent.name, folder.name, task))
                rows[key] = {"benchmark": spec["benchmark"], "suite": spec.get("suite"), "setting": folder.name,
                             "task": task, "solution": str(folder), "catalog": str(catalog),
                             "status": "pending", "episodes": ids, "results": []}
            write(catalog, spec)
        if not rows or (tasks and set(tasks) - {r["task"] for r in rows.values()}):
            raise ValueError("No solutions found for one or more requested tasks")
        state = {"schema": "cam-solution-check/2", "purpose": "published_policy_execution_check",
                 "started": time.time(), "status": "running", "tasks": rows,
                 "expected_policies": len(rows), "expected": len(rows) * episodes,
                 "episodes_per_policy": episodes, "recording": "results",
                 "gpus": list(gpus), "workers_per_gpu": workers_per_gpu,
                 "model_calls": 0, "published_scores_unchanged": True}
    # Recovery happens before admission under the same scheduler lock as execution.
    # Existing outcomes are retained in a receipt; scored failures never enter this path.
    if retry_infrastructure:
        from cam.harness.recover import recover_test
        with locked(batch / ".scheduler.lock"):
            write(batch / f"status_before_recovery_{time.time_ns()}.json", state)
            for row in state["tasks"].values():
                if not row.get("run"):
                    continue
                recorded = {r["episode"]: r for r in row["results"]}
                retried = set()
                for episode in row["episodes"]:
                    folder = Path(row["run"]) / "tasks" / row["task"] / "rollouts/test/R000" / f"ep{episode:03d}"
                    result = recorded.get(episode, {})
                    on_disk = read(folder / "result.json") if (folder / "result.json").exists() else None
                    retry = (on_disk.get("scorable") is False if on_disk is not None else
                             folder.exists() or result.get("scorable") is False or result.get("status") == "worker_error")
                    if retry:
                        recovery = recover_test(row["run"], row["task"], episode, 0,
                            "Release sanity check: retry unscorable simulator infrastructure failure with identical inputs")
                        row.setdefault("infrastructure_retries", []).append({"episode": episode, "result": result, **recovery})
                        retried.add(episode)
                if retried:
                    row.update(results=[r for r in row["results"] if r["episode"] not in retried], status="pending")
                    row.pop("preparation_error", None)
            state.update(status="running", gpus=list(gpus), workers_per_gpu=workers_per_gpu)
            state.setdefault("recovery_launches", []).append({"time": time.time(), "gpus": list(gpus),
                "workers_per_gpu": workers_per_gpu, "min_free_gib": min_free_gib})
            write(batch / "status.json", state)
    pending = queue.Queue()
    for key, row in state["tasks"].items():
        if row["status"] not in ("passed", "failed"):
            pending.put(key)
    guard = threading.Lock()
    admission = {gpu: threading.Lock() for gpu in gpus}
    last_launch = {gpu: 0.0 for gpu in gpus}
    preparation = threading.Lock()

    def save():
        rows = list(state["tasks"].values())
        results = [r for row in rows for r in row["results"]]
        state.update(updated=time.time(), completed=len(results),
                     passed=sum(r["runnable"] for r in results),
                     failed=sum(not r["runnable"] for r in results),
                     policies_completed=sum(r["status"] in ("passed", "failed") for r in rows),
                     policies_passed=sum(r["status"] == "passed" for r in rows),
                     preparation_errors=sum(bool(r.get("preparation_error")) for r in rows))
        state["infrastructure_retries"] = sum(len(r.get("infrastructure_retries", [])) for r in rows)
        write(batch / "status.json", state)
        write(batch / "results.json", state)
        write(output / "latest.json", {"batch": str(batch), "status": state["status"]})
        write(output / "results.json", {"batch": str(batch), **state})
        lines = ["# Published solution execution checks", "",
                 f"{episodes} complete held-out episodes per frozen policy; logs and results only. This is not a full benchmark score.", "",
                 f"Episodes: {state['completed']}/{state['expected']}; runnable: {state['passed']}; execution errors: {state['failed']}.", "",
                 "| Benchmark | Setting | Task | Check | Runnable | Task successes |", "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r.get('suite') or r['benchmark']} | {r['setting']} | {r['task']} | {r['status']} | "
                         f"{sum(e['runnable'] for e in r['results'])}/{episodes} | {sum(e.get('success', False) for e in r['results'])}/{episodes} |")
        temporary = batch / "PROGRESS.md.partial"
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(batch / "PROGRESS.md")

    def worker(gpu):
        while not (batch / "STOP.json").exists():
            try:
                key = pending.get_nowait()
            except queue.Empty:
                return
            row = state["tasks"][key]
            try:
                with guard:
                    row.update(status="preparing", gpu=gpu)
                    save()
                with preparation:
                    if not row.get("run"):
                        prepared = prepare_solutions(batch / "runs", row["solution"], [row["task"]],
                                                     catalog=row["catalog"], recording="results")
                        row["run"] = prepared["runs"][0]["root"]
                    root = Path(row["run"])
                    plan = read(root / "run.json")
                    write(root / "sanity_check.json", {"purpose": state["purpose"], "expected": episodes,
                          "task": row["task"], "episodes": row["episodes"], "batch": str(batch),
                          "full_benchmark_expected": len(plan["test_episodes"][row["task"]])})
                existing = {r["episode"] for r in row["results"]}
                for episode in row["episodes"]:
                    if episode in existing:
                        continue
                    if (batch / "STOP.json").exists():
                        with guard:
                            row["status"] = "interrupted"
                            save()
                        return
                    result_path = root / "tasks" / row["task"] / "rollouts/test/R000" / f"ep{episode:03d}/result.json"
                    if not result_path.exists():
                        with guard:
                            row.update(status="waiting_for_gpu", current_episode=episode)
                            save()
                        with admission[gpu]:
                            time.sleep(max(0, 5 - (time.monotonic() - last_launch[gpu])))
                            while free_mib(gpu) < min_free_gib * 1024:
                                if (batch / "STOP.json").exists():
                                    return
                                time.sleep(5)
                            last_launch[gpu] = time.monotonic()
                        with guard:
                            row.update(status="running", started=time.time())
                            save()
                        log = root / f"episode_{episode:03d}_{time.time_ns()}.log"
                        code = subprocess_job(root, ["_episode", str(root), row["task"], str(episode),
                                              "--split", "test", "--budget", "0"], log, gpu,
                                              plan["config"]["policy_timeout"] + 600,
                                              python=plan.get("simulator_python"))
                        row["exit_code"] = code
                    if result_path.is_file():
                        result = read(result_path)
                        item = {k: result.get(k) for k in ("episode", "status", "success", "scorable", "seconds", "step_calls", "error")}
                        item.update(runnable=executable_result(result), result=str(result_path))
                    else:
                        item = {"episode": episode, "status": "worker_error", "success": False,
                                "scorable": False, "result": str(result_path), "runnable": False,
                                "error": "No result produced; inspect episode process log"}
                    with guard:
                        row["results"].append(item)
                        save()
                with guard:
                    row.update(status="passed" if all(r["runnable"] for r in row["results"]) else "failed",
                               finished=time.time())
                    save()
            except Exception:
                with guard:
                    row.update(status="failed", preparation_error=traceback.format_exc(), finished=time.time())
                    save()
            finally:
                pending.task_done()

    with locked(batch / ".scheduler.lock"):
        save()
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            list(pool.map(worker, slots))
        with guard:
            if (batch / "STOP.json").exists():
                state["status"] = "stopped"
            else:
                state["status"] = "complete" if state["passed"] == state["expected"] else "needs_attention"
            state["finished"] = time.time()
            save()
    return {"batch": str(batch), **{k: state[k] for k in ("expected", "completed", "passed", "failed", "status")}}
