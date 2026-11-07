"""Exact episode counts. One evaluation is one point, with no inferred repeats."""

from collections import Counter
import csv
from pathlib import Path
import statistics
import time
import os

from cam.core.budget import Ledger
from cam.core.history import select_checkpoint
from cam.io import read, write
from cam.providers.usage import summarize_sessions


def metrics(rows, expected):
    scored = [r for r in rows if r["scorable"]]
    successful = [r for r in scored if r["success"]]
    complete = len(scored) == expected and len(rows) == expected
    return {
        "expected": expected,
        "recorded": len(rows),
        "scored": len(scored),
        "successes": len(successful),
        "complete": complete,
        "infrastructure_errors": sum(not r["scorable"] for r in rows),
        "success_rate": len(successful) / expected if complete and expected else None,
        "observed_success_rate": len(successful) / len(scored) if scored else None,
        "mean_success_steps": statistics.mean(r["steps"] for r in successful)
        if successful
        else None,
        "mean_decisions": statistics.mean(r["decisions"] for r in scored)
        if scored
        else None,
        "statuses": dict(Counter(r["status"] for r in rows)),
    }


def matched_efficiency(groups):
    """Paired comparison on exactly the same successful task/episode identities."""
    result = []
    keys = sorted(groups)
    for i, left in enumerate(keys):
        a = {
            (r["task"], r["episode"]): r
            for r in groups[left]
            if r["scorable"] and r["success"]
        }
        for right in keys[i + 1 :]:
            b = {
                (r["task"], r["episode"]): r
                for r in groups[right]
                if r["scorable"] and r["success"]
            }
            shared = a.keys() & b.keys()
            result.append(
                {
                    "left": left,
                    "right": right,
                    "matched_successes": len(shared),
                    "left_mean_steps": statistics.mean(a[k]["steps"] for k in shared)
                    if shared
                    else None,
                    "right_mean_steps": statistics.mean(b[k]["steps"] for k in shared)
                    if shared
                    else None,
                }
            )
    return result


def build(root):
    root = Path(root).resolve()
    plan = read(root / "run.json")
    if plan.get("schema") != "cam-run/1":
        raise ValueError(
            "Only new CaM runs are accepted; historical paths are references"
        )
    results, development = [], []
    for task in plan["tasks"]:
        folder = root / "tasks" / task
        ledger = Ledger(folder / "state/ledger.sqlite")
        attempts = ledger.rows()
        usage = summarize_sessions(folder / "logs", plan["config"]["coding"]["agent"])
        completion = (
            read(folder / "logs/completion.json")
            if (folder / "logs/completion.json").exists()
            else {}
        )
        final = (
            read(folder / "policy/freeze.json")
            if (folder / "policy/freeze.json").exists()
            else None
        )
        state = completion.get("status", "pending")
        sessions = sorted((folder / "logs").glob("session_*"))
        if sessions and not (sessions[-1] / "completion.json").exists():
            process_file = sessions[-1] / "process.json"
            state = "starting"
            if process_file.exists():
                try:
                    os.kill(int(read(process_file)["pid"]), 0)
                    state = "running"
                except ProcessLookupError:
                    state = "interrupted"
            queue_file = root / "queues/learning/status.json"
            if queue_file.exists():
                queued = read(queue_file).get("jobs", {}).get(task, {})
                if queued.get("status") == "failed":
                    state = "failed"
        development.append(
            {
                "task": task,
                "status": "finished" if final else state,
                "demonstrations": plan["demonstrations_per_task"][task],
                "attempts": attempts,
                "usage": usage,
                "frozen": final,
                "train_episodes_visited": len({r["episode"] for r in attempts}),
                "train_episodes_allowed": len(ledger.config()["train_episodes"]),
                "first_success_attempt": next(
                    (r["id"] for r in attempts if r["status"] == "success"), None
                ),
            }
        )
        for path in sorted((folder / "rollouts/test").glob("R*/ep*/result.json")):
            row = read(path)
            maximum = row["maximum_rollouts"]
            _, checkpoint = select_checkpoint(folder, maximum)
            if (
                row["task"] != task
                or row["episode"] not in plan["test_episodes"][task]
                or row["program_hash"] != checkpoint["program_hash"]
                or row["benchmark_hash"] != plan["benchmark_hash"]
                or row["api_calls"] != 0
            ):
                raise ValueError(f"Invalid or changed evaluation provenance: {path}")
            results.append(
                {**row, "directory": path.parent.relative_to(root).as_posix()}
            )
    ids = [(r["task"], r["maximum_rollouts"], r["episode"]) for r in results]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate held-out evaluation identity")
    budgets = sorted({r["maximum_rollouts"] for r in results})
    config_path = root / "evaluation_plan.json"
    if config_path.exists():
        budgets = sorted(set(budgets) | set(read(config_path)["budgets"]))
    if not budgets:
        budgets = [0] if plan["config"]["mode"] == "demos" else [1, 5, 10, 20]
        budgets = [n for n in budgets if n <= plan["config"]["budget"]]
    expected = sum(len(plan["test_episodes"][task]) for task in plan["tasks"])
    groups = {
        f"R{n:03d}": [r for r in results if r["maximum_rollouts"] == n] for n in budgets
    }
    settings = []
    for n in budgets:
        rows = groups[f"R{n:03d}"]
        by_task = {
            task: metrics(
                [r for r in rows if r["task"] == task], len(plan["test_episodes"][task])
            )
            for task in plan["tasks"]
        }
        difficulty = {}
        for level in ("easy", "medium", "hard"):
            count = sum(
                sum(
                    r["difficulty"] == level
                    for r in read(
                        root
                        / "inputs/metadata/test"
                        / f"record_dataset_{task}_metadata.json"
                    )["records"]
                )
                for task in plan["tasks"]
            )
            difficulty[level] = metrics(
                [r for r in rows if r["difficulty"] == level], count
            )
        settings.append(
            {
                "budget": n,
                **metrics(rows, expected),
                "tasks": by_task,
                "difficulty": difficulty,
            }
        )
    known = [
        t["usage"]["reported_cost_usd"]
        for t in development
        if t["usage"]["reported_cost_usd"] is not None
    ]
    summary = {
        "schema": "cam-report/1",
        "updated": time.time(),
        # Organized runs share a timetag across tasks/settings. Keep the full
        # experiment identity in display and paired-comparison keys.
        "name": "/".join(root.parts[-5:])
        if root.parent.name in ("demo_only", "interaction_only", "demo_interaction")
        else root.name,
        "benchmark": plan["config"]["benchmark"],
        "benchmark_hash": plan["benchmark_hash"],
        "config": plan["config"],
        "tasks": plan["tasks"],
        "expected_episodes_per_setting": expected,
        "settings": settings,
        "matched_efficiency": matched_efficiency(groups),
        "development": development,
        "episodes": results,
        "cost": {
            "reported_cost_usd": sum(known) if known else None,
            "complete": bool(development)
            and all(t["usage"]["cost_complete"] for t in development),
            "tasks_with_reported_cost": len(known),
            "task_count": len(development),
            "evaluation_api_cost_usd": 0,
            "gpu_cost_included": False,
        },
        "statistics_rule": "One exact rate per complete evaluation. No confidence bands or invented repeats. "
        "Partial observed rates exclude infrastructure errors and are explicitly labeled partial. "
        "Efficiency uses successful episodes; paired comparisons use common successful identities.",
    }
    write(root / "reports/summary.json", summary)
    columns = [
        "task",
        "episode",
        "difficulty",
        "maximum_rollouts",
        "selected_checkpoint",
        "status",
        "success",
        "scorable",
        "steps",
        "step_calls",
        "decisions",
        "seconds",
        "program_hash",
        "api_calls",
        "api_cost_usd",
    ]
    with (root / "reports/episodes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return summary
