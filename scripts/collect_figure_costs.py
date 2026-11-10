"""Audit figure API costs from saved sessions and results, without running policies.

python scripts/collect_figure_costs.py --archive public_runs --output /new/path/costs.json
The archive is read-only; the output must be a new file.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from cam.reporting.costs import Usage


def collect(archive):
    def source(path):
        return {"path": str(path.relative_to(archive) if path.is_relative_to(archive) else path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def session(path, expected):
        usage = Usage()
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    usage.feed(json.loads(line))
        summary = usage.summary()
        if not summary["sessions_with_final_usage"] or not math.isclose(
            summary["reported_cost_usd"], expected, abs_tol=1e-9
        ):
            raise ValueError(f"Session cost does not match the report: {path}")
        return {**source(path), "reported_cost_usd": summary["reported_cost_usd"],
                "final_session_count": summary["sessions_with_final_usage"]}

    def evaluation(paths, expected, program_hash=None):
        if len(paths) != expected:
            raise ValueError(f"Expected {expected} evaluation records, found {len(paths)}")
        digest = hashlib.sha256()
        successes = 0
        for path in sorted(paths):
            record = json.loads(path.read_text())
            if (record.get("split") != "test"
                or record["status"] not in ("success", "failure", "fail", "decision_limit", "timeout")
                or not record.get("scorable", True) or record["api_calls"] != 0
                or record.get("api_cost_usd", record.get("cost")) != 0):
                raise ValueError(f"Unscored result or nonzero/missing API cost: {path}")
            if program_hash and record["program_hash"] != program_hash:
                raise ValueError(f"Evaluation uses a different frozen program: {path}")
            successes += bool(record["success"])
            item = source(path)
            digest.update(f'{item["path"]}\t{item["sha256"]}\n'.encode())
        return {"successes": successes, "episodes": len(paths), "api_calls": 0,
                "api_cost_usd": 0, "records_sha256": digest.hexdigest()}

    settings = [
        ("rmbench", "rmbench", "demo_only_50", 10, 100),
        ("mikasa_robo", "mikasa_robo_vla_suite", "demo_interaction_50", 5, 50),
        ("robomme", "robomme", "demo_only_25", 16, 50),
        ("robomemarena", "robomemarena", "demo_interaction_50", 26, 50),
    ]
    benchmarks = {}
    for benchmark, directory, setting, expected_tasks, expected_episodes in settings:
        root = archive / "dev_logs" / directory / setting
        tasks = []
        if benchmark == "robomme":
            for report in sorted(root.glob("*/costs/api_cost.json")):
                task_root = report.parents[1]
                cost = json.loads(report.read_text())
                task = task_root.name
                amount = cost["reported_cost_usd"]
                tasks.append({
                    "task": task, "development_api_cost_usd": amount,
                    "report": source(report),
                    "sessions": [session(task_root / "agents/agent.jsonl", amount)],
                    "evaluation": evaluation(list((archive / "eval" / benchmark / setting
                        / task / "R000").glob("ep*/result.json")), expected_episodes),
                })
        else:
            pattern = ("*/20261110T190926_055654Z/reports/summary.json"
                       if directory == "mikasa_robo_vla_suite" else "*/reports/summary.json")
            for report in sorted(root.glob(pattern)):
                summary = json.loads(report.read_text())
                if len(summary["development"]) != 1:
                    raise ValueError(f"Expected one task per report: {report}")
                development = summary["development"][0]
                usage = development["usage"]
                if development["status"] != "finished" or not usage["cost_complete"]:
                    raise ValueError(f"Incomplete development: {report}")
                run, task = report.parents[1], development["task"]
                logs = run / "tasks" / task / "logs"
                sessions = [session(logs / row["session"] / "events.jsonl", row["reported_cost_usd"])
                            for row in usage["sessions"]]
                if set(logs.glob("session_*/events.jsonl")) != {
                    archive / row["path"] for row in sessions
                } or not math.isclose(sum(s["reported_cost_usd"] for s in sessions),
                                     usage["reported_cost_usd"], abs_tol=1e-9):
                    raise ValueError(f"Session coverage does not match: {report}")
                program_hash = development["frozen"]["program_hash"]
                evaluation_root = run
                if not (run / summary["episodes"][0]["directory"] / "result.json").exists():
                    # Development-only exports retain the original held-out artifact location.
                    match = re.search(r"Original run: `([^`]+)`", (run / "ARCHIVE.md").read_text())
                    if match is None:
                        raise ValueError(f"Missing original evaluation location: {run}")
                    evaluation_root = Path(match.group(1))
                    if (evaluation_root / "reports/summary.json").read_bytes() != report.read_bytes():
                        raise ValueError(f"Original and archived reports differ: {report}")
                row = {"task": task, "development_api_cost_usd": usage["reported_cost_usd"],
                       "report": source(report), "sessions": sessions, "program_hash": program_hash,
                       "evaluation_root": str(evaluation_root.relative_to(archive)
                                              if evaluation_root.is_relative_to(archive) else evaluation_root),
                       "evaluation": evaluation([evaluation_root / e["directory"] / "result.json"
                                                 for e in summary["episodes"]],
                                                expected_episodes, program_hash)}
                if "physics_backend_by_task" in summary:
                    row["physics_backend"] = summary["physics_backend_by_task"][task]
                tasks.append(row)
        if len(tasks) != expected_tasks or len({t["task"] for t in tasks}) != expected_tasks:
            raise ValueError(f"Expected {expected_tasks} unique tasks for {benchmark}")
        benchmarks[benchmark] = {
            "source_root": str(root.relative_to(archive)), "setting": setting,
            "task_count": len(tasks),
            "development_api_cost_usd": math.fsum(t["development_api_cost_usd"] for t in tasks),
            "inference_api_cost_usd": 0,
            "evaluation_successes": sum(t["evaluation"]["successes"] for t in tasks),
            "evaluation_episodes": sum(t["evaluation"]["episodes"] for t in tasks),
            "tasks": tasks,
        }
    return {"schema": "cam-benchmark-api-costs/1", "archive": "public_runs",
            "cost_basis": "Final Claude CLI session estimates; not audited billing. USD. "
                          "Excludes local compute, data preparation, and other experiment settings.",
            "development_scope": "Total across all tasks in each selected development setting: "
                                 "RMBench 10, MIKASA VLA suite 5, RoboMME 16, RoboMemArena 26.",
            "verification": "Every session total matches its report. Every selected evaluation "
                            "result is scored and records zero API calls and zero API cost. "
                            "Result digests hash sorted relative-path, tab, file-SHA256, newline records.",
            "benchmarks": benchmarks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.archive.absolute())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    for name, row in data["benchmarks"].items():
        print(f'{name}: ${row["development_api_cost_usd"]:.2f} development, $0 inference; '
              f'{row["evaluation_successes"]}/{row["evaluation_episodes"]} successful episodes')
