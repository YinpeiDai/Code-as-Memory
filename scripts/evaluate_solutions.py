"""Evaluate every policy in a published setting with one shared GPU worker limit.

Example: python scripts/evaluate_solutions.py solutions/robomme/demo_only_25 \
    --output validate_runs/robomme_full --gpus 0 1 --workers-per-gpu 5
Tasks run in sequence; each task distributes its episodes across all GPU slots.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from cam.harness.jobs import gpu_slots, locked
from cam.harness.solutions import prepare_solutions
from cam.io import read, write
from cam.reporting.progress import write_progress


def evaluate(args):
    gpu_slots(args.gpus, args.workers_per_gpu)
    if args.resume:
        batch = args.resume.resolve()
        plan = read(batch / "plan.json")
        if Path(plan["solution"]) != args.solution.resolve():
            raise ValueError("Resume the same published solution setting")
    else:
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = args.output.resolve()
        batch = output / "batches" / tag
        prepared = prepare_solutions(output / "runs", args.solution, args.tasks, timetag=tag)
        rows = prepared["runs"]
        benchmark = args.solution.parent.name
        plan = {"solution": str(args.solution.resolve()), "runs": rows,
                "benchmarks": {benchmark: {"tasks": [r["task"] for r in rows]}}, "created": time.time()}
        write(batch / "plan.json", plan)
    with locked(batch / ".scheduler.lock"):
        state = read(batch / "status.json") if (batch / "status.json").exists() else {
            "schema": "cam-published-evaluation/1", "tasks": {r["task"]: {"status": "pending", "run": r["root"]} for r in plan["runs"]}}
        state.update(status="running", gpus=args.gpus, workers_per_gpu=args.workers_per_gpu)
        def save():
            state["updated"] = time.time()
            write(batch / "status.json", state)
            write_progress(batch, "Published solution evaluation", state)
        save()
        for task, row in state["tasks"].items():
            if row["status"] == "finished":
                continue
            run = Path(row["run"])
            argv = [sys.executable, str(run / "runner.py"), "evaluate", str(run), "--budgets", "0",
                    "--gpus", *map(str, args.gpus), "--workers-per-gpu", str(args.workers_per_gpu)]
            if (run / "queues/evaluation/status.json").exists():
                argv.append("--resume")
            row.update(status="running", command=argv)
            save()
            log = batch / f"{task}_{time.time_ns()}.log"
            with log.open("x") as stream:
                result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
            row.update(status="finished" if result.returncode == 0 else "needs_attention", log=str(log))
            if result.returncode:
                state["status"] = "needs_attention"
                save()
                return {"batch": str(batch), "status": state["status"]}
            save()
        state["status"] = "finished"
        save()
    return {"batch": str(batch), "status": state["status"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--output", type=Path, default=Path("validate_runs"))
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--workers-per-gpu", type=int, default=5)
    parser.add_argument("--resume", type=Path, help="Resume a previously prepared batch")
    result = evaluate(parser.parse_args())
    print(__import__("json").dumps(result, indent=2))
    raise SystemExit(int(result["status"] != "finished"))
