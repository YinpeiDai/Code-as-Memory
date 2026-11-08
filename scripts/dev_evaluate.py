"""Host-side development evaluation of a frozen CaM policy on held-out TRAINING seeds.

Runs the run's frozen ``policy/`` through the run's own pinned harness, benchmark
snapshot, runtime, interpreters and limits on training identities that are neither
assigned demonstrations nor the run's simulator training subset. Test seeds are
never used. Results go to ``outputs/dev_eval/...``; the run directory, its ledger
and its evidence are not modified. Use this to diagnose failures before deciding
on a guide change; report held-out test results only via ``cam evaluate``.

  .venv/bin/python scripts/dev_evaluate.py RUN_LEAF --episodes 10 --gpus 0 --workers 4
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

WORKER = r"""
import json, sys
plan = json.loads(open(sys.argv[1]).read())
sys.path.insert(0, plan["harness"])
sys.dont_write_bytecode = True
from cam.benchmarks.registry import get_adapter
from cam.harness.broker import execute
adapter = get_adapter(plan["benchmark"], plan["benchmark_source"])
result = execute(adapter, plan["metadata"], plan["task"], plan["episode"], "train",
                 plan["program"], plan["runtime"], plan["folder"], plan["config"],
                 policy_python=plan["policy_python"])
print(json.dumps({k: result[k] for k in ("status", "steps", "step_calls", "decisions", "seconds")}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--workers", type=int, default=4, help="workers per GPU")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / "run.json").read_text())
    (task,) = plan["tasks"]
    sys.path.insert(0, str(run / "inputs/harness"))
    from cam.benchmarks.registry import get_adapter
    from cam.core.budget import program_hash
    from cam.io import read, write

    policy = run / "tasks" / task / "policy"
    frozen = read(policy / "freeze.json")
    if program_hash(policy) != frozen["program_hash"]:
        raise SystemExit("Frozen policy hash mismatch")
    source = run / "inputs/benchmark" / plan["benchmark_package"]
    adapter = get_adapter(plan["config"]["benchmark"], source)
    used = {int(r["episode"]) for r in read(run / "inputs/dataset.json")["tasks"][task]["episodes"]}
    used |= set(read(run / "tasks" / task / "state/config.json")["train_episodes"])
    test = {int(r["seed"]) for r in adapter.records(task, "test")}
    candidates = sorted(
        (r for r in adapter.records(task, "train") if int(r["episode"]) not in used and int(r["seed"]) not in test),
        key=lambda r: int(r["episode"]),
    )
    if not candidates:
        raise SystemExit("No held-out training identities available")
    count = min(args.episodes, len(candidates))
    picks = [candidates[round(i * (len(candidates) - 1) / max(1, count - 1))] for i in range(count)]
    picks = list({int(r["episode"]): r for r in picks}.values())
    out = (args.output or Path("outputs/dev_eval") / plan["config"]["benchmark"] / task / run.name).resolve()
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    write(out / "metadata" / f"record_dataset_{task}_metadata.json", {"env_id": task, "records": picks})
    write(out / "plan.json", {
        "run": str(run), "task": task, "program_hash": frozen["program_hash"],
        "episodes": [int(r["episode"]) for r in picks],
        "rule": "held-out training identities: not assigned demos, not simulator training seeds, not test seeds",
    })
    python = plan.get("simulator_python", sys.executable)
    slots = [g for g in args.gpus for _ in range(args.workers)]

    def one(job):
        index, record = job
        episode = int(record["episode"])
        folder = out / f"ep{episode:05d}"
        if (folder / "result.json").exists():
            return read(folder / "result.json")
        spec = out / f".plan_{episode}.json"
        write(spec, {
            "harness": str(run / "inputs/harness"), "benchmark": plan["config"]["benchmark"],
            "benchmark_source": str(source), "metadata": str(out / "metadata"), "task": task,
            "episode": episode, "program": str(policy), "runtime": str(run / "runtime"),
            "folder": str(folder), "config": plan["config"], "policy_python": plan.get("policy_python"),
        })
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slots[index % len(slots)]),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": str(run / "inputs/harness")}
        with (out / f"ep{episode:05d}.log").open("w") as log:
            subprocess.run([python, "-c", WORKER, str(spec)], stdout=log, stderr=subprocess.STDOUT,
                           env=env, timeout=plan["config"]["policy_timeout"] + 900)
        return read(folder / "result.json") if (folder / "result.json").exists() else {
            "episode": episode, "status": "harness_error", "success": False, "scorable": False}

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        results = list(pool.map(one, enumerate(picks)))
    scored = [r for r in results if r.get("scorable")]
    summary = {
        "task": task, "run": str(run), "episodes": len(results), "scored": len(scored),
        "successes": sum(r["success"] for r in scored),
        "statuses": {s: sum(r["status"] == s for r in results) for s in sorted({r["status"] for r in results})},
        "per_episode": [{k: r.get(k) for k in ("episode", "status", "steps", "decisions", "error")} for r in results],
    }
    write(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("task", "episodes", "scored", "successes", "statuses")}))


if __name__ == "__main__":
    main()
