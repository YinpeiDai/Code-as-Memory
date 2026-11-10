"""Time frozen RoboMME perception/memory decisions on recorded observations.

This is an offline timing study, not a new simulator evaluation. It never learns,
changes a policy, executes act tools, or produces new success scores. Timing covers
code_policy.decide, including its validation and memory copies; it excludes H5
loading, process startup, act-tool execution, rendering, simulation and robot I/O.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import h5py
import numpy as np

from cam.core.workspace import Workspace
from cam.core.code_policy import decide
from cam.core.tools import to_json
from cam.runtime.observations import Frame, Observations

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def frame_at(h5, index):
    # Same Frame and read-only arrays as the runtime; decoding is outside timing.
    frame = Frame.__new__(Frame)
    frame.phase = "exec" if int(h5["phase"][index]) else "demo"
    frame.step = int(h5["step"][index])
    frame.sensors = tuple(key for key in h5 if key not in ("phase", "step"))
    for key in frame.sensors:
        value = np.asarray(h5[key][index])
        value.setflags(write=False)
        setattr(frame, key, value)
    return frame


def equivalent(actual, expected):
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(equivalent(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(equivalent(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-10))
    return actual == expected


def profile(archive, output, episodes, repeats):
    output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads((ROOT / "configs/solutions/robomme/demo_only_25.json").read_text())
    tasks = sorted(path.name for path in (ROOT / "solutions/robomme/demo_only_25").iterdir() if path.is_dir())
    run = {"schema": "cam-offline-decision-profile/1", "created": time.time(),
           "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "script_sha256": sha(Path(__file__)), "python": platform.python_version(),
           "numpy": np.__version__, "platform": platform.platform(),
           "hardware": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
           "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
           "episodes_per_task": episodes, "repeats": repeats, "warmups": 1,
           "selection": "First N recorded test episodes in numeric order, without filtering by outcome.",
           "timing_scope": __doc__.strip(), "tasks": tasks,
           "runtime_hashes": {str(p.relative_to(ROOT)): sha(p) for base in ["src/cam/core", "src/cam/runtime"] for p in (ROOT / base).glob("*.py")}}
    write(output / "study.json", run)
    samples, task_stats, costs = [], [], []
    for task in tasks:
        frozen = archive / "dev_logs/robomme/demo_only_25" / task / "frozen_policy"
        for source in (frozen / "tools").glob("*.py"):
            assert sha(source) == sha(ROOT / "solutions/robomme/demo_only_25" / task / source.name), task
        ws = Workspace.load(frozen)
        toolkit = ws.toolkit()
        cost_path = archive / "dev_logs/robomme/demo_only_25" / task / "costs/api_cost.json"
        cost = json.loads(cost_path.read_text())
        assert cost["sessions_with_final_usage"] >= 1 and cost["query_usage_matches_session_totals"]
        costs.append({"task": task, "api_cost_usd": cost["reported_cost_usd"],
                      "source": str(cost_path.relative_to(archive)), "sha256": sha(cost_path)})
        paths = sorted((archive / "eval/robomme/demo_only_25" / task / "R000").glob("ep*/result.json"),
                       key=lambda p: int(p.parent.name[2:]))[:episodes]
        assert len(paths) == episodes
        task_samples = []
        for result_path in paths:
            result = json.loads(result_path.read_text())
            last = None
            with h5py.File(result_path.parent / "observations.h5") as h5:
                labels = [("E" if phase else "D") + str(step) for phase, step in zip(h5["phase"][:], h5["step"][:])]
                assert len(set(labels)) == len(labels), (task, "ambiguous recorded frame labels")
                lookup = {label: i for i, label in enumerate(labels)}
                for number, row in enumerate(result["rows"]):
                    first, end = [lookup[label] for label in row["frames"]]
                    obs = Observations(tuple(frame_at(h5, i) for i in range(first, end + 1)))
                    durations = []
                    for repeat in range(repeats + 1):
                        memory = copy.deepcopy(row["memory_in"])
                        start = time.perf_counter_ns()
                        decision = decide(ws, toolkit, result["goal"], obs, memory, last)
                        ms = (time.perf_counter_ns() - start) / 1e6
                        if not equivalent(to_json(decision.act, digits=12), to_json(row["act"], digits=12)) or not equivalent(decision.memory_written, row["memory_written"]):
                            raise ValueError(f"Recorded decision differs: {task}, episode {result['episode']}, decision {number}")
                        if repeat:
                            durations.append(ms)
                    item = {"task": task, "episode": result["episode"], "decision": number,
                            "frames": row["frames"], "frame_count": len(obs.frames),
                            "milliseconds": durations, "median_ms": float(np.median(durations)),
                            "matches_recorded_decision": True, "source": str(result_path.relative_to(archive)),
                            "source_sha256": sha(result_path)}
                    samples.append(item)
                    task_samples.append(item["median_ms"])
                    last = f"{row['act']['tool']} -> {row['outcome']}" if "outcome" in row else None
        task_stats.append({"task": task, "episodes": len(paths), "decisions": len(task_samples),
                           "median_ms": float(np.median(task_samples)), "p95_ms": float(np.percentile(task_samples, 95)),
                           "mean_ms": float(np.mean(task_samples))})
        write(output / "samples.json", samples)
        write(output / "tasks.json", task_stats)
        (output / "PROGRESS.md").write_text(f"# RoboMME offline decision timing\n\n{len(task_stats)}/{len(tasks)} tasks complete; {len(samples)} recorded decisions reproduced.\n")
        print(f"{task}: {len(task_samples)} decisions, median {np.median(task_samples):.3f} ms", flush=True)
    values = [row["median_ms"] for row in samples]
    summary = {"schema": run["schema"], "tasks": len(tasks), "episodes": episodes * len(tasks),
               "decisions": len(samples), "repeats": repeats,
               "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
               "mean_ms": float(np.mean(values)), "development_api_cost_usd": sum(row["api_cost_usd"] for row in costs),
               "evaluation_api_cost_usd": 0, "cost_basis": "Final Claude CLI session estimates; excludes local compute and data preparation; not a billing audit.",
               "timing_scope": "Offline perception + memory decision; excludes act tools, I/O and simulator execution.",
               "costs": costs, "completed": time.time()}
    write(output / "results.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "costs"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "public_runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.episodes < 1 or args.repeats < 1:
        parser.error("episodes and repeats must be positive")
    profile(args.archive.resolve(), args.output.resolve(), args.episodes, args.repeats)
