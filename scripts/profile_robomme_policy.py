"""Profile complete frozen RoboMME policy computation on recorded episodes.

Includes perceive_decide and act_* helpers. Robot I/O is replaced by recorded
observations, and every emitted command is checked against the recorded trace.
This is offline timing, not a new evaluation or a real-time controller test.
H5 loading, simulator/robot time, IPC, imports and replay checks are excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import h5py
import numpy as np

from cam.core.code_policy import decide
from cam.core.tools import ACT, outcome, to_json
from cam.core.workspace import Workspace
from cam.interface.robomme.adapter import RoboMME
from cam.runtime.observations import Observations
from cam.runtime.robot import EpisodeEnd, Robot
from profile_robomme_decisions import equivalent, frame_at

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class ReplayMismatch(BaseException):
    """Never swallowed as a recoverable policy/tool error."""


class RecordedRobot(Robot):
    def __init__(self, current, following, actions, start, terminal, spec):
        super().__init__(current, None, None, action_spec=spec, max_steps=10000)
        self.following, self.actions = following, actions
        self.offset, self.terminal = start, terminal
        self.bursts_ns = []
        self.ready_ns = 0

    def step(self, *values, **named):
        action = self._command(values, named)
        stop = time.perf_counter_ns()
        self.bursts_ns.append(stop - self.ready_ns)
        i = self.offset + self._count
        if i >= len(self.actions) or not np.allclose(action, self.actions[i], rtol=1e-9, atol=1e-8):
            expected = self.actions[i] if i < len(self.actions) else "end of trace"
            raise ReplayMismatch(f"Command {i}: {action} != {expected}")
        self._produced.append(self.following[i])
        self._count += 1
        self.ready_ns = time.perf_counter_ns()
        if i == len(self.actions) - 1:
            raise EpisodeEnd(self.terminal)
        return self.now()


def run_episode(ws, toolkit, result, frames, commands, spec):
    first_exec = next(i for i, frame in enumerate(frames) if frame.phase == "exec")
    initial, following = frames[:first_exec + 1], frames[first_exec + 1:]
    if len(following) != len(commands):
        raise ReplayMismatch("Recording does not have exactly one response frame per command")
    current = initial
    memory = {field.name: None for field in ws.fields}
    last, offset = None, 0
    decision_times, action_times, bursts = [], [], []
    for index, expected in enumerate(result["rows"]):
        obs = Observations(tuple(current))
        start = time.perf_counter_ns()
        decision = decide(ws, toolkit, result["goal"], obs, memory, last)
        decision_ns = time.perf_counter_ns() - start
        if not equivalent(to_json(decision.act, digits=12), to_json(expected["act"], digits=12)) or not equivalent(decision.memory_written, expected["memory_written"]):
            raise ReplayMismatch(f"Decision {index} differs from its recorded action or memory")
        memory = decision.memory_written
        robot = RecordedRobot(obs.current, following, commands, offset, result["status"], spec)
        robot.ready_ns = time.perf_counter_ns()
        ended = False
        try:
            returned = outcome(toolkit.call, decision.act["tool"], robot, decision.act["args"], ACT)
        except EpisodeEnd as exc:
            ended = True
            if exc.status != result["status"]:
                raise ReplayMismatch("Different terminal status")
        tail_ns = time.perf_counter_ns() - robot.ready_ns
        action_ns = sum(robot.bursts_ns) + tail_ns
        decision_times.append(decision_ns / 1e6)
        action_times.append(action_ns / 1e6)
        # A segment starts when an observation is available and ends when the
        # next command is ready. Add the preceding decision to its first segment.
        segments = list(robot.bursts_ns)
        if segments:
            segments[0] += decision_ns
        bursts.extend(ns / 1e6 for ns in segments)
        offset += robot._count
        if ended:
            if index != len(result["rows"]) - 1 or offset != len(commands):
                raise ReplayMismatch("Episode terminated before the end of its trace")
            break
        if returned != expected.get("outcome"):
            raise ReplayMismatch(f"Action {index} outcome differs: {returned}")
        last = f"{decision.act['tool']} -> {returned}"
        current = robot.frames() or [robot.now()]
    if offset != len(commands) or not ended:
        raise ReplayMismatch("Trace did not terminate at the recorded final command")
    return {"decision_ms": decision_times, "action_helper_ms": action_times,
            "command_compute_ms": bursts,
            "total_compute_ms": sum(decision_times) + sum(action_times),
            "commands_verified": offset, "decisions_verified": len(decision_times)}


def main(args):
    output, archive = args.output.resolve(), args.archive.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks = args.tasks or sorted(p.name for p in (ROOT / "solutions/robomme/demo_only_25").iterdir() if p.is_dir())
    metadata = {"schema": "cam-offline-full-policy-profile/1", "created": time.time(),
                "scope": __doc__, "episodes_per_task": args.episodes, "repeats": args.repeats,
                "warmups": 1, "tasks": tasks, "selection": "First N test episodes numerically, without outcome filtering.",
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "python": platform.python_version(), "numpy": np.__version__,
                "script_sha256": digest(Path(__file__)),
                "hardware": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
                "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "runtime_hashes": {str(p.relative_to(ROOT)): digest(p) for base in ("src/cam/core", "src/cam/runtime") for p in (ROOT / base).glob("*.py")}}
    write(output / "study.json", metadata)
    shutil.copyfile(__file__, output / Path(__file__).name)
    shutil.copyfile(Path(__file__).with_name("profile_robomme_decisions.py"), output / "profile_robomme_decisions.py")
    samples = []
    for task in tasks:
        frozen = archive / "dev_logs/robomme/demo_only_25" / task / "frozen_policy"
        hashes = {}
        for path in (frozen / "tools").glob("*.py"):
            assert digest(path) == digest(ROOT / "solutions/robomme/demo_only_25" / task / path.name)
            hashes[path.name] = digest(path)
        ws = Workspace.load(frozen)
        toolkit = ws.toolkit()
        spec = RoboMME().action_spec(task)
        paths = sorted((archive / "eval/robomme/demo_only_25" / task / "R000").glob("ep*/result.json"), key=lambda p: int(p.parent.name[2:]))[:args.episodes]
        assert len(paths) == args.episodes
        for path in paths:
            result = json.loads(path.read_text())
            commands = [r["action"] for r in map(json.loads, (path.parent / "actions.jsonl").read_text().splitlines())]
            with h5py.File(path.parent / "observations.h5") as h5:
                frames = tuple(frame_at(h5, i) for i in range(len(h5["step"])))
            trials = []
            for repetition in range(args.repeats + 1):
                trial = run_episode(ws, toolkit, result, frames, commands, spec)
                if repetition:
                    trials.append(trial)
            item = {"task": task, "episode": result["episode"], "recorded_status": result["status"],
                    "native_control_steps": result["steps"], "recorded_wall_seconds": result["seconds"],
                    "source_result": str(path.relative_to(archive)), "source_result_sha256": digest(path),
                    "source_actions_sha256": digest(path.parent / "actions.jsonl"),
                    "source_observations_sha256": digest(path.parent / "observations.h5"),
                    "policy_hashes": hashes, "trials": trials,
                    "median_total_compute_ms": float(np.median([t["total_compute_ms"] for t in trials])),
                    "median_decision_compute_ms": float(np.median([sum(t["decision_ms"]) for t in trials])),
                    "median_action_helper_compute_ms": float(np.median([sum(t["action_helper_ms"]) for t in trials]))}
            samples.append(item)
            write(output / "samples.json", samples)
            (output / "PROGRESS.md").write_text(f"# RoboMME full policy timing\n\n{len(samples)}/{len(tasks) * args.episodes} recorded episodes verified.\n")
            print(f"{task} ep{result['episode']}: {item['median_total_compute_ms']:.1f} ms policy / {result['steps']} steps; decisions {item['median_decision_compute_ms']:.1f}, helpers {item['median_action_helper_compute_ms']:.1f} ms", flush=True)
            del frames
    bursts = [float(np.median(values)) for item in samples for values in zip(*(t["command_compute_ms"] for t in item["trials"]))]
    total = sum(r["median_total_compute_ms"] for r in samples)
    steps = sum(r["native_control_steps"] for r in samples)
    summary = {"schema": metadata["schema"], "episodes": len(samples), "tasks": len(tasks),
               "native_control_steps": steps, "verified_commands": len(bursts),
               "total_policy_compute_ms": total, "mean_compute_ms_per_control_step": total / steps,
               "mean_policy_compute_ms_per_episode": total / len(samples),
               "decision_compute_ms": sum(r["median_decision_compute_ms"] for r in samples),
               "action_helper_compute_ms": sum(r["median_action_helper_compute_ms"] for r in samples),
               "command_latency_median_ms": float(np.median(bursts)),
               "command_latency_p95_ms": float(np.percentile(bursts, 95)),
               "command_latency_max_ms": max(bursts), "all_commands_and_decisions_match": True,
               "scope": __doc__, "completed": time.time()}
    write(output / "results.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "public_runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    if args.episodes < 1 or args.repeats < 1:
        parser.error("episodes and repeats must be positive")
    main(args)
