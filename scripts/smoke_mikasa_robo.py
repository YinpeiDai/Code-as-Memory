"""Native MIKASA-Robo smoke test through the real CaM broker (no model calls).

Runs in the MIKASA-Robo interpreter on one GPU, one fresh process per episode,
like ``cam _episode``. A scripted policy runs in the real Bubblewrap policy
worker against a pinned copy of the benchmark (with the linked asset tree):

* every case must end with a native terminal status and leave a consistent
  ``result.json``, ``observations.h5``, ``episode.mp4`` and ``actions.npy``;
* determinism: the same seed gives an identical first frame in fresh processes
  and after resetting another seed in the same environment first.

    CUDA_VISIBLE_DEVICES=0 envs/mikasa_robo/.venv/bin/python -I \
        scripts/smoke_mikasa_robo.py --output outputs/mikasa_robo/smoke
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TASKS = ("ShellGameTouch-VLA-v0", "RememberColor9-VLA-v0")
LONG = "BatteriesCheckerHard-6-VLA-v0"  # longest registered horizon (2160)
TERMINAL = ("success", "fail", "timeout")
SENSORS = ("base_rgb", "wrist_rgb", "tcp", "gripper_opening")
POLICY = '''def perceive_decide(obs, goal, memory, last):
    """Scripted smoke policy: alternate short descents and ascents."""
    count = (memory["count"] or 0) + 1
    phase = count % 4
    dz = -0.4 if phase == 1 else 0.4 if phase == 3 else 0.0
    gripper = -1.0 if phase == 2 else 1.0
    return {"memory": {"count": count}, "act": {"tool": "act_move", "args": {"dz": dz, "gripper": gripper, "steps": STEPS}}}

def act_move(robot, dz, gripper, steps):
    """Move vertically for a number of native steps, then report the TCP height."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, dz, 0.0, 0.0, 0.0, gripper)
    return {"z": float(robot.now().tcp[2])}
'''
REPLAY = '''ACTIONS = ACTION_LIST

def perceive_decide(obs, goal, memory, last):
    """Open-loop replay of one recorded demonstration (diagnostic only)."""
    count = (memory["count"] or 0) + 1
    return {"memory": {"count": count}, "act": {"tool": "act_replay", "args": {"start": (count - 1) * STEPS, "count": STEPS}}}

def act_replay(robot, start, count):
    """Send recorded commands start..start+count, then hold."""
    for i in range(int(start), int(start) + int(count)):
        command = ACTIONS[i] if i < len(ACTIONS) else [0.0] * 6 + [ACTIONS[-1][6]]
        robot.step(*command)
    return {"sent": int(count)}
'''


def frame_hash(frame):
    digest = hashlib.sha256()
    for name in SENSORS:
        digest.update(frame[name].tobytes())
    return digest.hexdigest()


def policy_python():
    code = "import json; from cam.harness.isolation import policy_python; print(json.dumps(policy_python()))"
    out = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def prepare(out, tasks, steps_per_action, replay=None):
    from cam.benchmarks.base import pin_source
    from cam.benchmarks.registry import get_adapter
    from cam.core.history import materialize
    from cam.harness.prepare import make_runtime
    from cam.io import write

    adapter = get_adapter("mikasa_robo")
    source = out / "benchmark" / adapter.package
    assets = pin_source(adapter, source)
    make_runtime(out / "runtime", demonstrations=False)
    for task in tasks:
        for split in ("train", "test"):
            rows = adapter.records(task, split)[:2]
            write(out / "metadata" / split / f"record_dataset_{task}_metadata.json", {"env_id": task, "records": rows})
        files = {
            "task.json": json.dumps({"task": task, "runtime": "python", "memory": [{"name": "count", "type": "count", "description": "decisions"}]}).encode(),
            "policy.md": b"Scripted smoke policy.\n",
            "tools/policy.py": POLICY.replace("STEPS", str(steps_per_action)).encode(),
        }
        materialize(out / "programs" / task, files)
    if replay:
        import h5py

        task, episode = replay["task"], replay["episode"]
        rows = [r for r in adapter.records(task, "train") if int(r["episode"]) == episode]
        write(out / "metadata" / "train" / f"record_dataset_{task}_metadata.json", {"env_id": task, "records": rows})
        with h5py.File(replay["h5"], "r") as h5:
            actions = h5[f"episode_{episode}"]["action"][:].astype(float).tolist()  # exact float32 values
        program = REPLAY.replace("ACTION_LIST", json.dumps(actions)).replace("STEPS", str(steps_per_action))
        materialize(
            out / "programs" / (task + "_replay"),
            {
                "task.json": json.dumps({"task": task, "runtime": "python", "memory": [{"name": "count", "type": "count", "description": "decisions"}]}).encode(),
                "policy.md": b"Open-loop demonstration replay (diagnostic).\n",
                "tools/policy.py": program.encode(),
            },
        )
    return {"source": str(source), "assets": assets}


def run_case(case):
    """One episode in this (fresh) process via cam.harness.broker.execute."""
    from cam.benchmarks.registry import get_adapter
    from cam.harness.broker import execute
    from cam.io import read

    out, task, split, episode = Path(case["out"]), case["task"], case["split"], case["episode"]
    adapter = get_adapter("mikasa_robo", case["source"])
    if case.get("reset_probe"):
        # Fresh env: reset another seed first, then this episode's seed.
        other = 1 - episode
        env = adapter.make_environment(out / "metadata" / split, task, other, split, False)
        try:
            first = adapter.begin(env, builtin_demo=False, depth=False).frames[0]
            env.seed = int(next(r["seed"] for r in adapter.records(task, split) if int(r["episode"]) == episode))
            again = adapter.begin(env, builtin_demo=False, depth=False).frames[0]
        finally:
            env.close()
        print(json.dumps({"other_hash": frame_hash(first), "reset_after_other_hash": frame_hash(again)}))
        return
    if case.get("backend"):
        import cam.benchmarks.mikasa_robo as mikasa

        mikasa.MAKE_KWARGS["sim_backend"] = case["backend"]  # diagnostic override only
    config = read(ROOT / "configs/mikasa_robo/demos_claude_code.json")
    folder = out / "episodes" / case["name"]
    result = execute(
        adapter, out / "metadata" / split, task, episode, split, out / "programs" / case.get("program", task),
        out / "runtime", folder, config, policy_python=case["policy_python"],
    )
    print(json.dumps({"result": result}))


def check(folder, result, sensors):
    import h5py
    import imageio.v2 as imageio
    import numpy as np

    problems = []
    if result["status"] not in TERMINAL or not result["scorable"]:
        problems.append(f"non-terminal status {result['status']}")
    with h5py.File(folder / "observations.h5", "r") as h5:
        n = len(h5["step"])
        if set(h5) != set(sensors) | {"step", "phase"}:
            problems.append(f"observation fields {sorted(h5)}")
        if n != result["steps"] + 1 or not np.array_equal(h5["step"][:], np.arange(n)) or (h5["phase"][:] != 1).any():
            problems.append("frame count/step/phase mismatch")
        first = {name: h5[name][0] for name in sensors}
    actions = np.load(folder / "actions.npy")
    if actions.shape != (result["step_calls"], 7) or result["step_calls"] != result["steps"]:
        problems.append(f"actions {actions.shape} vs {result['step_calls']} calls, {result['steps']} steps")
    frames, shape = 0, None
    with imageio.get_reader(folder / "episode.mp4") as reader:
        for image in reader:
            frames, shape = frames + 1, shape or image.shape
    if frames != result["video_frames"] or frames != n or shape[:2] != (128, 256):
        problems.append(f"video frames {frames} shape {shape} vs {n} observations")
    return problems, frame_hash(first)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/mikasa_robo/smoke")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--long-task", default=LONG, help="'' skips the long-horizon timing case")
    parser.add_argument("--steps-per-action", type=int, default=60)
    parser.add_argument("--replay", help="TASK:EPISODE:EXTRACTED_H5 open-loop demo replay (diagnostic)")
    parser.add_argument("--replay-backends", nargs="+", default=["gpu"], choices=["gpu", "cpu"])
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        return run_case(json.loads(args.case))
    out = (args.output / time.strftime("%Y%m%dT%H%M%S")).resolve()
    tasks = list(args.tasks) + ([args.long_task] if args.long_task else [])
    replay = None
    if args.replay:
        task, episode, h5 = args.replay.split(":", 2)
        replay = {"task": task, "episode": int(episode), "h5": str(Path(h5).resolve())}
    setup = prepare(out, tasks, args.steps_per_action, replay)
    pp = policy_python()
    cases = []
    for task in args.tasks:
        cases += [
            {"name": f"{task}_test0_a", "task": task, "split": "test", "episode": 0},
            {"name": f"{task}_test0_b", "task": task, "split": "test", "episode": 0},
            {"name": f"{task}_test0_reset", "task": task, "split": "test", "episode": 0, "reset_probe": True},
            {"name": f"{task}_train0", "task": task, "split": "train", "episode": 0},
        ]
    if args.long_task:
        cases.append({"name": f"{args.long_task}_test0", "task": args.long_task, "split": "test", "episode": 0})
    for backend in args.replay_backends if replay else ():
        cases.append({
            "name": f"{replay['task']}_train{replay['episode']}_replay_{backend}", "task": replay["task"],
            "split": "train", "episode": replay["episode"], "program": replay["task"] + "_replay",
            "backend": None if backend == "gpu" else backend,
        })
    from cam.benchmarks.registry import get_adapter

    adapter = get_adapter("mikasa_robo", setup["source"])
    summary, failures = {"output": str(out), "assets": setup["assets"], "cases": {}}, 0
    for case in cases:
        case.update(out=str(out), source=setup["source"], policy_python=pp)
        started = time.time()
        proc = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--case", json.dumps(case)],
            capture_output=True, text=True,
        )
        wall = time.time() - started
        (out / "episodes").mkdir(exist_ok=True)
        (out / "episodes" / f"{case['name']}.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            summary["cases"][case["name"]] = {"error": proc.stderr[-2000:]}
            failures += 1
            continue
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        if case.get("reset_probe"):
            summary["cases"][case["name"]] = {**report, "process_seconds": round(wall, 2)}
            continue
        result = report.pop("result")
        sensors = list(adapter.interface(case["task"], False)["sensors"])
        problems, first = check(out / "episodes" / case["name"], result, sensors)
        summary["cases"][case["name"]] = {
            "status": result["status"],
            "steps": result["steps"],
            "decisions": result["decisions"],
            "episode_seconds": round(result["seconds"], 2),
            "process_seconds": round(wall, 2),
            "seconds_per_step": round(result["seconds"] / max(1, result["steps"]), 4),
            "first_frame": first,
            **report,
            "problems": problems,
        }
        failures += bool(problems)
    for task in args.tasks:
        a, b, r = (summary["cases"].get(f"{task}_test0_{x}", {}) for x in ("a", "b", "reset"))
        same = a.get("first_frame") is not None and a.get("first_frame") == b.get("first_frame")
        after = r.get("reset_after_other_hash") == a.get("first_frame")
        differs = r.get("other_hash") not in (None, a.get("first_frame"))
        summary.setdefault("determinism", {})[task] = {
            "fresh_processes_identical": same,
            "identical_after_other_seed_reset": after,
            "other_seed_differs": differs,
        }
        failures += not (same and after and differs)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
