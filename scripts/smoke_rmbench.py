"""Native RMBench smoke check through the real broker. Not a learned-policy evaluation.

Runs in the benchmark interpreter (like ``cam _episode``) with a scripted,
observation-free motion policy in the isolated CaM policy worker, then checks the
recorded artifacts, statuses and per-action wall time, and that a seed's first
frame is reproducible across fresh processes. With ``--run`` it uses a prepared
run's pinned benchmark copy, pinned harness, pinned metadata and recorded policy
interpreter (no learning, no ledger attempt, no test result is written).

  CUDA_VISIBLE_DEVICES=1 envs/rmbench/.venv/bin/python scripts/smoke_rmbench.py \
      --source outputs/rmbench/pinned/rmbench --output outputs/rmbench/smoke/<tag>
  CUDA_VISIBLE_DEVICES=1 envs/rmbench/.venv/bin/python scripts/smoke_rmbench.py \
      --run runs/rmbench/<task>/.../<timetag> --split train --output outputs/rmbench/smoke/<tag>
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

POLICY = '''def perceive_decide(obs, goal, memory, last):
    """Scripted smoke motion; ignores the scene."""
    return {"memory": {}, "act": {"tool": "act_batch", "args": {"count": BATCH}}}


def act_batch(robot, count):
    """Cycle hold / both arms up / both arms back / close / open / unreachable left target."""
    errors = 0
    for i in range(count):
        frame = robot.now()
        left, right = list(frame.left_ee_pose), list(frame.right_ee_pose)
        phase = i % 8
        if phase == 1:
            left[2] += AMPLITUDE
            right[2] += AMPLITUDE
            left[1] += AMPLITUDE
            right[1] += AMPLITUDE
        elif phase == 2:
            left[2] -= AMPLITUDE
            right[2] -= AMPLITUDE
            left[1] -= AMPLITUDE
            right[1] -= AMPLITUDE
        elif phase == 5:
            left[0] += 3.0
        grip = 0.0 if phase in (3, 4) else 1.0
        try:
            robot.step(*left, grip, *right, grip)
        except Exception:
            errors += 1
    return {"actions": count, "motion_errors": errors}
'''

FIRST_FRAME = """
import sys, hashlib, json, os
sys.path.insert(0, {harness!r})
from pathlib import Path
from cam.interface.rmbench.adapter import RMBench
a = RMBench({source!r})
env = a.make_environment(Path({metadata!r}), {task!r}, {episode}, {split!r}, False)
frame = a.begin(env, builtin_demo=False, depth=False).frames[0]
out = {{k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in frame.items() if hasattr(v, 'tobytes')}}
env.close()
print('DIGEST ' + json.dumps(out), flush=True)
os._exit(0)
"""


def cam_policy_python():
    """The CaM 3.11 sandbox interpreter, as recorded by cam prepare."""
    out = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-c",
            "import json; from cam.harness.isolation import policy_python; print(json.dumps(policy_python()))",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd="/tmp",
    )
    return json.loads(out.stdout)


def first_frame(harness, source, metadata, task, episode, split):
    """Fresh-process first observation digest for the determinism check."""
    code = FIRST_FRAME.format(
        harness=str(harness), source=str(source), metadata=str(metadata),
        task=task, episode=episode, split=split,
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd="/tmp", timeout=900
    )
    line = [x for x in result.stdout.splitlines() if x.startswith("DIGEST ")]
    if not line:
        raise RuntimeError(result.stdout[-2000:] + result.stderr[-4000:])
    return json.loads(line[-1][7:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--source", type=Path, help="pinned RMBench copy with assets/")
    where.add_argument("--run", type=Path, help="prepared CaM run leaf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", help="defaults to the prepared run's task or observe_and_pickup")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--episode", type=int, help="defaults to the first pinned episode")
    parser.add_argument("--batch", type=int, default=25, help="actions per decision")
    parser.add_argument("--amplitude", type=float, default=0.04, help="metres per scripted move")
    parser.add_argument("--no-determinism", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.run:
        run = args.run.resolve()
        plan = json.loads((run / "run.json").read_text())
        harness = run / "inputs/harness"
        source = run / "inputs/benchmark" / plan["benchmark_package"]
        task = args.task or plan["tasks"][0]
        records = json.loads(
            (run / "inputs/metadata" / args.split / f"record_dataset_{task}_metadata.json").read_text()
        )["records"]
        policy_python = plan["policy_python"]
    else:
        run, plan = None, None
        harness, source = ROOT / "src", args.source.resolve()
        task = args.task or "observe_and_pickup"
        records = None
        policy_python = cam_policy_python()
    sys.path.insert(0, str(harness))
    import h5py
    import imageio.v2 as imageio
    import numpy as np

    from cam.interface.rmbench.adapter import RMBench
    from cam.config import RunConfig
    from cam.core.history import materialize
    from cam.harness.broker import execute
    from cam.harness.prepare import make_runtime
    from cam.io import write

    import cam

    if not Path(cam.__file__).resolve().is_relative_to(harness.resolve()):
        raise SystemExit(f"Imported {cam.__file__}, not the harness under {harness}")
    adapter = RMBench(source)
    records = records or adapter.records(task, args.split)
    record = next(
        r for r in records if args.episode is None or int(r["episode"]) == args.episode
    )
    episode = int(record["episode"])
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write(root / "metadata" / f"record_dataset_{task}_metadata.json", {"records": [record]})
    materialize(
        root / "program",
        {
            "task.json": json.dumps({"task": task, "runtime": "python", "memory": []}).encode(),
            "policy.md": b"Scripted RMBench smoke motion; not a task policy.\n",
            "tools/policy.py": POLICY.replace("BATCH", str(args.batch))
            .replace("AMPLITUDE", repr(args.amplitude))
            .encode(),
        },
    )
    make_runtime(root / "runtime")
    limit = adapter.limits[task]
    config = RunConfig(
        benchmark="rmbench",
        mode="demos",
        depth=False,
        builtin_demo=False,
        max_decisions=limit // args.batch + 2,
        max_action_steps=args.batch,
        policy_timeout=7200,
    ).to_dict()
    started = time.time()
    result = execute(
        adapter, root / "metadata", task, episode, args.split, root / "program",
        root / "runtime", root / "rollout", config, policy_python=policy_python,
    )
    folder = root / "rollout"
    checks = {"status": result["status"], "steps": result["steps"], "step_calls": result["step_calls"]}
    interface = adapter.interface(task, False)
    with h5py.File(folder / "observations.h5", "r") as h5:
        checks["h5_frames"] = int(len(h5["step"]))
        checks["h5_fields_match"] = sorted(k for k in h5 if k not in ("step", "phase")) == sorted(
            interface["sensors"]
        )
        checks["h5_last_step"] = int(h5["step"][-1])
    reader = imageio.get_reader(folder / "episode.mp4")
    checks["mp4_frames"] = sum(1 for _ in reader)
    checks["mp4_shape"] = list(reader.get_data(0).shape)
    reader.close()
    checks["actions_shape"] = list(np.load(folder / "actions.npy").shape)
    times = [json.loads(x)["time"] for x in (folder / "actions.jsonl").read_text().splitlines()]
    gaps = np.diff(times) if len(times) > 1 else np.array([0.0])
    checks["seconds_per_action"] = {
        "mean": float(gaps.mean()), "median": float(np.median(gaps)), "p95": float(np.percentile(gaps, 95)),
        "max": float(gaps.max()),
    }
    checks["seconds_to_first_action"] = times[0] - started if times else None
    outcomes = [json.loads(x) for x in (folder / "decisions.jsonl").read_text().splitlines()]
    checks["motion_errors_reported"] = sum(
        json.loads(o["output"]).get("motion_errors", 0)
        for o in outcomes
        if o["kind"] == "outcome" and o["output"].startswith("{")
    )
    finished = result["status"] == "timeout" and result["steps"] == limit == result["step_calls"]
    succeeded = result["status"] == "success" and result["steps"] == result["step_calls"]
    frames = result["step_calls"] + 1
    checks["passed"] = (
        (finished or succeeded)
        and checks["h5_frames"] == frames == checks["mp4_frames"]
        and checks["h5_fields_match"]
        and checks["actions_shape"] == [result["step_calls"], 16]
        and checks["motion_errors_reported"] >= 1
    )
    if not args.no_determinism:
        digests = [
            first_frame(harness, source, root / "metadata", task, episode, args.split) for _ in range(2)
        ]
        checks["first_frame_identical"] = digests[0] == digests[1]
        checks["first_frame_differences"] = sorted(k for k in digests[0] if digests[0][k] != digests[1].get(k))
        checks["passed"] = checks["passed"] and checks["first_frame_identical"]
    checks["wall_seconds"] = time.time() - started
    checks["episode_seconds"] = result["seconds"]
    write(
        root / "smoke.json",
        {
            "purpose": "native adapter smoke check, not experimental results",
            "run": str(run) if run else None,
            "source": str(source),
            "harness": str(harness),
            "record": record,
            "amplitude": args.amplitude,
            "result": result,
            "checks": checks,
        },
    )
    print(json.dumps(checks, indent=2), flush=True)
    sys.stdout.flush()
    os._exit(0 if checks["passed"] else 1)  # SAPIEN/CuRobo teardown can hang at exit


if __name__ == "__main__":
    main()
