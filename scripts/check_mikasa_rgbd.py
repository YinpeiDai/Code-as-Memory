"""Check live MIKASA v2 RGB-D through the isolated policy worker, on training only.

This infrastructure smoke check makes no model calls or held-out evaluations.
Use the simulator interpreter with PYTHONPATH=src and a fresh output directory.
"""

import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from cam.benchmarks.base import source_hash
from cam.benchmarks.registry import get_adapter
from cam.config import RunConfig
from cam.core.history import materialize
from cam.harness.broker import execute
from cam.harness.prepare import make_runtime, training_schedule
from cam.io import read, sha256, write

ROOT = Path(__file__).resolve().parents[1]
POLICY = '''
def _inspect(frame):
    results = {}
    for camera in ["front", "wrist"]:
        rgb = getattr(frame, camera + "_rgb")
        depth = getattr(frame, camera + "_depth")
        K = getattr(frame, camera + "_K")
        extrinsic = getattr(frame, camera + "_extrinsic")
        assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
        assert depth.shape == (256, 256) and depth.dtype == np.int16
        assert not rgb.flags.writeable and not depth.flags.writeable
        assert np.any(depth > 0) and np.all(depth >= 0)
        assert K.shape == (3, 3) and K[0, 0] > 0 and K[1, 1] > 0
        assert extrinsic.shape == (3, 4) and np.isfinite(extrinsic).all()
        results[camera] = {"rgb": list(rgb.shape), "depth": list(depth.shape),
                           "positive_depth_pixels": int(np.count_nonzero(depth))}
    return {"step": int(frame.step), "cameras": results}

def perceive_decide(obs, goal, memory, last):
    """Check the initial observation delivered to policy code."""
    _inspect(obs.current)
    return {"memory": {}, "act": {"tool": "act_check", "args": {}}}

def act_check(robot):
    """Check reset and two consecutive control-step observations."""
    frames = [_inspect(robot.now())]
    for _ in range(2):
        robot.step(0, 0, 0, 0, 0, 0, 1)
        frames.append(_inspect(robot.now()))
    return {"checked_frames": frames}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/mikasa_robo_v2_50")
    parser.add_argument("--task", default="TakeItBack-VLA-v0")
    args = parser.parse_args()
    adapter = get_adapter("mikasa_robo_v2")
    config = RunConfig.from_dict(read(ROOT / "configs/mikasa_robo_v2/both_v2_claude_code.json"))
    manifest = read(args.dataset / "manifest.json")
    demos = manifest["tasks"][args.task]["episodes"]
    assigned = training_schedule(adapter, args.task, demos, config)
    excluded = {r["seed"] for r in demos + assigned + adapter.records(args.task, "test")}
    identity = next(r for r in adapter.records(args.task, "train") if r["seed"] not in excluded)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "metadata" / f"record_dataset_{args.task}_metadata.json", {"records": [identity]})
    runtime = make_runtime(output / "runtime", demonstrations=False, mikasa_motion_preview=True)
    program = output / "program"
    materialize(program, {
        "task.json": json.dumps({"task": args.task, "runtime": "python", "memory": []}).encode(),
        "policy.md": b"Host-authored sensor infrastructure check; not a learned solution.\n",
        "tools/policy.py": POLICY.encode(),
    })
    code = "import json; from cam.harness.isolation import policy_python; print(json.dumps(policy_python()))"
    python = json.loads(subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-c", code], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout)
    result = execute(adapter, output / "metadata", args.task, identity["episode"], "train",
                     program, runtime, output / "rollout", {**config.to_dict(), "max_decisions": 1},
                     policy_python=python)
    assert result["status"] == "decision_limit" and result["steps"] == 2, result
    assert result["api_calls"] == 0
    decisions = [json.loads(line) for line in (output / "rollout/decisions.jsonl").read_text().splitlines()]
    outcomes = [row for row in decisions if row["kind"] == "outcome"]
    assert len(outcomes) == 1 and "checked_frames" in json.dumps(outcomes), outcomes
    with h5py.File(output / "rollout/observations.h5") as h5:
        assert h5["step"][:].tolist() == [0, 1, 2]
        fields = {name: {"shape": list(h5[name].shape), "dtype": str(h5[name].dtype)} for name in adapter.interface(args.task, True)["sensors"]}
        for camera in ("front", "wrist"):
            assert h5[camera + "_rgb"].shape == (3, 256, 256, 3)
            depth = h5[camera + "_depth"][:]
            assert depth.shape == (3, 256, 256) and depth.dtype == np.int16
            assert np.all(np.any(depth > 0, axis=(1, 2)))
    report = {
        "passed": True, "kind": "training_sensor_infrastructure_check", "task": args.task,
        "training_identity": identity, "outside_assigned_demo_and_interaction_seeds": True,
        "frames_checked": 3, "fields": fields, "isolated_policy_checked_frames": True,
        "held_out_episodes_used": 0, "model_calls": 0,
        "benchmark_source_hash": source_hash(adapter),
        "adapter_sha256": sha256(ROOT / "src/cam/benchmarks/mikasa_robo_v2.py"),
        "script_sha256": sha256(Path(__file__)),
    }
    write(output / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
