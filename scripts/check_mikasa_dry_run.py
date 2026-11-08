"""Host-only API regression: replay recorded TRAINING actions with motion queries.

Uses the current CaM adapter and the run's pinned benchmark. Never changes a run,
starts a learner, uses test episodes, or copies diagnostic evidence to feedback.
Run with the MIKASA simulator interpreter and PYTHONPATH=src.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from cam.benchmarks.mikasa_robo import MikasaRobo, frame
from cam.benchmarks.mikasa_robo_api.preview import preview


def snapshot(native):
    """Host-only state comparison; private values are not written to the report."""
    import torch

    values = {}

    def collect(prefix, obj):
        if isinstance(obj, torch.Tensor):
            values[prefix] = obj.detach().cpu().numpy().copy()
        elif isinstance(obj, np.ndarray):
            values[prefix] = obj.copy()
        elif isinstance(obj, (int, float, bool)):
            values[prefix] = np.asarray(obj)
        elif isinstance(obj, dict):
            for name, child in obj.items():
                collect(f"{prefix}/{name}", child)
        elif isinstance(obj, (list, tuple)):
            for i, child in enumerate(obj):
                collect(f"{prefix}/{i}", child)

    collect("state", native.get_state_dict())
    collect("native", vars(native))
    for name, ctrl in native.agent.controller.controllers.items():
        collect("controller/" + name, vars(ctrl))
        pose = getattr(ctrl, "_target_pose", None)
        if pose is not None:
            collect("controller/" + name + "/target_pose", pose.raw_pose)
    px = native.scene.px
    for name in dir(px):
        if name.startswith("cuda_"):
            value = getattr(px, name)
            if hasattr(value, "torch"):
                collect("gpu/" + name, value.torch())
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run / "run.json").read_text())
    if plan["config"]["benchmark"] != "mikasa_robo" or len(plan["tasks"]) != 1:
        parser.error("Provide a single-task original MIKASA-Robo run")
    task = plan["tasks"][0]
    recorded = args.run / "tasks" / task / f"rollouts/train/R{args.attempt:03}"
    metadata = json.loads((recorded / "private_metadata.json").read_text())
    actions = np.load(recorded / "actions.npy")
    args.output.mkdir(parents=True, exist_ok=False)
    adapter = MikasaRobo(args.run / "inputs/benchmark/mikasa_robo")
    env = adapter.make_environment(args.run / "inputs/metadata/train", task, metadata["episode"], "train", False)
    reports, checked = [], set()
    error = None
    try:
        obs, _ = env.reset()
        native = env.env.unwrapped
        arm = native.agent.controller.controllers["arm"]
        with h5py.File(recorded / "observations.h5") as h5:
            for step in range(len(actions) + 1):
                current = frame(obs, step)
                for field in ("base_rgb", "wrist_rgb", "tcp", "gripper_opening", "step"):
                    if not np.array_equal(current[field], h5[field][step]):
                        raise AssertionError(f"Recorded {field} changed at frame {step}")
                if step == len(actions):
                    break
                before = snapshot(native)
                command = actions[step, :6].tolist()
                report = preview(native, command, "delta")
                assert report == preview(native, command, "delta"), "Repeated query changed result"
                # World absolute input and its returned normalized command must agree.
                target = current["tcp"].astype(float).copy()
                target[0] += .02
                target[5] += .05
                absolute = preview(native, target, "absolute")
                equivalent = preview(native, absolute["command"], "delta")
                assert absolute["joint_target_rad"] == equivalent["joint_target_rad"]
                after = snapshot(native)
                assert before.keys() == after.keys(), "Query added simulator state"
                changed = [k for k in before if not np.array_equal(before[k], after[k], equal_nan=True)]
                assert not changed, f"Query changed state: {changed}"
                checked.update(before)
                obs, terminated, truncated, _ = env.step(actions[step])
                actual = arm._target_qpos.detach().cpu().numpy()[0]
                assert np.array_equal(actual, report["joint_target_rad"]), f"Target mismatch at step {step + 1}"
                reports.append({"action_step": step + 1, **report})
                if (step + 1) % 20 == 0:
                    print(f"Verified {step + 1}/{len(actions)} recorded actions", flush=True)
                if (terminated or truncated) and step + 1 != len(actions):
                    raise AssertionError("Replay terminated earlier than the original")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.close()
        (args.output / "previews.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in reports))
        result = {
            "passed": error is None, "error": error,
            "source_training_rollout": str(recorded.resolve()),
            "actions_sha256": hashlib.sha256((recorded / "actions.npy").read_bytes()).hexdigest(),
            "verified_actions": len(reports), "expected_actions": len(actions),
            "queries_per_action": 4,
            "state_fields_checked": sorted(checked),
            "unsafe_action_steps": [r["action_step"] for r in reports if not r["within_joint_limits"]],
            "exact_controller_targets": error is None,
            "exact_recorded_observations": error is None,
            "unchanged_state_across_queries": error is None,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "state_fields_checked"}), flush=True)


if __name__ == "__main__":
    main()
