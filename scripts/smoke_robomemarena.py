"""Native RoboMemArena smoke checks through the real broker. Not a policy evaluation.

Run with the benchmark interpreter and the CaM sources on PYTHONPATH:

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src envs/robomemarena/.venv/bin/python -B \\
      scripts/smoke_robomemarena.py --output outputs/robomemarena/smoke

Scripted (non-learned) programs exercise: observation-only stepping and
decision_limit; first-frame and trajectory determinism for one seed; a full
2500-step timeout for wall-clock calibration; training seed 100 of T04 versus the
released demonstration frame 0; and an open-loop replay of a released T10 (and,
if downloaded, T01) demonstration that must end in native ``success``.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import imageio.v2 as imageio
import numpy as np

from cam.interface.robomemarena import adapter as rma
from cam.interface.registry import get_adapter
from cam.config import RunConfig
from cam.core.history import materialize
from cam.harness.broker import execute
from cam.harness.prepare import make_runtime
from cam.io import read, sha256, write

CHECKOUT = rma.checkout()
HOLD = "robot.step(0, 0, 0, 0, 0, 0, -1)"


def program(folder, task, body):
    materialize(
        folder,
        {
            "task.json": json.dumps({"task": task, "runtime": "python", "memory": []}).encode(),
            "policy.md": b"Scripted native smoke check; not a learned policy.",
            "tools/policy.py": (
                "def perceive_decide(obs, goal, memory, last):\n"
                '    """Always run the scripted action tool."""\n'
                '    return {"memory": {}, "act": {"tool": "act_script", "args": {}}}\n\n'
                "def act_script(robot):\n"
                '    """Scripted native smoke motion; not a task solution."""\n' + body
            ).encode(),
        },
    )
    return folder


def hold(steps):
    return f"    for _ in range({steps}):\n        {HOLD}\n    return {{'steps': {steps}}}\n"


def replay(actions):
    rows = json.dumps(np.round(np.clip(actions, -1, 1), 8).tolist())
    return (
        f"    for a in {rows}:\n        robot.step(*a)\n"
        f"    while True:\n        robot.step(0, 0, 0, 0, 0, 0, {float(np.sign(actions[-1, 6]))})\n"
    )


def demo(raw, task, seed, first_only=False):
    """Verified released recording (all subtasks, or only the first for frame 0)."""
    item = read(rma.METADATA / "release.json")["tasks"][task]
    rows = item["seeds"][str(seed)][: 1 if first_only else None]
    paths = [raw / "hf" / item["directory"] / row["name"] for row in rows]
    for path, row in zip(paths, rows):
        if not path.is_file() or path.stat().st_size != row["size"] or sha256(path) != row["sha256"]:
            return None
    out = {"action": [], "agentview": None, "wrist": None}
    for path in paths:
        with h5py.File(path, "r") as h5:
            group = h5["data/demo_0"]
            out["action"].append(group["actions"][:])
            if out["agentview"] is None:
                out["agentview"] = group["obs/agentview_rgb"][0]
                out["wrist"] = group["obs/eye_in_hand_rgb"][0]
    out["action"] = np.concatenate(out["action"])
    return out


def run(adapter, root, name, task, split, episode, body, config, policy):
    depth = config["depth"]
    metadata = root / "metadata" / name / split
    write(metadata / f"record_dataset_{task}_metadata.json", {"records": adapter.records(task, split)})
    started = time.time()
    result = execute(
        adapter, metadata, task, episode, split,
        program(root / "programs" / name, task, body),
        root / "runtime", root / "episodes" / name, config, policy_python=policy,
    )
    folder = root / "episodes" / name
    with h5py.File(folder / "observations.h5", "r") as h5:
        fields = sorted(json.loads(h5.attrs["sensors"]))
        frames = len(h5["step"])
        first = {key: h5[key][0] for key in fields}
    reader = imageio.get_reader(folder / "episode.mp4")
    video = sum(1 for _ in reader)
    reader.close()
    summary = {
        "name": name, "task": task, "split": split, "episode": episode,
        "status": result["status"], "scorable": result["scorable"], "steps": result["steps"],
        "step_calls": result["step_calls"], "frames": frames, "video_frames": video,
        "wall_seconds": round(time.time() - started, 1),
        "seconds_per_step": round(result["seconds"] / max(1, result["step_calls"]), 4),
        "fields": fields,
        "metrics": result.get("native_metrics") or (read(folder / "private_native_metrics.json") if (folder / "private_native_metrics.json").exists() else None),
    }
    assert result["scorable"], (name, result)
    assert frames == result["step_calls"] + 1 == video, summary
    assert fields == sorted(adapter.interface(task, depth)["sensors"]), fields
    print(json.dumps({k: v for k, v in summary.items() if k != "fields"}))
    return summary, first


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, default=CHECKOUT / "data/raw/robomemarena")
    parser.add_argument("--skip-full", action="store_true", help="skip the 2500-step timeout run")
    parser.add_argument("--rgb-only", action="store_true", help="omit the live depth sensors")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    policy = json.loads(
        subprocess.run(
            [str(CHECKOUT / ".venv/bin/python"), "-c",
             "import json; from cam.harness.isolation import policy_python; print(json.dumps(policy_python()))"],
            capture_output=True, text=True, check=True, env={**os.environ, "PYTHONPATH": str(CHECKOUT / "src")},
        ).stdout
    )
    adapter = get_adapter("robomemarena")
    make_runtime(root / "runtime")
    base = dict(benchmark="robomemarena", mode="demos", depth=not args.rgb_only, builtin_demo=False, max_action_steps=rma.MAX_STEPS, training_seed_count=None)
    one = RunConfig(**base, max_decisions=1, policy_timeout=3600).to_dict()
    report = {"purpose": "native adapter smoke check, not experimental results", "api_calls": 0, "runs": []}

    a, first_a = run(adapter, root, "hold_a", "T04_ButterNonEmptyDrawer", "test", 0, hold(200), one, policy)
    b, first_b = run(adapter, root, "hold_b", "T04_ButterNonEmptyDrawer", "test", 0, hold(200), one, policy)
    report["runs"] += [a, b]
    assert a["status"] == b["status"] == "decision_limit"
    with h5py.File(root / "episodes/hold_a/observations.h5") as x, h5py.File(root / "episodes/hold_b/observations.h5") as y:
        report["determinism"] = {
            "first_frame_identical": all(np.array_equal(first_a[k], first_b[k]) for k in first_a),
            "trajectory_identical": all(np.array_equal(x[k][:], y[k][:]) for k in json.loads(x.attrs["sensors"])),
        }
    _, other = run(adapter, root, "hold_seed51", "T04_ButterNonEmptyDrawer", "test", 1, hold(1), one, policy)
    report["determinism"]["different_seed_differs"] = not np.array_equal(other["agentview_rgb"], first_a["agentview_rgb"])
    print(report["determinism"])

    train = {int(r["seed"]): int(r["episode"]) for r in adapter.records("T04_ButterNonEmptyDrawer", "train")}
    s100, first = run(adapter, root, "t04_train_seed100", "T04_ButterNonEmptyDrawer", "train", train[100], hold(1), one, policy)
    report["runs"].append(s100)
    recorded = demo(args.raw_data, "T04_ButterNonEmptyDrawer", 100, first_only=True)
    if recorded is not None:
        diff = {
            key: float(np.abs(first[key + "_rgb"].astype(int) - recorded[key].astype(int)).mean())
            for key in ("agentview", "wrist")
        }
        flipped = float(np.abs(first["agentview_rgb"][::-1].astype(int) - recorded["agentview"].astype(int)).mean())
        imageio.imwrite(root / "t04_seed100_vs_demo.png", np.vstack([
            np.hstack([first["agentview_rgb"], first["wrist_rgb"]]),
            np.hstack([recorded["agentview"], recorded["wrist"]]),
        ]))
        report["t04_seed100_vs_demo_frame0_mae"] = {**diff, "agentview_if_flipped": flipped}
        print(report["t04_seed100_vs_demo_frame0_mae"])
    else:
        report["t04_seed100_vs_demo_frame0_mae"] = "raw demonstration not yet downloaded/verified"

    for task in ("T10_PourWineMugTwice", "T01_CookiesTomatoBasket", "T04_ButterNonEmptyDrawer"):
        records = adapter.records(task, "train")
        recorded = demo(args.raw_data, task, int(records[0]["seed"]))
        if recorded is None:
            report[f"replay_{task}"] = "raw demonstration not yet downloaded/verified"
            continue
        summary, _ = run(adapter, root, f"replay_{task}", task, "train", 0, replay(recorded["action"]), one, policy)
        summary["demo_frames"] = len(recorded["action"])
        report["runs"].append(summary)

    if not args.skip_full:
        full, _ = run(adapter, root, "full_timeout", "T01_CookiesTomatoBasket", "test", 0, hold(rma.MAX_STEPS + 5), one, policy)
        assert full["status"] == "timeout" and full["steps"] == rma.MAX_STEPS, full
        report["runs"].append(full)
    write(root / "smoke_report.json", report)
    print("wrote", root / "smoke_report.json")


if __name__ == "__main__":
    main()
