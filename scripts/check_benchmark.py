"""Check a pinned benchmark with a blank policy on one training episode.

Run with CaM's Python 3.11, e.g.:
  .venv/bin/python scripts/check_benchmark.py --benchmark mikasa_robo \
      --task ShellGameTouch-VLA-v0 --output outputs/check_mikasa --gpu 0

No coding client, learned policy, or held-out evaluation is involved. The trusted
broker uses the benchmark environment; the policy uses CaM's isolated runtime.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(path):
    sys.dont_write_bytecode = True
    plan = json.loads(path.read_text())
    root = path.parent
    sys.path.insert(0, str(root / "harness"))
    from cam.interface.registry import get_adapter
    from cam.harness.broker import execute
    from cam.io import write

    adapter = get_adapter(plan["benchmark"], root / "benchmark")
    result = execute(
        adapter, root / "metadata", plan["task"], plan["episode"], "train",
        root / "program", root / "runtime", root / "rollout", plan["config"],
        policy_python=plan["policy_python"],
    )
    import h5py
    import imageio.v2 as imageio
    import numpy as np

    interface = adapter.interface(plan["task"], plan["config"]["depth"])
    with h5py.File(root / "rollout/observations.h5") as h5:
        frames = len(h5["step"])
        assert set(h5) == set(interface["sensors"]) | {"step", "phase"}
        for name, spec in interface["sensors"].items():
            shape = h5[name].shape[1:]
            assert len(shape) == len(spec["shape"]) and all(
                declared is None or actual == declared
                for actual, declared in zip(shape, spec["shape"])
            ), name
    with imageio.get_reader(root / "rollout/episode.mp4") as movie:
        video_frames = sum(1 for _ in movie)
    actions = np.load(root / "rollout/actions.npy")
    passed = (
        result["status"] in ("decision_limit", "success", "fail", "timeout")
        and result["step_calls"] > 0
        and frames == video_frames == result["video_frames"]
        and actions.shape == (result["step_calls"], len(interface["action"]["names"]))
        and result["api_calls"] == 0
    )
    write(root / "check.json", {
        "purpose": "training-only infrastructure diagnostic; not learned-policy performance",
        "passed": passed, "benchmark": plan["benchmark"], "task": plan["task"],
        "status": result["status"], "steps": result["steps"], "step_calls": result["step_calls"],
        "frames": frames, "video_frames": video_frames,
        "seconds": result["seconds"], "api_calls": result["api_calls"],
    })
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark")
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.benchmark or not args.task or not args.output or args.steps < 1:
        parser.error("provide --benchmark, --task, --output and positive --steps")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import cam
    from cam.interface.base import pin_source
    from cam.interface.registry import get_adapter, simulator_python
    from cam.config import RunConfig
    from cam.core.history import materialize
    from cam.harness.isolation import policy_python
    from cam.harness.prepare import make_runtime, starter, simulator_snapshot
    from cam.io import copy_tree, tree_hash, write

    adapter = get_adapter(args.benchmark)
    record = adapter.records(args.task, "train")[0]
    depth = args.benchmark == "robomme"
    interface = adapter.interface(args.task, depth)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    assets = pin_source(adapter, root / "benchmark")
    copy_tree(Path(cam.__file__).parent, root / "harness/cam")
    make_runtime(root / "runtime", demonstrations=False)
    materialize(root / "program", {
        "task.json": json.dumps({"task": args.task, "runtime": "python", "memory": []}).encode(),
        "policy.md": b"Blank-policy infrastructure check; not a learned solution.\n",
        "tools/policy.py": starter(interface).encode(),
    })
    write(root / "metadata" / f"record_dataset_{args.task}_metadata.json", {"records": [record]})
    python = simulator_python(adapter) or sys.executable
    plan = {
        "benchmark": args.benchmark, "task": args.task, "episode": int(record["episode"]),
        "config": RunConfig(
            benchmark=args.benchmark, mode="demos", depth=depth, builtin_demo=False,
            max_decisions=args.steps, max_action_steps=1, policy_timeout=120,
        ).to_dict(),
        "policy_python": policy_python(), "simulator_environment": simulator_snapshot(python),
        "benchmark_hash": tree_hash(root / "benchmark"), "benchmark_assets": assets,
        "harness_hash": tree_hash(root / "harness/cam"),
    }
    write(root / "input.json", plan)
    with (root / "process.log").open("xb") as log:
        result = subprocess.run(
            [str(python), "-I", str(Path(__file__).resolve()), "--worker", str(root / "input.json")],
            stdout=log, stderr=subprocess.STDOUT, timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu),
                 "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        )
    if (root / "check.json").exists():
        print((root / "check.json").read_text())
    else:
        print(f"Check failed; inspect {root / 'process.log'}", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
