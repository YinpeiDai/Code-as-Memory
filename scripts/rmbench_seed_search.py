"""Find RMBench's official evaluation seeds for one task (run in envs/rmbench).

Mirrors ``third_party/rmbench/scripts/eval_policy_xpolicylab.py:eval_remote_policy``
with ``seed=0`` (start 100000), ``task_config=demo_clean`` and the default expert
check: one persistent task object walks seeds upward; a seed is kept when
``setup_demo`` is stable, the scripted ``play_once`` expert plans
(``plan_success``) and ``check_success()`` holds, and the evaluation
``setup_demo`` succeeds again. Policy rollouts are not run. Progress is appended
to ``<log-dir>/<task>.jsonl`` (resumable only by restarting from scratch, since
the official search is sequential within one process).

  CUDA_VISIBLE_DEVICES=1 envs/rmbench/.venv/bin/python scripts/rmbench_seed_search.py \
      --task put_back_block --source outputs/rmbench/pinned/rmbench
"""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cam.benchmarks.rmbench import RMBench, TASK_CONFIG, EMBODIMENT  # noqa: E402
from cam.benchmarks.base import source_hash  # noqa: E402


def versions():
    import importlib.metadata as m

    names = ("torch", "sapien", "mplib", "nvidia-curobo", "numpy", "warp-lang")
    return {name: m.version(name) for name in names}


def gpu():
    import torch

    name = torch.cuda.get_device_name(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.split("\n")[0].strip()
    return {"name": name, "driver": driver, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", type=Path, required=True, help="pinned RMBench copy with assets/")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start", type=int, default=100000)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "outputs/rmbench/seed_search")
    parser.add_argument("--output", type=Path, help="pinned test metadata JSON to write")
    args = parser.parse_args()
    args.log_dir = args.log_dir.resolve()  # the simulator import changes directory
    args.output = args.output.resolve() if args.output else None
    reference = RMBench()  # the checkout whose hash identifies the benchmark snapshot
    adapter = RMBench(args.source.resolve())
    module = adapter.import_native(args.task)
    from envs.utils.create_actor import UnStableError

    task_args = adapter.task_arguments(args.task)
    task_env = getattr(module, args.task)()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log = (args.log_dir / f"{args.task}.jsonl").open("w")
    started = time.time()
    found, skipped, seed = [], [], args.start

    def note(**row):
        row["time"] = time.time()
        log.write(json.dumps(row) + "\n")
        log.flush()

    def close(clear_cache=False):
        try:
            task_env.close_env(clear_cache=clear_cache)
        except Exception:
            pass

    while len(found) < args.count and seed < args.start + args.max_attempts:
        t0 = time.time()
        try:
            task_env.setup_demo(now_ep_num=len(found), seed=seed, is_test=True, **task_args)
            task_env.play_once()
            task_env.close_env()
        except UnStableError:
            close()
            skipped.append({"seed": seed, "reason": "unstable"})
            note(seed=seed, kept=False, reason="unstable", seconds=time.time() - t0)
            seed += 1
            continue
        except Exception as exc:
            close()
            skipped.append({"seed": seed, "reason": "expert_error", "error": f"{type(exc).__name__}: {exc}"})
            note(seed=seed, kept=False, reason="expert_error", error=traceback.format_exc()[-1500:], seconds=time.time() - t0)
            seed += 1
            continue
        if not (task_env.plan_success and task_env.check_success()):
            skipped.append({"seed": seed, "reason": "expert_failed"})
            note(seed=seed, kept=False, reason="expert_failed", plan_success=bool(task_env.plan_success), seconds=time.time() - t0)
            seed += 1
            continue
        try:
            task_env.setup_demo(now_ep_num=len(found), seed=seed, is_test=True, **task_args)
        except Exception as exc:
            close()
            reason = "unstable" if isinstance(exc, UnStableError) else "setup_error"
            skipped.append({"seed": seed, "reason": reason})
            note(seed=seed, kept=False, reason=reason, seconds=time.time() - t0)
            seed += 1
            continue
        found.append(seed)
        close(clear_cache=(len(found) + 1) % int(task_args["clear_cache_freq"]) == 0)
        note(seed=seed, kept=True, episode=len(found) - 1, seconds=time.time() - t0)
        print(f"[seed-search] {args.task}: {len(found)}/{args.count} seed={seed}", file=sys.stderr, flush=True)
        seed += 1
    if len(found) < args.count:
        raise SystemExit(f"Only {len(found)} seeds found in {args.max_attempts} attempts")
    revision = subprocess.run(
        ["git", "-C", str(reference.source), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    value = {
        "env_id": args.task,
        "split": "test",
        "records": [{"episode": i, "seed": s} for i, s in enumerate(found)],
        "provenance": {
            "rule": "official eval_policy_xpolicylab.py seed search: start 100000*(1+seed) with "
            "seed=0, keep seeds that are stable, whose play_once expert plans and "
            "check_success() holds, and whose evaluation setup_demo succeeds",
            "benchmark": "rmbench",
            "rmbench_commit": revision,
            "benchmark_hash": source_hash(reference),
            "task_config": TASK_CONFIG,
            "embodiment": EMBODIMENT,
            "expert_check": True,
            "start_seed": args.start,
            "last_seed_tried": seed - 1,
            "skipped": skipped,
            "gpu": gpu(),
            "host": platform.node(),
            "python": sys.version.split()[0],
            "packages": versions(),
            "seconds": time.time() - started,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    output = args.output or ROOT / f"src/cam/benchmarks/metadata/rmbench/test/record_dataset_{args.task}_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + "\n")
    print(f"[seed-search] wrote {output}", file=sys.stderr, flush=True)
    sys.stdout.flush()
    os._exit(0)  # SAPIEN/CuRobo subprocess teardown can hang at interpreter exit


if __name__ == "__main__":
    main()
