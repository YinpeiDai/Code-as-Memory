"""MIKASA-Robo v2 demonstrations: released expert actions replayed at the v2 interface.

The pinned RLDS release (``mikasa-robo/mikasa-robo-vla-rlds``) has 128x128 RGB only.
Its recorded commands reproduce exactly when replayed under each collector's own
simulator setup (verified: identical first frames, matching TCP, native success):

* ``data_source = MP`` tasks were recorded by ``get_mikasa_robo_datasets_motion_planning``
  in one environment with the default (CPU) physics backend and the plain
  flatten/EEF wrapper chain, reset with the episode seed;
* ``data_source = PPO`` tasks were recorded by ``get_mikasa_robo_datasets`` in batches
  of 10 GPU environments reset with seeds ``10k..10k+9`` and the collector's wrapper
  list (which includes the cue-phase no-op wrapper). Scene randomization depends on
  the whole batch, so the batch is rebuilt and every released batch-mate's commands
  are replayed alongside (zeros where none were released).

Rendering at 256x256 RGB-D does not change physics. Each replay records the v2
frames (front/wrist RGB, int16 depth, intrinsics, extrinsics, TCP, joints,
fingers) before every recorded command and stops at the first native success;
only replays that reach native success are kept, in a documented seeded random
order over all released episodes, up to ``count`` per task (shortfalls recorded,
never padded). The orchestrator runs in the CaM environment (release verification
and parsing, no simulator); one worker per task runs in the MIKASA interpreter.
"""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from cam.benchmarks.base import source_hash
from cam.benchmarks.mikasa_robo import METADATA, dataset_name
from cam.benchmarks.registry import get_adapter, simulator_python, checkout
from cam.data import demonstrations
from cam.io import read, sha256, write

BENCHMARK = "mikasa_robo_v2"
IMAGE = (256, 256, 3)

def spec(dtype, shape, description):
    return {"dtype": dtype, "shape": list(shape), "description": description}


DECLARED = demonstrations.declaration(
    {
        "front_rgb": spec("uint8", IMAGE, "front (MIKASA base_camera) RGB 256x256"),
        "wrist_rgb": spec("uint8", IMAGE, "wrist (MIKASA hand_camera) RGB 256x256"),
        "front_depth": spec("int16", IMAGE[:2], "front depth, int16 optical-axis millimetres, 0 unknown"),
        "wrist_depth": spec("int16", IMAGE[:2], "wrist depth, int16 optical-axis millimetres, 0 unknown"),
        "front_extrinsic": spec("float32", (3, 4), "front 3x4 world-to-camera [R t], OpenCV axes, metres"),
        "wrist_extrinsic": spec("float32", (3, 4), "wrist 3x4 world-to-camera [R t], OpenCV axes, metres"),
        "tcp": spec("float32", (6,), "measured TCP XYZ metres (world) and roll/pitch/yaw radians"),
        "joints": spec("float32", (7,), "Panda arm joint positions J1..J7, radians"),
        "fingers": spec("float32", (2,), "Panda finger joint positions, metres"),
        "action": spec("float32", (7,), "recorded expert command in robot.step units (normalized pd_ee_delta_pose)"),
    },
    {
        "front_K": spec("float32", (3, 3), "front 3x3 intrinsics of the 256x256 image (OpenCV)"),
        "wrist_K": spec("float32", (3, 3), "wrist 3x3 intrinsics of the 256x256 image (OpenCV)"),
    },
)
ALIGNMENT = (
    "action[t] is the released expert command issued after observing frame t; frame t+1 is "
    "its result. The episode ends at the first native success; that result frame is not recorded."
)
NOOP_NOTE = " During the task's initial cue phase the environment replaces commands by zero."
STEP_SEMANTICS = "Native control step: frame t is the observation after t commands; frame 0 is the reset observation."
SELECTION = {
    "rule": "seeded random order over all released training episodes; keep replays that reach native success",
    "seed": "cam.data.demonstrations.selection_seed('mikasa_robo_v2', task)",
}
REPLAY = {
    "MP": "single environment, default (CPU) physics, motion-planning collector wrapper chain, reset(seed)",
    "PPO": "batched GPU environments (collector batch size), seeds 10k..10k+9, collector wrappers; batch-mates replay their released commands",
}


# ----------------------------------------------------------------------------
# Orchestrator (CaM environment): verify and parse the release, then replay.
# ----------------------------------------------------------------------------


def release_episodes(raw, adapter, task):
    """All released episodes of a task: hash-verified, parsed, identity-checked."""
    from cam.benchmarks import mikasa_robo_data as v1

    train = {int(r["episode"]): r for r in adapter.records(task, "train")}
    info = v1.release()
    expected = v1.task_files(task, info["files"])
    folder = Path(raw) / dataset_name(task) / v1.VERSION
    present = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
    if present != set(expected):
        raise ValueError(f"Raw release files differ for {task}")
    native = v1.verify_json(folder, expected, "metadata.json")
    seeds = [int(s) for s in native["episode_seeds"]]
    if native["env_id"] != task or seeds != [int(train[k]["seed"]) for k in range(len(train))]:
        raise ValueError(f"Release metadata disagrees with pinned identities: {task}")
    episodes = {}
    for name in v1.shards(expected):
        digest = hashlib.sha256()
        for payload in v1.tfrecords(folder / name, digest):
            item = v1.parse(payload, task, train)
            if item["episode"] in episodes:
                raise ValueError(f"Duplicate released episode {task}/{item['episode']}")
            episodes[item["episode"]] = {
                "episode": item["episode"],
                "seed": int(train[item["episode"]]["seed"]),
                "goal": item["goal"],
                "action": item["action"],
                "tcp": item["proprio"][:, :6],
            }
        if digest.hexdigest() != expected[name]["sha256"]:
            raise ValueError(f"Shard SHA-256 differs from the release: {task}/{name}")
    if set(episodes) != set(train):
        raise ValueError(f"Released episodes differ from pinned identities: {task}")
    return episodes, info, native


def candidate_order(episodes, task):
    rng = np.random.default_rng(demonstrations.selection_seed(BENCHMARK, task))
    ids = sorted(episodes)
    return [ids[i] for i in rng.permutation(len(ids))]


def assets_dir(adapter):
    """ManiSkill asset root (YCB cups for ShellGame*), exactly as the adapter resolves it."""
    from cam.benchmarks.mikasa_robo import default_assets

    pinned = adapter.source / adapter.asset_dir
    path = pinned if pinned.is_dir() else default_assets()
    if path is None or not Path(path).is_dir():
        raise FileNotFoundError("MIKASA-Robo ManiSkill assets missing; see envs/mikasa_robo/README.md")
    return Path(path).resolve()


def prepare_job(raw, output, adapter, task, count, videos):
    episodes, info, native = release_episodes(raw, adapter, task)
    pinned = adapter.metadata(task, "train")
    folder = Path(output) / "jobs" / task
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(
        folder / "release.npz",
        **{f"action_{k}": v["action"] for k, v in episodes.items()},
        **{f"tcp_{k}": v["tcp"] for k, v in episodes.items()},
    )
    job = {
        "task": task,
        "source": str(adapter.source),
        "assets": str(assets_dir(adapter)),
        "output": str(Path(output).resolve()),
        "count": int(count),
        "videos": bool(videos),
        "data_source": pinned["data_source"],
        "noop": bool(pinned.get("noop_cue_phase", False)),
        "order": candidate_order(episodes, task),
        "episodes": {str(k): {"seed": v["seed"], "goal": v["goal"]} for k, v in episodes.items()},
        "source_label": f"{info['repo']}@{info['revision']}/{dataset_name(task)}/1.0.0 (replayed)",
    }
    write(folder / "job.json", job)
    return folder / "job.json"


def run_worker(job_path, gpu, log):
    adapter = get_adapter(BENCHMARK)
    python = simulator_python(adapter)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(checkout() / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = os.environ.get("CAM_REPLAY_THREADS", "3")
    with open(log, "a") as stream:
        result = subprocess.run(
            [str(python), "-m", "cam.benchmarks.mikasa_robo_v2_data", "worker", str(job_path)],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,  # an interactive prompt must fail, never hang
        )
    if result.returncode:
        raise RuntimeError(f"Replay worker failed ({result.returncode}); see {log}")
    return read(Path(job_path).parent / "result.json")


def extract(
    raw_data,
    output,
    benchmark=BENCHMARK,
    count=100,
    tasks=None,
    depth=True,
    videos=True,
    workers=4,
    benchmark_source=None,
    resume=False,
    gpus=None,
):
    if gpus is None:  # CAM_REPLAY_GPUS="0,1" spreads task workers over GPUs
        gpus = tuple(int(g) for g in os.environ.get("CAM_REPLAY_GPUS", "0").split(",") if g.strip())
    if benchmark != BENCHMARK or not depth:
        raise ValueError("mikasa_robo_v2 demonstrations are RGB-D replays")
    raw_data, output = Path(raw_data).resolve(), Path(output).resolve()
    adapter = get_adapter(benchmark, benchmark_source)
    tasks = list(tasks or adapter.tasks())
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(adapter.tasks()) or workers < 1:
        raise ValueError("Choose distinct native tasks and positive workers")
    from cam.benchmarks import mikasa_robo_data as v1

    info = v1.release()
    output.mkdir(parents=True, exist_ok=resume)
    manifest = {
        "schema": "cam-dataset/1",
        "demonstration_schema": demonstrations.SCHEMA,
        "created": time.time(),
        "benchmark": benchmark,
        "benchmark_hash": source_hash(adapter),
        "raw_data": str(raw_data),
        "depth": True,
        "fields": DECLARED,
        "selection": {"count": int(count), **SELECTION},
        "requested_tasks": tasks,
        "videos": bool(videos),
        "release": {"repo": info["repo"], "revision": info["revision"], "listing_sha256": sha256(METADATA / "release.json")},
        "replay": REPLAY,
        "tasks": {},
    }
    if resume and (output / "manifest.partial.json").exists():
        previous = read(output / "manifest.partial.json")
        for key in ("benchmark", "benchmark_hash", "raw_data", "fields", "selection", "requested_tasks", "videos", "release"):
            if manifest[key] != previous[key]:
                raise ValueError(f"Replay resume configuration changed: {key}")
        manifest["created"], manifest["tasks"] = previous["created"], previous["tasks"]
    write(output / "manifest.partial.json", manifest)
    pending = [t for t in tasks if t not in manifest["tasks"]]
    logs = output / "logs"
    logs.mkdir(exist_ok=True)

    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
    import queue

    # Two stages. NFS-bound preparation (release verification/parsing, reuse checks,
    # hashing) runs in a few processes ahead of recording; recording runs in exactly
    # ``workers`` slots spread evenly over the GPUs (e.g. 8 = 4 per GPU).
    slots = queue.Queue()
    for i in range(workers):
        slots.put(gpus[i % len(gpus)] if gpus else None)

    def record(task, job):
        gpu = slots.get()
        try:
            result = run_worker(job, gpu, logs / f"{task}.log")
        finally:
            slots.put(gpu)
        return "done", task, _item(task, output, result)

    prepare_args = [(task, str(raw_data), str(output), int(count), bool(videos), bool(resume), benchmark_source) for task in pending]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("CAM_REPLAY_PREPARE", "3"))) as prep, ThreadPoolExecutor(max_workers=workers) as rec:
        running = {prep.submit(prepare_task, args) for args in prepare_args}
        while running:
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                kind, task, value = future.result()
                if kind == "job":
                    running.add(rec.submit(record, task, value))
                else:  # "done": a recorded (or reused) task
                    manifest["tasks"][task] = value
                    write(output / "manifest.partial.json", manifest)
    manifest["tasks"] = {task: manifest["tasks"][task] for task in tasks}
    write(output / "manifest.json", manifest)
    (output / "manifest.partial.json").unlink(missing_ok=True)
    validate_dataset(output)
    return manifest


def _item(task, output, result):
    destination = Path(output) / result["file"]
    return {
        "task": task,
        "file": result["file"],
        "sha256": sha256(destination),
        "episodes": result["episodes"],
        "replay": {k: result[k] for k in ("data_source", "attempted", "accepted", "rejected", "shortfall")},
    }


def finished_result(output, task, count):
    """A completed worker result that can be reused on resume, or None.

    Reused only when the job had the same count and the H5 holds exactly the
    accepted episodes with their recorded lengths."""
    import h5py

    folder = Path(output) / "jobs" / task
    if not (folder / "result.json").is_file() or not (folder / "job.json").is_file():
        return None
    job, result = read(folder / "job.json"), read(folder / "result.json")
    destination = Path(output) / result["file"]
    if job["count"] != count or not destination.is_file():
        return None
    with h5py.File(destination, "r") as h5:
        stored = {key: h5[key]["action"].shape[0] for key in h5}
    if stored != {f"episode_{e['episode']}": e["frames"] for e in result["episodes"]}:
        raise ValueError(f"Completed replay of {task} disagrees with its result record")
    return result


def prepare_task(packed):
    """NFS-bound stage: reuse a verified finished task, or write its replay job."""
    task, raw_data, output, count, videos, resume, benchmark_source = packed
    output = Path(output)
    adapter = get_adapter(BENCHMARK, benchmark_source)
    if resume:
        result = finished_result(output, task, count)
        if result is not None:
            return "done", task, _item(task, output, result)
        for path in (output / "h5" / f"record_dataset_{task}.h5.partial", output / "rollouts" / task):
            if path.exists():  # never overwrite partial evidence
                target = output / "interrupted" / str(time.time_ns()) / path.name
                target.parent.mkdir(parents=True)
                shutil.move(path, target)
    return "job", task, prepare_job(raw_data, output, adapter, task, count, videos)


def validate_dataset(root, verify_hashes=True):
    manifest = demonstrations.validate_dataset(root, verify_hashes)
    if manifest["benchmark"] != BENCHMARK or manifest["fields"] != DECLARED or not manifest["depth"]:
        raise ValueError("Not a mikasa_robo_v2 RGB-D replay dataset")
    adapter = get_adapter(BENCHMARK)
    for task, item in manifest["tasks"].items():
        train = {int(r["episode"]): int(r["seed"]) for r in adapter.records(task, "train")}
        for row in item["episodes"]:
            if train.get(int(row["episode"])) != int(row["seed"]):
                raise ValueError(f"Demonstration identity is not a pinned training episode: {task}/{row['episode']}")
    return manifest


# ----------------------------------------------------------------------------
# Worker (MIKASA interpreter): collector-faithful replay with v2 sensors.
# ----------------------------------------------------------------------------


def _environment(job):
    import gymnasium as gym

    sys.path.insert(0, job["source"])
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401  (registers env IDs)
    from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
    from mikasa_robo_suite.vla.utils.wrappers import ConvertJointsToEEFXyzRpyGripperWrapper, StateOnlyTensorToDictWrapper

    from cam.benchmarks.mikasa_robo_v2 import recorder_class

    sensors = {"obs_mode": "rgbd", "sensor_configs": {"width": IMAGE[1], "height": IMAGE[0]}}
    task = job["task"]
    if job["data_source"] == "MP":  # get_mikasa_robo_datasets_motion_planning._create_pd_ee_rollout_env
        recorder = recorder_class()(
            gym.make(task, num_envs=1, control_mode="pd_ee_delta_pose", render_mode="rgb_array", reward_mode="normalized_dense", **sensors)
        )
        env = StateOnlyTensorToDictWrapper(recorder)
        batch = 1
    else:  # get_mikasa_robo_datasets.collect_batched_data_from_ckpt (env_rgb)
        from mikasa_robo_suite.vla.dataset_collectors import get_mikasa_robo_datasets as collector

        batch = collector.BATCH_SIZE_OVERRIDE_BY_ENV.get(task, collector.DEFAULT_BATCH_SIZE)
        recorder = recorder_class()(
            gym.make(task, num_envs=batch, control_mode="pd_ee_delta_pose", render_mode="all", sim_backend="gpu", reward_mode="normalized_dense", **sensors)
        )
        env = recorder
        wrappers, _ = collector.env_info(task)
        for cls, kwargs in wrappers:
            env = cls(env, **kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False, oracle=False, joints=True)
    env = ConvertJointsToEEFXyzRpyGripperWrapper(env)
    return env, recorder, batch


def _select(value, index):
    """Index env ``index`` of a batched (nested) observation, keeping a batch axis of 1."""
    if isinstance(value, dict):
        return {k: _select(v, index) for k, v in value.items()}
    if hasattr(value, "shape") and len(value.shape) and value.shape[0] > index:
        return value[index : index + 1]
    return value


def worker(job_path):
    import imageio.v2 as imageio
    import torch

    from cam.benchmarks.mikasa_robo_v2 import frame

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    job_path = Path(job_path)
    job = read(job_path)
    if job["assets"]:
        os.environ["MS_ASSET_DIR"] = job["assets"]
    release = np.load(job_path.parent / "release.npz")
    task, output = job["task"], Path(job["output"])
    env, recorder, batch = _environment(job)
    device = env.unwrapped.device
    seed_to_episode = {int(v["seed"]): int(k) for k, v in job["episodes"].items()}
    destination = output / "h5" / f"record_dataset_{task}.h5"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".h5.partial")
    accepted, rejected, attempted = [], [], 0
    with demonstrations.create(partial, BENCHMARK, task, DECLARED, True, job["source_label"]) as h5:
        for episode in job["order"]:
            if len(accepted) >= job["count"]:
                break
            attempted += 1
            info = job["episodes"][str(episode)]
            seed, actions, released_tcp = int(info["seed"]), release[f"action_{episode}"], release[f"tcp_{episode}"]
            if batch == 1:
                index, seeds = 0, int(seed)
                mates = {0: actions}
            else:
                k, index = divmod(seed, batch)
                seeds = [k * batch + j for j in range(batch)]
                mates = {j: release[f"action_{seed_to_episode[s]}"] for j, s in enumerate(seeds) if s in seed_to_episode}
            obs, start = env.reset(seed=seeds)
            goal = start.get("language_instruction")
            goal = goal[index] if isinstance(goal, (list, tuple)) else goal
            rows, deviation, success, step = [], 0.0, False, 0
            for step, command in enumerate(actions):
                raw = _select(recorder.last, index)
                rows.append(frame(raw, {k: _select(v, index) for k, v in obs.items() if k in ("rgb", "proprio")}, step))
                rows[-1]["action"] = np.asarray(command, dtype=np.float32)
                deviation = max(deviation, float(np.abs(rows[-1]["tcp"][:3] - released_tcp[step, :3]).max()))
                if batch == 1:
                    obs, _, terminated, truncated, result = env.step(np.clip(command, -1, 1))
                    done = bool(np.asarray(terminated).reshape(-1)[0] or np.asarray(truncated).reshape(-1)[0])
                    success = bool(np.asarray(result["success"]).reshape(-1)[0])
                else:
                    batch_action = torch.zeros((batch, 7), device=device)
                    for j, values in mates.items():
                        if step < len(values):
                            batch_action[j] = torch.as_tensor(values[step], device=device)
                    obs, _, terminated, truncated, result = env.step(batch_action)
                    done = False  # the collector ignored per-env terminations
                    success = bool(result["success"][index])
                if success or done:
                    break
            if not success:
                rejected.append({"episode": episode, "seed": seed, "reason": "replay did not reach native success", "steps": step + 1, "max_tcp_deviation_m": round(deviation, 5)})
                continue
            if goal != info["goal"]:
                raise ValueError(f"Replayed instruction differs from the release: {task}/{episode}")
            length = len(rows)
            group = demonstrations.Episode(
                h5,
                DECLARED,
                episode,
                length,
                {"goal": info["goal"], "difficulty": None, "action_alignment": ALIGNMENT + (NOOP_NOTE if job["noop"] else ""), "step_semantics": STEP_SEMANTICS},
                static={"front_K": rows[0]["front_K"], "wrist_K": rows[0]["wrist_K"]},
            )
            names = [n for n in DECLARED["frame"] if n not in ("step", "is_demo")]
            for start_row in range(0, length, 32):
                chunk = rows[start_row : start_row + 32]
                group.append({**{n: np.stack([r[n] for r in chunk]) for n in names}, "is_demo": np.zeros(len(chunk), dtype=bool)})
            group.close()
            video = None
            if job["videos"]:
                video = output / "rollouts" / task / f"episode_{episode}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                with imageio.get_writer(video, fps=20, codec="libx264", quality=7, ffmpeg_params=["-threads", "1"], ffmpeg_log_level="error") as stream:
                    for r in rows:
                        stream.append_data(np.hstack([r["front_rgb"], r["wrist_rgb"]]))
                video = video.relative_to(output).as_posix()
            accepted.append(
                {
                    "episode": int(episode),
                    "seed": seed,
                    "frames": length,
                    "demo_frames": 0,
                    "goal": info["goal"],
                    "video": video,
                    "release_frames": int(len(actions)),
                    "max_tcp_deviation_m": round(deviation, 5),
                }
            )
            print(f"{task}: accepted {len(accepted)}/{job['count']} (episode {episode}, {length} frames, dev {deviation:.4f} m)", flush=True)
    env.close()
    partial.replace(destination)
    result = {
        "file": destination.relative_to(output).as_posix(),
        "data_source": job["data_source"],
        "attempted": attempted,
        "accepted": len(accepted),
        "rejected": rejected,
        "shortfall": max(0, job["count"] - len(accepted)),
        "episodes": sorted(accepted, key=lambda r: r["episode"]),
    }
    write(job_path.parent / "result.json", result)


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        worker(sys.argv[2])
    else:
        raise SystemExit("usage: python -m cam.benchmarks.mikasa_robo_v2_data worker JOB")
