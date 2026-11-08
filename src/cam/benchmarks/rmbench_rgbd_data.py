"""Regenerate native RMBench experts at released training seeds with aligned RGB-D.

The release does NOT contain its dense control paths. This is explicitly new
expert collection, not exact replay of released HDF5 state rows. Native expert
paths are retained privately so these newly collected trajectories can be replayed.
Every candidate is attempted once; only native successful collections enter H5s.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import h5py
import numpy as np

from cam.benchmarks import rmbench, rmbench_data
from cam.benchmarks.base import source_hash
from cam.benchmarks.registry import simulator_python
from cam.data import demonstrations as demos
from cam.io import read, sha256, write

PROTOCOL = "rmbench-native-expert-rgbd/1"
ORIGIN = (
    "New official-expert collection at released RMBench training seeds; dense "
    "original paths are not released. RGB and depth are rendered together; "
    "not depth appended to the original JPEG frames."
)
ALIGNMENT = (
    "Sampled native scripted-expert states, not action commands. Rendered RGB-D "
    "and robot state are synchronous; no released state row is replayed as an action."
)


def fields():
    declared = rmbench_data.fields()
    for name in rmbench.CAMERAS:
        declared["frame"][name + "_rgb"]["description"] = name + " RGB rendered during native expert regeneration"
        declared["frame"][name + "_depth"] = {
            "dtype": "int16", "shape": [rmbench.HEIGHT, rmbench.WIDTH],
            "description": name + " aligned optical-axis depth, millimetres; 0 invalid/out of range",
        }
    return declared


class Recorder:
    """Append only public fields at the official collector's picture boundaries."""

    def __init__(self, path, task, episode, goal):
        self.h5 = demos.create(path, "rmbench", task, fields(), True, ORIGIN)
        self.group = self.h5.create_group(f"episode_{episode}")
        self.group.attrs.update(
            episode=episode, goal=goal, difficulty="", action_alignment=ALIGNMENT,
            step_semantics="Recording frame index; native collector save_freq=15, including motion boundaries.",
        )
        self.n = 0
        self.first = None
        for name, spec in fields()["frame"].items():
            shape = tuple(spec["shape"])
            self.group.create_dataset(
                name, shape=(0, *shape), maxshape=(None, *shape), dtype=spec["dtype"],
                chunks=(1, *shape) if shape else (256,), compression="lzf" if shape else None,
            )

    def append(self, obs):
        frame = rmbench.frame(obs, self.n, depth=True)
        if self.first is None:
            self.first = frame
        values = {k: v for k, v in frame.items() if k not in ("phase", "step")}
        values.update(step=self.n, is_demo=False)
        for name, spec in fields()["frame"].items():
            value = np.asarray(values[name], dtype=spec["dtype"])
            if value.shape != tuple(spec["shape"]):
                raise ValueError(f"Bad sensor shape: {name}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite sensor: {name}")
            ds = self.group[name]
            ds.resize(self.n + 1, axis=0)
            ds[self.n] = value
        self.n += 1

    def close(self):
        self.h5.close()


def worker(job_path):
    """One native episode per process: planner/renderer failures stay isolated."""
    job = read(job_path)
    folder = Path(job["folder"])
    result = {"episode": job["episode"], "seed": job["seed"], "status": "infrastructure_error",
              "success": False, "frames": 0, "protocol": PROTOCOL}
    started = time.time()
    native = recorder = None
    try:
        adapter = rmbench.RMBench(job["source"])
        module = adapter.import_native(job["task"])
        args = adapter.task_arguments(job["task"])
        # Official collection, not eval take_action or a generated policy.
        args.update(eval_mode=False, need_plan=True, save_data=False, render_freq=0,
                    save_freq=15, save_path=str(folder / "native"))
        args["data_type"]["depth"] = True
        native = getattr(module, job["task"])()
        native.setup_demo(now_ep_num=job["episode"], seed=job["seed"], **args)
        recorder = Recorder(folder / "observations.h5", job["task"], job["episode"], adapter.goal(job["task"]))

        def picture():
            # Same get_obs call as native _take_picture; only the storage changes.
            native.language_annotation_cache += 1
            recorder.append(native.get_obs())
            native.FRAME_IDX += 1

        native._take_picture = picture
        native.save_data = True
        native.play_once()
        success = bool(native.plan_success and native.check_success())
        # Terminal snapshot, even when the last collector boundary preceded the check.
        picture()
        native.save_traj_data(job["episode"])
        result.update(status="success" if success else "expert_failed", success=success,
                      plan_success=bool(native.plan_success), frames=recorder.n)
        if recorder.first is not None:
            first = recorder.first
            decode = rmbench_data.release_decoder(adapter.source)
            with h5py.File(job["raw_episode"], "r") as original:
                result["reset_rgb_mae_vs_release"] = {
                    name: float(np.abs(first[name + "_rgb"].astype(float) -
                                       rmbench_data.images(original, group, 0, 1, decode)[0]).mean())
                    for name, (_, group) in rmbench.CAMERAS.items()
                }
            result["reset_depth_valid_fraction"] = {
                name: float((first[name + "_depth"] > 0).mean()) for name in rmbench.CAMERAS
            }
            if not all(v > 0 for v in result["reset_depth_valid_fraction"].values()):
                raise ValueError("A camera has no valid depth")
    except Exception:
        result.update(status="infrastructure_error", success=False, error=traceback.format_exc())
    finally:
        if recorder is not None:
            result["frames"] = recorder.n
            recorder.close()
        if native is not None:
            try:
                rmbench.NativeEnvironment(native, "", 0).close()
            except Exception:
                result["close_error"] = traceback.format_exc()
        result["seconds"] = time.time() - started
        if (folder / "observations.h5").exists():
            result["observations_sha256"] = sha256(folder / "observations.h5")
        write(folder / "result.json", result)
    return result


def gpu_processes(gpu):
    """Fail closed if GPU process accounting is unavailable."""
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, name = line.split(",", 1)
        rows.append({"pid": int(pid.strip()), "name": name.strip()})
    return rows


@contextmanager
def exclusive_gpu(output, gpu, *, allow_sharing=False, workers=1):
    """One collector owns each GPU; sharing other workloads requires opt-in."""
    lock_root = Path(os.environ.get("CAM_RGBD_LOCK_DIR", "/tmp/cam-rmbench-rgbd-locks"))
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / f"gpu_{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            busy = gpu_processes(gpu)
            if allow_sharing:
                free = int(subprocess.check_output(
                    ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    text=True).strip())
                # Measured native pilot including two planner children: ~6.4 GiB.
                if free >= workers * 7000 + 2048:
                    write(output / "audit" / f"gpu_{gpu}.json", {
                        "gpu": gpu, "workers": workers, "allow_sharing": True,
                        "free_mib": free, "existing_processes": busy, "updated": time.time()})
                    break
            elif not busy:
                break
            write(output / "status.json", {"status": "waiting_for_gpu", "gpu": gpu,
                                           "blocking_processes": busy, "updated": time.time()})
            time.sleep(20)
        yield


def gpu_allocation(workers, gpus):
    gpus = tuple(int(g) for g in gpus)
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("Choose distinct nonnegative GPU IDs")
    if workers < len(gpus) or workers % len(gpus) or workers // len(gpus) > 6:
        raise ValueError("Distribute workers evenly across GPUs, at most six per GPU")
    return {gpu: workers // len(gpus) for gpu in gpus}


def reuse_capture(job_path, source_pool):
    """Carry forward a completed attempt, including failures, without rerunning it."""
    job = read(job_path)
    folder = Path(job["folder"])
    old = Path(source_pool) / "jobs" / job["task"] / folder.name
    if (folder / "result.json").exists() or not (old / "result.json").exists():
        return
    prior, result = read(old / "job.json"), read(old / "result.json")
    for key in ("task", "episode", "seed", "raw_crc32", "protocol"):
        if prior[key] != job[key]:
            raise ValueError(f"Reused capture identity changed: {key}")
    if result["episode"] != job["episode"] or result["seed"] != job["seed"] or result["protocol"] != PROTOCOL:
        raise ValueError("Reused result identity changed")
    if result.get("observations_sha256"):
        if sha256(old / "observations.h5") != result["observations_sha256"]:
            raise ValueError("Reused capture changed")
        shutil.copy2(old / "observations.h5", folder / "observations.h5")
    if (old / "native").exists():
        shutil.copytree(old / "native", folder / "native")
    if (old / "worker.log").exists():
        shutil.copy2(old / "worker.log", folder / "reused_worker.log")
    result["reused_from"] = {"folder": str(old), "job_sha256": sha256(old / "job.json"),
                             "result_sha256": sha256(old / "result.json"), "source": prior["source"]}
    write(folder / "result.json", result)


def run_episode(job_path, python, gpu):
    job = read(job_path)
    folder = Path(job["folder"])
    if (folder / "result.json").exists():
        result = read(folder / "result.json")
        if result.get("observations_sha256") and sha256(folder / "observations.h5") != result["observations_sha256"]:
            raise ValueError("Existing episode capture changed")
        return result
    if (folder / "observations.h5").exists():
        raise ValueError(f"Unfinished episode evidence requires explicit recovery: {folder}")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    with (folder / "worker.log").open("x") as log:
        proc = subprocess.Popen(
            [str(python), "-m", "cam.benchmarks.rmbench_rgbd_data", "worker", str(job_path)],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        write(folder / "process.json", {"pid": proc.pid, "gpu": gpu, "started": time.time()})
        try:
            code = proc.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = -signal.SIGTERM
    if not (folder / "result.json").exists():
        write(folder / "result.json", {"episode": job["episode"], "seed": job["seed"],
              "status": "infrastructure_error", "success": False, "frames": 0,
              "returncode": code, "protocol": PROTOCOL})
    return read(folder / "result.json")


def assemble(output, task, records):
    """Materialize a self-contained successful-only training H5."""
    destination = output / "h5" / f"record_dataset_{task}.h5"
    destination.parent.mkdir(exist_ok=True)
    results = [read(output / "jobs" / task / f"ep{int(r['episode']):05d}" / "result.json") for r in records]
    if destination.exists():
        raise FileExistsError(destination)
    episodes = []
    with demos.create(destination.with_suffix(".h5.partial"), "rmbench", task, fields(), True, ORIGIN) as dst:
        for r in results:
            if not r["success"]:
                continue
            episode = int(r["episode"])
            srcpath = output / "jobs" / task / f"ep{episode:05d}" / "observations.h5"
            if sha256(srcpath) != r["observations_sha256"]:
                raise ValueError("Episode changed before assembly")
            with h5py.File(srcpath) as src:
                key = f"episode_{episode}"
                src.copy(key, dst)
                goal = src[key].attrs["goal"]
            episodes.append({"episode": episode, "seed": int(r["seed"]), "frames": r["frames"],
                             "demo_frames": 0, "goal": goal, "video": None})
    destination.with_suffix(".h5.partial").replace(destination)
    return {"task": task, "file": destination.relative_to(output).as_posix(),
            "sha256": sha256(destination), "episodes": episodes,
            "replay": {"protocol": PROTOCOL, "attempted": len(results), "accepted": len(episodes),
                       "shortfall": len(results) - len(episodes), "identities": results}}


def extract(raw_data, output, benchmark="rmbench", count="all", tasks=None, depth=True,
            videos=False, workers=4, benchmark_source=None, resume=False, gpu=None,
            gpus=None, allow_gpu_sharing=False, reuse=None):
    if benchmark != "rmbench" or not depth:
        raise ValueError("Choose RMBench RGB-D")
    output, raw_data = Path(output).resolve(), Path(raw_data).resolve()
    adapter = rmbench.RMBench(benchmark_source)
    python = simulator_python(adapter)
    tasks = list(tasks or adapter.tasks())
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(adapter.tasks()):
        raise ValueError("Choose distinct RMBench tasks")
    gpu = int(os.environ.get("CAM_REPLAY_GPU", "1")) if gpu is None else int(gpu)
    allocation = gpu_allocation(workers, gpus if gpus is not None else (gpu,))
    output.mkdir(parents=True, exist_ok=resume)
    manifest = {"schema": "cam-dataset/1", "demonstration_schema": demos.SCHEMA,
                "created": time.time(), "benchmark": benchmark, "benchmark_hash": source_hash(adapter),
                "raw_data": str(raw_data), "depth": True, "fields": fields(), "requested_tasks": tasks,
                "selection": {"count": count, "rule": "deterministic training identities chosen before execution; retain native successful collections only; no padding"},
                "replay": {"protocol": PROTOCOL, "source": ORIGIN, "save_freq": 15},
                "collection": {"workers_by_gpu": {str(g): n for g, n in allocation.items()}, "allow_gpu_sharing": allow_gpu_sharing,
                               "reuse": str(Path(reuse).resolve()) if reuse else None},
                "videos": False, "tasks": {}}
    partial = output / "manifest.partial.json"
    if resume:
        previous = read(output / "manifest.json" if (output / "manifest.json").exists() else partial)
        for key in ("benchmark", "benchmark_hash", "raw_data", "depth", "fields", "requested_tasks", "selection", "replay", "collection"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume inputs changed: {key}")
        manifest["created"], manifest["tasks"] = previous["created"], previous["tasks"]
        for item in manifest["tasks"].values():
            if sha256(output / item["file"]) != item["sha256"]:
                raise ValueError("Completed dataset changed")
    write(partial, manifest)
    if reuse:
        previous_path = Path(reuse) / "manifest.json"
        previous = read(previous_path if previous_path.exists() else Path(reuse) / "manifest.partial.json")
        for key in ("benchmark", "benchmark_hash", "raw_data", "depth", "fields", "replay"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Capture reuse inputs differ: {key}")
    task_records = {t: demos.random_select(adapter.records(t, "train"), count, "rmbench", t)
                    for t in tasks if t not in manifest["tasks"]}
    expected_count = sum(map(len, task_records.values()))
    progress = {"completed": 0, "successes": 0, "expected": expected_count,
                "workers_by_gpu": allocation}
    progress_lock = threading.Lock()

    def collect(path, device):
        result = run_episode(path, python, device)
        with progress_lock:
            progress["completed"] += 1
            progress["successes"] += bool(result["success"])
            write(output / "status.json", {"status": "collecting", **progress, "updated": time.time()})
            print(json.dumps(progress), flush=True)
        return result

    with ExitStack() as stack:
        pools = {}
        for device, slots in sorted(allocation.items()):
            stack.enter_context(exclusive_gpu(output, device, allow_sharing=allow_gpu_sharing, workers=slots))
            pools[device] = stack.enter_context(ThreadPoolExecutor(max_workers=slots))
        futures, pilot_ok, dispatch_index = [], False, 0
        devices = list(allocation)
        for task, selected in task_records.items():
            records = adapter.records(task, "train")
            if {r["seed"] for r in records} & {r["seed"] for r in adapter.records(task, "test")}:
                raise ValueError("Training/test seed overlap")
            write(output / "verification.json", {"task": task, "status": "verifying", "updated": time.time()})
            digest, audit = rmbench_data.verify_release(raw_data, task, records, rmbench_data.release_provenance(task))
            write(output / "audit" / f"{task}.release.json", {"zip_sha256": digest, "identities": audit})
            _, directory = rmbench_data.task_paths(raw_data, task)
            for r in selected:
                episode = int(r["episode"])
                original = directory / "data" / f"episode_{episode:07d}.hdf5"
                expected = next(a for a in audit if int(a["episode"]) == episode)
                if rmbench_data.crc32(original) != expected["crc32"]:
                    raise ValueError(f"Raw episode differs from verified archive: {original}")
                folder = output / "jobs" / task / f"ep{episode:05d}"
                folder.mkdir(parents=True, exist_ok=resume)
                spec = {"source": str(adapter.source), "task": task, **r, "raw_episode": str(original),
                        "folder": str(folder), "raw_crc32": expected["crc32"], "protocol": PROTOCOL}
                path = folder / "job.json"
                if path.exists() and read(path) != spec:
                    raise ValueError("Episode job changed")
                write(path, spec)
                if reuse:
                    reuse_capture(path, reuse)
                device = devices[dispatch_index % len(devices)]
                dispatch_index += 1
                if not pilot_ok:
                    first = collect(path, device)
                    if first["status"] == "infrastructure_error":
                        raise RuntimeError(f"RGB-D pilot infrastructure error: {folder / 'result.json'}")
                    if first["success"]:
                        pilot_ok = True
                        write(output / "pilot.json", {"job": str(path), "result": first})
                else:
                    futures.append(pools[device].submit(collect, path, device))
        for future in as_completed(futures):
            future.result()
    for task, records in task_records.items():
        manifest["tasks"][task] = assemble(output, task, records)
        write(partial, manifest)
    write(output / "manifest.json", manifest)
    partial.unlink(missing_ok=True)
    validate_dataset(output)
    write(output / "status.json", {"status": "complete", "updated": time.time(),
          "counts": {t: len(v["episodes"]) for t, v in manifest["tasks"].items()},
          "total": sum(len(v["episodes"]) for v in manifest["tasks"].values())})
    return manifest


def validate_dataset(root, verify_hashes=True):
    manifest = demos.validate_dataset(root, verify_hashes)
    if manifest["benchmark"] != "rmbench" or not manifest["depth"] or manifest["fields"] != fields():
        raise ValueError("Not a regenerated RMBench RGB-D dataset")
    if manifest.get("replay", {}).get("protocol") != PROTOCOL:
        raise ValueError("Missing regeneration provenance")
    adapter = rmbench.RMBench()
    for task, item in manifest["tasks"].items():
        train = {int(r["episode"]): int(r["seed"]) for r in adapter.records(task, "train")}
        test = {int(r["seed"]) for r in adapter.records(task, "test")}
        accepted = {int(r["episode"]): r for r in item["replay"]["identities"] if r["success"] and r["status"] == "success"}
        for r in item["episodes"]:
            ep, seed = int(r["episode"]), int(r["seed"])
            if train.get(ep) != seed or seed in test or ep not in accepted or int(accepted[ep]["seed"]) != seed:
                raise ValueError("Unverified training identity or success")
            if r["demo_frames"] != 0:
                raise ValueError("RMBench has no reset video")
    return manifest


if __name__ == "__main__":
    if sys.argv[1:2] != ["worker"]:
        raise SystemExit("usage: python -m cam.benchmarks.rmbench_rgbd_data worker JOB")
    worker(Path(sys.argv[2]).resolve())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # native planner/renderer libraries can hang during interpreter shutdown
