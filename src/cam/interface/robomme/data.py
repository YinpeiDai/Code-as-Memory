"""Extract only assigned, observed fields from native RoboMME training H5 files."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import time

import h5py
import imageio.v2 as imageio
import numpy as np

from cam.interface.registry import get_adapter
from cam.io import read, sha256, tree_hash, write

FIELDS = {
    "front_rgb": "obs/front_rgb",
    "wrist_rgb": "obs/wrist_rgb",
    "tcp": "obs/eef_state",
    "joints": "obs/joint_state",
    "fingers": "obs/gripper_state",
    "front_extrinsic": "obs/front_camera_extrinsic",
    "wrist_extrinsic": "obs/wrist_camera_extrinsic",
    "joint_action": "action/joint_action",
    "eef_action": "action/eef_action",
    "waypoint_action": "action/waypoint_action",
    "is_demo": "info/is_video_demo",
}
DEPTH = {camera + "_depth": "obs/" + camera + "_depth" for camera in ("front", "wrist")}
ROOT_ATTRS = {
    "schema",
    "task",
    "split",
    "depth",
    "tcp_frame",
    "depth_units",
    "invalid_depth",
}
EPISODE_ATTRS = {"episode", "difficulty", "goal", "action_alignment", "step_semantics"}


def published_manifest(root, manifest, adapter=None):
    """Read the released 25-demo archive without changing or rereading its H5s.

    Exact train/test metadata must match. The current native source is explicitly
    a compatibility target, not a claim about the historical acquisition build.
    Normalize the historical ``source`` field to the harness's ``raw_data`` key.
    """
    from cam.interface.base import source_hash
    adapter = adapter or get_adapter("robomme")
    for task, item in manifest["tasks"].items():
        for split in ("train", "test"):
            path = adapter.source / "env_metadata" / split / f"record_dataset_{task}_metadata.json"
            if sha256(path) != item.get(split + "_metadata_sha256"):
                raise ValueError(f"Published RoboMME {split} identities changed: {task}")
    return {**manifest, "schema": "cam-dataset/1", "benchmark_hash": source_hash(adapter),
            "raw_data": manifest.get("raw_data") or manifest["source"],
            "compatibility": {"source_schema": manifest["schema"],
                              "source_manifest_sha256": sha256(Path(root) / "manifest.json"),
                              "metadata_hashes_verified": True,
                              "benchmark_hash_role": "current compatibility target; historical build unspecified",
                              "h5_reads": 0, "h5_bytes_changed": False}}


def scalar(value):
    value = np.asarray(value).reshape(-1)[0]
    return (
        value.decode()
        if isinstance(value, bytes)
        else value.item()
        if hasattr(value, "item")
        else value
    )


def select(records, count):
    if count == "all":
        return sorted(records, key=lambda row: int(row["episode"]))
    count = int(count)
    if count < 3:
        raise ValueError("Balanced sampling requires at least three demonstrations")
    selected = []
    for index, difficulty in enumerate(("easy", "medium", "hard")):
        candidates = sorted(
            (r for r in records if r["difficulty"] == difficulty),
            key=lambda r: int(r["episode"]),
        )
        take = count // 3 + int(index < count % 3)
        if len(candidates) < take:
            raise ValueError(
                f"Not enough {difficulty} demonstrations for balanced sampling"
            )
        selected.extend(
            candidates[i]
            for i in np.rint(np.linspace(0, len(candidates) - 1, take)).astype(int)
        )
    if len({int(row["episode"]) for row in selected}) != count:
        raise ValueError("Duplicate selected episode")
    return selected


def frame_keys(group):
    keys = sorted(
        (k for k in group if k.startswith("timestep_")), key=lambda k: int(k[9:])
    )
    if not keys or [int(k[9:]) for k in keys] != list(range(len(keys))):
        raise ValueError(
            f"Expected contiguous modern timestep_N recording: {group.name}"
        )
    return keys


def frame(group, index, depth):
    row = {
        key: np.asarray(group[path][()])
        for key, path in (FIELDS | (DEPTH if depth else {})).items()
    }
    row["is_demo"] = np.asarray(bool(scalar(row["is_demo"])))
    row["step"] = np.asarray(index, dtype=np.int64)
    # Some raw files mix float32/float64 across frames; preserve numeric precision.
    row["waypoint_action"] = row["waypoint_action"].astype(np.float64)
    row["waypoint_action_valid"] = np.asarray(
        bool(np.isfinite(row["waypoint_action"]).all())
    )
    for camera in ("front", "wrist"):
        rgb = row[camera + "_rgb"]
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Expected HxWx3 uint8 RGB")
        if depth:
            value = row[camera + "_depth"]
            if value.ndim == 3 and value.shape[-1] == 1:
                value = value[..., 0]
            if value.dtype != np.int16 or value.shape != rgb.shape[:2]:
                raise ValueError("Expected aligned int16 millimetre depth")
            row[camera + "_depth"] = value
    if any(
        not np.isfinite(value).all()
        for key, value in row.items()
        if key != "waypoint_action"
    ):
        raise ValueError("Nonfinite observation or expert command")
    return row


def export_task(job):
    raw, output, source, benchmark, task, count, depth, videos = job
    raw, output = Path(raw), Path(output)
    adapter = get_adapter(benchmark, source)
    records, test = adapter.records(task, "train"), adapter.records(task, "test")
    selected = select(records, count)
    original = raw / f"record_dataset_{task}.h5"
    before = original.stat()
    destination = output / "h5" / original.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".h5.partial")
    episodes, audit = [], []
    with h5py.File(original, "r") as src, h5py.File(partial, "x") as dst:
        if set(src) != {f"episode_{int(row['episode'])}" for row in records}:
            raise ValueError(f"Raw training IDs disagree with native metadata: {task}")
        for record in records:
            group = src[f"episode_{record['episode']}"]
            keys = frame_keys(group)
            if (
                int(scalar(group["setup/seed"][()])) != int(record["seed"])
                or scalar(group["setup/difficulty"][()]) != record["difficulty"]
            ):
                raise ValueError(
                    f"Raw seed/difficulty mismatch: {task}/{record['episode']}"
                )
            if int(record["seed"]) in {int(r["seed"]) for r in test}:
                raise ValueError("Training/test seed overlap")
            if not bool(scalar(group[keys[-1] + "/info/is_completed"][()])):
                raise ValueError("Raw training recording is not completed")
            audit.append({**record, "frames": len(keys), "verified_completed": True})
        dst.attrs.update(
            schema="cam-demonstrations/1",
            task=task,
            split="train",
            depth=depth,
            tcp_frame="native measured TCP",
            depth_units="millimetres along camera optical axis",
            invalid_depth=0,
        )
        for record in selected:
            episode = int(record["episode"])
            group, video = src[f"episode_{episode}"], None
            keys = frame_keys(group)
            target = dst.create_group(f"episode_{episode}")
            target.attrs.update(
                episode=episode,
                difficulty=record["difficulty"],
                goal=str(scalar(group["setup/task_goal"][()])),
                action_alignment="Unspecified by source; do not assume action[t] produced state[t].",
                step_semantics="Recording frame index, not a native control-step counter.",
            )
            for camera in ("front", "wrist"):
                target.create_dataset(
                    camera + "_K", data=group[f"setup/{camera}_camera_intrinsic"][()]
                )
            first = frame(group[keys[0]], 0, depth)
            for name, value in first.items():
                target.create_dataset(
                    name,
                    shape=(len(keys), *value.shape),
                    dtype=value.dtype,
                    chunks=(1, *value.shape),
                    compression="lzf",
                )
            movie = output / "rollouts" / task / f"episode_{episode}.mp4"
            try:
                if videos:
                    movie.parent.mkdir(parents=True, exist_ok=True)
                    video = imageio.get_writer(
                        movie,
                        fps=20,
                        codec="libx264",
                        quality=7,
                        ffmpeg_params=["-threads", "1"],
                        ffmpeg_log_level="error",
                    )
                for start in range(0, len(keys), 32):
                    batch = [
                        first if i == 0 else frame(group[keys[i]], i, depth)
                        for i in range(start, min(start + 32, len(keys)))
                    ]
                    for name in first:
                        target[name][start : start + len(batch)] = np.stack(
                            [row[name] for row in batch]
                        )
                    if video:
                        for row in batch:
                            video.append_data(
                                np.hstack([row["front_rgb"], row["wrist_rgb"]])
                            )
            finally:
                if video:
                    video.close()
            flags = target["is_demo"][:]
            demo_count = int(flags.sum())
            if demo_count == len(keys) or not np.array_equal(
                flags, np.arange(len(keys)) < demo_count
            ):
                raise ValueError(
                    "Expected native video prefix followed by expert execution"
                )
            episodes.append(
                {
                    **record,
                    "frames": len(keys),
                    "demo_frames": demo_count,
                    "goal": str(target.attrs["goal"]),
                    "video": movie.relative_to(output).as_posix() if videos else None,
                }
            )
    after = original.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Raw file changed during extraction")
    partial.replace(destination)
    item = {
        "task": task,
        "file": destination.relative_to(output).as_posix(),
        "sha256": sha256(destination),
        "episodes": episodes,
        "source_file": str(original),
        "source_size": before.st_size,
        "source_mtime_ns": before.st_mtime_ns,
    }
    write(output / "audit" / (task + ".json"), {"all_training_episodes": audit, **item})
    return task, item


def extract(
    raw_data,
    output,
    benchmark="robomme",
    count=25,
    tasks=None,
    depth=True,
    videos=True,
    workers=2,
    benchmark_source=None,
    resume=False,
):
    raw_data, output = Path(raw_data).resolve(), Path(output).resolve()
    adapter = get_adapter(benchmark, benchmark_source)
    tasks = tasks or adapter.tasks()
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or set(tasks) - set(adapter.tasks())
        or workers < 1
    ):
        raise ValueError("Choose distinct native tasks and positive extraction workers")
    count = "all" if count == "all" else int(count)
    output.mkdir(parents=True, exist_ok=resume)
    manifest = {
        "schema": "cam-dataset/1",
        "created": time.time(),
        "benchmark": benchmark,
        "benchmark_hash": tree_hash(adapter.source),
        "raw_data": str(raw_data),
        "depth": depth,
        "selection": {
            "count": count,
            "rule": "balanced difficulty, evenly spaced ranks; remainder easy then medium",
        },
        "requested_tasks": tasks,
        "videos": videos,
        "tasks": {},
    }
    if resume:
        previous = read(
            output / "manifest.json"
            if (output / "manifest.json").exists()
            else output / "manifest.partial.json"
        )
        for key in (
            "benchmark",
            "benchmark_hash",
            "raw_data",
            "depth",
            "selection",
            "requested_tasks",
            "videos",
        ):
            if manifest[key] != previous[key]:
                raise ValueError(f"Extraction resume configuration changed: {key}")
        manifest["created"] = previous["created"]
    jobs = []
    for task in tasks:
        audit = output / "audit" / (task + ".json")
        if resume and audit.exists():
            item = read(audit)
            stat = Path(item["source_file"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (
                item["source_size"],
                item["source_mtime_ns"],
            ) or sha256(output / item["file"]) != item["sha256"]:
                raise ValueError(f"Completed extraction changed: {task}")
            manifest["tasks"][task] = {
                key: value
                for key, value in item.items()
                if key != "all_training_episodes"
            }
            continue
        if resume:
            # Preserve interrupted files and videos rather than overwriting them.
            paths = [
                output / "h5" / f"record_dataset_{task}{suffix}"
                for suffix in (".h5", ".h5.partial")
            ]
            paths += [output / "rollouts" / task]
            for path in paths:
                if path.exists():
                    target = output / "interrupted" / str(time.time_ns()) / path.name
                    target.parent.mkdir(parents=True)
                    shutil.move(path, target)
        jobs.append(
            (raw_data, output, adapter.source, benchmark, task, count, depth, videos)
        )
    write(output / "manifest.partial.json", manifest)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task, item in pool.map(export_task, jobs):
            manifest["tasks"][task] = item
            write(output / "manifest.partial.json", manifest)
    write(output / "manifest.json", manifest)
    (output / "manifest.partial.json").unlink(missing_ok=True)
    return manifest


def validate_dataset(root, verify_hashes=True):
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    historical = manifest.get("schema") == "cap-regular-extraction/2"
    if historical:
        manifest = published_manifest(root, manifest)
    if manifest.get("schema") != "cam-dataset/1":
        raise ValueError("Expected a fresh CaM raw-data extraction")
    fields = set(FIELDS) | {"step", "waypoint_action_valid", "front_K", "wrist_K"}
    if historical:
        fields.add("tcp_quaternion_wxyz")
    if manifest["depth"]:
        fields |= set(DEPTH)
    for task, item in manifest["tasks"].items():
        path = root / item["file"]
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("Extracted data must be an internal regular file")
        if verify_hashes and sha256(path) != item["sha256"]:
            raise ValueError("Extracted data hash mismatch")
        with h5py.File(path, "r") as h5:
            expected_attrs = ({"action_alignment", "control_freq", "depth", "depth_description", "excluded_fields",
                               "schema", "source_format", "split", "step_semantics", "task"} if historical else ROOT_ATTRS)
            if (
                set(h5.attrs) != expected_attrs
                or h5.attrs["schema"] != ("cap-regular-fewshot/2" if historical else "cam-demonstrations/1")
                or h5.attrs["task"] != task
                or bool(h5.attrs["depth"]) != manifest["depth"]
                or h5.attrs["split"] != "train"
                or (not historical and h5.attrs["tcp_frame"] != "native measured TCP")
                or (not historical and int(h5.attrs["invalid_depth"]) != 0)
            ):
                raise ValueError("Invalid public H5 attributes")
            if set(h5) != {f"episode_{row['episode']}" for row in item["episodes"]}:
                raise ValueError("Unassigned demonstrations")
            for key in h5:
                if not isinstance(h5.get(key, getlink=True), h5py.HardLink):
                    raise ValueError("H5 links cannot expose external evidence")
                group = h5[key]
                if set(group) != fields or set(group.attrs) != ({"episode", "difficulty", "goal"} if historical else EPISODE_ATTRS):
                    raise ValueError("Unexpected public H5 fields or annotations")
                for field in group:
                    if (
                        not isinstance(group.get(field, getlink=True), h5py.HardLink)
                        or not isinstance(group[field], h5py.Dataset)
                        or group[field].is_virtual
                        or group[field].external
                        or len(group[field].attrs)
                    ):
                        raise ValueError(
                            "Only self-contained allowlisted H5 datasets are permitted"
                        )
                record = next(
                    row
                    for row in item["episodes"]
                    if key == f"episode_{row['episode']}"
                )
                count = record["frames"]
                if (
                    int(group.attrs["episode"]) != int(record["episode"])
                    or group.attrs["difficulty"] != record["difficulty"]
                    or not np.array_equal(group["step"][:], np.arange(count))
                    or group["tcp"].shape != (count, 6)
                ):
                    raise ValueError(
                        "Public episode identity, frame indices or native TCP shape changed"
                    )
                for camera in ("front", "wrist"):
                    rgb = group[camera + "_rgb"]
                    if (
                        rgb.shape[0] != count
                        or rgb.dtype != np.uint8
                        or len(rgb.shape) != 4
                        or rgb.shape[-1] != 3
                        or group[camera + "_K"].shape != (3, 3)
                        or group[camera + "_extrinsic"].shape != (count, 3, 4)
                    ):
                        raise ValueError("Invalid recorded image or calibration shape")
                    if manifest["depth"]:
                        depth = group[camera + "_depth"]
                        if depth.dtype != np.int16 or depth.shape != rgb.shape[:-1]:
                            raise ValueError(
                                "Depth must align exactly to the recorded RGB frames"
                            )
    return manifest
