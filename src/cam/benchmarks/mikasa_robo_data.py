"""Extract assigned MIKASA-Robo-VLA RLDS demonstrations into ``cam-demonstrations/2``.

Raw input is the lossless RLDS release (``mikasa-robo/mikasa-robo-vla-rlds`` at the
revision pinned in ``metadata/mikasa_robo/release.json``) laid out as
``<raw>/<task_folder>/1.0.0/``. Every task file is checked against the release
(LFS SHA-256 for shards, git blob SHA-1 for JSON) while it is read. Every released
episode is identified by ``episode_metadata/file_path`` ``.../train_data_{k}.npz``
and checked against the pinned training identities (seed, length, start/end flags,
constant instruction). Selected episodes copy only camera PNGs (decoded
losslessly), proprio (split into TCP pose and gripper opening) and the recorded
command; reward, discount, terminal flags and source paths are never written.
Runs in the CaM environment; no simulator imports.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
from pathlib import Path
import re
import shutil
import struct
import time

import numpy as np

from cam.benchmarks.base import source_hash
from cam.benchmarks.mikasa_robo import METADATA, IMAGE, dataset_name
from cam.benchmarks.registry import get_adapter
from cam.data import demonstrations
from cam.data.tfrecord import example
from cam.io import read, sha256, write

VERSION = "1.0.0"
FEATURES_SHA1 = "66e9dcda0d7cb89bfed3377ccf1ea01c5a5122be"  # identical for all 90 tasks
#: Every key of a released tf.train.Example; anything else is rejected.
RLDS_KEYS = {
    "steps/observation/image",
    "steps/observation/wrist_image",
    "steps/observation/proprio",
    "steps/action",
    "steps/language_instruction",
    "steps/is_first",
    "steps/is_last",
    "steps/is_terminal",
    "steps/reward",
    "steps/discount",
    "episode_metadata/file_path",
}
#: Read for identity/structure checks only; never written to the public H5.
EXCLUDED = (
    "steps/reward",
    "steps/discount",
    "steps/is_terminal",
    "steps/is_first",
    "steps/is_last",
    "episode_metadata/file_path",
)


def spec(dtype, shape, description):
    return {"dtype": dtype, "shape": list(shape), "description": description}


FIELDS = {
    "base_rgb": spec("uint8", IMAGE, "fixed base (scene) camera RGB; RLDS observation/image"),
    "wrist_rgb": spec("uint8", IMAGE, "wrist (hand) camera RGB; RLDS observation/wrist_image"),
    "tcp": spec(
        "float32",
        (6,),
        "measured TCP XYZ metres (scene frame) and roll/pitch/yaw radians; proprio[0:6]",
    ),
    "gripper_opening": spec(
        "float32", (1,), "sum of both finger positions, metres (0 closed, 0.08 open); proprio[6]"
    ),
    "action": spec(
        "float32",
        (7,),
        "recorded expert command (dx, dy, dz, drx, dry, drz, gripper), normalized "
        "pd_ee_delta_pose, issued after observing this frame",
    ),
}
DECLARED = demonstrations.declaration(FIELDS)
SELECTION = {
    "rule": "uniform random without replacement over released episode IDs "
    "(cam.data.demonstrations.random_select), sorted by episode ID",
    "seed_rule": "int(sha256('cam/mikasa_robo/<task>').hexdigest()[:8], 16)",
}
STEP_SEMANTICS = (
    "Native control step: frame t is the observation after t commands; frame 0 is the "
    "reset observation."
)
ALIGNMENT = (
    "action[t] was issued after observing frame t and produced frame t+1; the "
    "observation after the final command is not recorded."
)
NOOP_NOTE = (
    " In this task the native wrapper executes a zero command during the initial cue "
    "phase; recorded commands there were not executed."
)
FILE_PATH = re.compile(r"(?:^|/)(?P<task>[A-Za-z0-9_-]+)/train_data_(?P<k>\d+)\.npz$")


def release():
    return read(METADATA / "release.json")


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def task_files(task, files):
    prefix = f"{dataset_name(task)}/{VERSION}/"
    return {path[len(prefix) :]: value for path, value in files.items() if path.startswith(prefix)}


def shards(expected):
    return sorted(name for name in expected if ".tfrecord-" in name)


def tfrecords(path, digest):
    """TFRecord payloads of one shard, hashing every byte read (single pass)."""
    with open(path, "rb") as stream:
        while True:
            header = stream.read(12)
            digest.update(header)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header: {path}")
            (length,) = struct.unpack("<Q", header[:8])
            payload = stream.read(length)
            footer = stream.read(4)
            digest.update(payload)
            digest.update(footer)
            if len(payload) != length or len(footer) != 4:
                raise ValueError(f"Truncated TFRecord payload: {path}")
            yield payload


def png(data):
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != IMAGE[1::-1]:
            raise ValueError("Expected a 128x128 8-bit RGB PNG")
        value = np.asarray(image)
    if value.dtype != np.uint8 or value.shape != IMAGE:
        raise ValueError("Decoded camera image changed shape or dtype")
    return value


def parse(payload, task, train):
    """Decode one released episode's identity and arrays; validate its structure."""
    ex = example(payload)
    if set(ex) != RLDS_KEYS:
        raise ValueError(f"Unexpected RLDS features: {sorted(set(ex) ^ RLDS_KEYS)}")
    (path,) = ex["episode_metadata/file_path"]
    match = FILE_PATH.search(path.decode())
    if not match or match["task"] != task:
        raise ValueError(f"Unrecognized released episode path: {path!r}")
    episode = int(match["k"])
    if episode not in train:
        raise ValueError(f"Released episode {episode} is not a pinned training identity")
    images, wrist = ex["steps/observation/image"], ex["steps/observation/wrist_image"]
    length = len(images)
    language = {s.decode() for s in ex["steps/language_instruction"]}
    flags = {k: np.asarray(ex[f"steps/{k}"]) for k in ("is_first", "is_last")}
    proprio = np.asarray(ex["steps/observation/proprio"], dtype="<f4")
    command = np.asarray(ex["steps/action"], dtype="<f4")
    if (
        not length
        or len(wrist) != length
        or len(ex["steps/language_instruction"]) != length
        or proprio.size != 7 * length
        or command.size != 7 * length
        or any(len(v) != length for v in flags.values())
        or len(ex["steps/reward"]) != length
    ):
        raise ValueError(f"Inconsistent step counts: {task}/{episode}")
    if len(language) != 1 or not next(iter(language)).strip():
        raise ValueError(f"Instruction is not constant: {task}/{episode}")
    if not np.array_equal(flags["is_first"], np.arange(length) == 0) or not np.array_equal(
        flags["is_last"], np.arange(length) == length - 1
    ):
        raise ValueError(f"Episode boundaries are not one released episode: {task}/{episode}")
    proprio, command = proprio.reshape(length, 7), command.reshape(length, 7)
    if not (np.isfinite(proprio).all() and np.isfinite(command).all()):
        raise ValueError(f"Nonfinite proprio or command: {task}/{episode}")
    return {
        "episode": episode,
        "frames": length,
        "goal": language.pop(),
        "images": images,
        "wrist": wrist,
        "proprio": proprio,
        "action": command,
    }


def write_episode(h5, item, record, noop, video_path):
    length = item["frames"]
    episode = demonstrations.Episode(
        h5,
        DECLARED,
        record["episode"],
        length,
        {
            "goal": item["goal"],
            "difficulty": None,
            "action_alignment": ALIGNMENT + (NOOP_NOTE if noop else ""),
            "step_semantics": STEP_SEMANTICS,
        },
    )
    video = None
    try:
        if video_path is not None:
            import imageio.v2 as imageio

            video_path.parent.mkdir(parents=True, exist_ok=True)
            video = imageio.get_writer(
                video_path,
                fps=20,
                codec="libx264",
                quality=7,
                ffmpeg_params=["-threads", "1"],
                ffmpeg_log_level="error",
            )
        for start in range(0, length, 64):
            end = min(start + 64, length)
            base = np.stack([png(b) for b in item["images"][start:end]])
            wrist = np.stack([png(b) for b in item["wrist"][start:end]])
            episode.append(
                {
                    "base_rgb": base,
                    "wrist_rgb": wrist,
                    "tcp": np.ascontiguousarray(item["proprio"][start:end, :6]),
                    "gripper_opening": np.ascontiguousarray(item["proprio"][start:end, 6:]),
                    "action": np.ascontiguousarray(item["action"][start:end]),
                    "is_demo": np.zeros(end - start, dtype=bool),
                }
            )
            if video:
                for a, b in zip(base, wrist):
                    video.append_data(np.hstack([a, b]))
    finally:
        if video:
            video.close()
    episode.close()


def verify_json(folder, expected, name):
    path = folder / name
    if path.stat().st_size != expected[name]["size"] or git_blob_sha1(path) != expected[name]["git_sha1"]:
        raise ValueError(f"{path} differs from the pinned release")
    return read(path)


def export_task(job):
    raw, output, source, benchmark, task, count, videos = job
    raw, output = Path(raw), Path(output)
    adapter = get_adapter(benchmark, source)
    pinned = adapter.metadata(task, "train")
    train = {int(r["episode"]): r for r in adapter.records(task, "train")}
    test_seeds = {int(r["seed"]) for r in adapter.records(task, "test")}
    if test_seeds & {int(r["seed"]) for r in train.values()}:
        raise ValueError(f"Training/test seed overlap: {task}")
    info = release()
    expected = task_files(task, info["files"])
    folder = raw / dataset_name(task) / VERSION
    present = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
    if present != set(expected):
        raise ValueError(
            f"Raw release files differ for {task}: missing {sorted(set(expected) - present)}, "
            f"extra {sorted(present - set(expected))}"
        )
    if expected["features.json"]["git_sha1"] != FEATURES_SHA1:
        raise ValueError("Release feature schema differs from the supported RLDS layout")
    verify_json(folder, expected, "features.json")
    native = verify_json(folder, expected, "metadata.json")
    dataset_info = verify_json(folder, expected, "dataset_info.json")
    seeds = [int(s) for s in native["episode_seeds"]]
    if native["env_id"] != task or seeds != [int(train[k]["seed"]) for k in range(len(train))]:
        raise ValueError(f"Release metadata disagrees with pinned identities: {task}")
    (split,) = dataset_info["splits"]
    names = shards(expected)
    counts = [int(n) for n in split["shardLengths"]]
    if split["name"] != "train" or len(counts) != len(names) or sum(counts) != len(seeds):
        raise ValueError(f"Release shard layout disagrees with metadata: {task}")
    selected = {
        int(r["episode"]): r
        for r in demonstrations.random_select(list(train.values()), count, benchmark, task)
    }
    noop = bool(pinned.get("noop_cue_phase", False))
    stats = {name: (folder / name).stat() for name in expected}
    destination = output / "h5" / f"record_dataset_{task}.h5"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".h5.partial")
    audit, episodes, seen = [], {}, set()
    source_label = f"{info['repo']}@{info['revision']}/{dataset_name(task)}/{VERSION}"
    with demonstrations.create(partial, benchmark, task, DECLARED, False, source_label) as h5:
        for name, count_in_shard in zip(names, counts):
            digest, index = hashlib.sha256(), 0
            for index, payload in enumerate(tfrecords(folder / name, digest), start=1):
                item = parse(payload, task, train)
                k = item["episode"]
                if k in seen:
                    raise ValueError(f"Duplicate released episode {task}/{k}")
                seen.add(k)
                if item["frames"] != int(native["episode_lengths"][k]):
                    raise ValueError(f"Episode length disagrees with release metadata: {task}/{k}")
                audit.append(
                    {**train[k], "frames": item["frames"], "shard": name, "record": index - 1}
                )
                if k in selected:
                    movie = output / "rollouts" / task / f"episode_{k}.mp4" if videos else None
                    write_episode(h5, item, selected[k], noop, movie)
                    episodes[k] = {
                        **selected[k],
                        "frames": item["frames"],
                        "demo_frames": 0,
                        "goal": item["goal"],
                        "video": movie.relative_to(output).as_posix() if videos else None,
                    }
            if index != count_in_shard:
                raise ValueError(f"Shard {name} has {index} episodes, expected {count_in_shard}")
            if digest.hexdigest() != expected[name]["sha256"]:
                raise ValueError(f"Shard SHA-256 differs from the release LFS object: {task}/{name}")
    if seen != set(train) or set(episodes) != set(selected):
        raise ValueError(f"Released episodes differ from pinned identities: {task}")
    for name, before in stats.items():
        after = (folder / name).stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Raw file changed during extraction: {task}/{name}")
    partial.replace(destination)
    item = {
        "task": task,
        "file": destination.relative_to(output).as_posix(),
        "sha256": sha256(destination),
        "episodes": [episodes[k] for k in sorted(episodes)],
        "source": {
            "repo": info["repo"],
            "revision": info["revision"],
            "folder": f"{dataset_name(task)}/{VERSION}",
            "released_episodes": len(seen),
            "files": expected,
            "verified": "LFS SHA-256 for shards, git blob SHA-1 for JSON; all episodes parsed",
        },
        "source_files": {
            name: {"size": s.st_size, "mtime_ns": s.st_mtime_ns} for name, s in stats.items()
        },
    }
    write(
        output / "audit" / f"{task}.json",
        {"all_training_episodes": sorted(audit, key=lambda r: r["episode"]), **item},
    )
    return task, item


def extract(
    raw_data,
    output,
    benchmark="mikasa_robo",
    count=25,
    tasks=None,
    depth=False,
    videos=True,
    workers=2,
    benchmark_source=None,
    resume=False,
):
    raw_data, output = Path(raw_data).resolve(), Path(output).resolve()
    adapter = get_adapter(benchmark, benchmark_source)
    if depth:
        raise ValueError("The pinned MIKASA-Robo RLDS release has no depth; pass --rgb-only")
    tasks = tasks or adapter.tasks()
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or set(tasks) - set(adapter.tasks())
        or workers < 1
    ):
        raise ValueError("Choose distinct native tasks and positive extraction workers")
    count = "all" if count == "all" else int(count)
    info = release()
    output.mkdir(parents=True, exist_ok=resume)
    manifest = {
        "schema": "cam-dataset/1",
        "demonstration_schema": demonstrations.SCHEMA,
        "created": time.time(),
        "benchmark": benchmark,
        "benchmark_hash": source_hash(adapter),
        "raw_data": str(raw_data),
        "depth": False,
        "fields": DECLARED,
        "selection": {"count": count, **SELECTION},
        "requested_tasks": list(tasks),
        "videos": videos,
        "release": {
            "repo": info["repo"],
            "repo_type": info["repo_type"],
            "revision": info["revision"],
            "listing": "src/cam/benchmarks/metadata/mikasa_robo/release.json",
            "listing_sha256": sha256(METADATA / "release.json"),
            "excluded_rlds_features": list(EXCLUDED),
        },
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
            "fields",
            "selection",
            "requested_tasks",
            "videos",
            "release",
        ):
            if manifest[key] != previous[key]:
                raise ValueError(f"Extraction resume configuration changed: {key}")
        manifest["created"] = previous["created"]
    jobs = []
    for task in tasks:
        audit = output / "audit" / f"{task}.json"
        if resume and audit.exists():
            item = read(audit)
            folder = raw_data / item["source"]["folder"]
            for name, stat in item["source_files"].items():
                now = (folder / name).stat()
                if (now.st_size, now.st_mtime_ns) != (stat["size"], stat["mtime_ns"]):
                    raise ValueError(f"Raw data changed since extraction: {task}/{name}")
            if sha256(output / item["file"]) != item["sha256"]:
                raise ValueError(f"Completed extraction changed: {task}")
            manifest["tasks"][task] = {
                k: v for k, v in item.items() if k != "all_training_episodes"
            }
            continue
        if resume:
            # Preserve interrupted files and videos rather than overwriting them.
            paths = [
                output / "h5" / f"record_dataset_{task}{suffix}"
                for suffix in (".h5", ".h5.partial")
            ] + [output / "rollouts" / task]
            for path in paths:
                if path.exists():
                    target = output / "interrupted" / str(time.time_ns()) / path.name
                    target.parent.mkdir(parents=True)
                    shutil.move(path, target)
        jobs.append((raw_data, output, adapter.source, benchmark, task, count, videos))
    write(output / "manifest.partial.json", manifest)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task, item in pool.map(export_task, jobs):
            manifest["tasks"][task] = item
            write(output / "manifest.partial.json", manifest)
    manifest["tasks"] = {task: manifest["tasks"][task] for task in tasks}
    write(output / "manifest.json", manifest)
    (output / "manifest.partial.json").unlink(missing_ok=True)
    validate_dataset(output)
    return manifest


def validate_dataset(root, verify_hashes=True):
    manifest = demonstrations.validate_dataset(root, verify_hashes)
    if manifest["benchmark"] != "mikasa_robo" or manifest["depth"] or manifest["fields"] != DECLARED:
        raise ValueError("Not a MIKASA-Robo RGB extraction with the declared fields")
    for task, item in manifest["tasks"].items():
        for row in item["episodes"]:
            if int(row["demo_frames"]) != 0:
                raise ValueError(f"MIKASA-Robo recordings have no task-provided video: {task}")
    return manifest
