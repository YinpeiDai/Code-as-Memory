"""Extract allowlisted RMBench demonstrations from the released ``demo_clean`` data.

Raw layout (``--raw-data data/raw/rmbench``), kept unmodified:

  hf/data/<task>/demo_clean.zip                       released archive (preserved original)
  demo_clean/<task>/aloha_agilex/seed.txt              position i -> seed of episode i
  demo_clean/<task>/aloha_agilex/data/episode_%07d.hdf5

Every extracted episode file must be byte-identical to its zip member (size and
CRC-32) and the zip must match the Hugging Face LFS sha256 pinned in the train
metadata. Camera frames are JPEG in the release (lossy at the source); they are
decoded once with the release decoder ``data/decode_image_bit.py`` (RGB, no extra
channel swap) and stored as uint8 without further loss. Other fields are copied
unchanged, except world-to-camera extrinsics, which are derived from the recorded
camera-to-world OpenGL poses exactly as the live ``extrinsic_cv`` sensor is.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import zipfile
import zlib

import h5py
import imageio.v2 as imageio
import numpy as np

from cam.benchmarks.base import source_hash
from cam.benchmarks.registry import get_adapter
from cam.data import demonstrations
from cam.data.demonstrations import declaration, random_select, selection_seed
from cam.io import read, sha256, write

from . import rmbench
from .rmbench import ARMS, CAMERAS, HEIGHT, WIDTH

EMBODIMENT_DIR = "aloha_agilex"
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
STATE = ("arm_joint_states", "ee_joint_states", "ee_poses")
EXCLUDED = (
    "action/* (verified equal to the next frame's state; the final row has no images)",
    "language_annotation.json (scripted-expert subtask labels)",
    "scene_info.json",
    "video/*.mp4 (re-encoded head-camera previews)",
    "instruction/*.json beyond the single task instruction",
    "depth, point clouds, segmentation and simulator state (absent from the release)",
)
ACTION_ALIGNMENT = (
    "No commands are recorded. Consecutive frames are scripted-expert states saved "
    "about every 15 simulator steps (uneven at motion boundaries); they are not CaM "
    "actions."
)
STEP_SEMANTICS = "Recording frame index, not a native take_action counter."


def fields():
    frame = {}
    for name, (_, group) in CAMERAS.items():
        frame[name + "_rgb"] = {
            "dtype": "uint8",
            "shape": [HEIGHT, WIDTH, 3],
            "description": f"{name} camera RGB decoded from the released JPEG ({group})",
        }
    for name, (_, group) in CAMERAS.items():
        frame[name + "_K"] = {
            "dtype": "float32",
            "shape": [3, 3],
            "description": f"{name} camera 3x3 OpenCV intrinsics in pixels ({group}/intrinsic_matrix)",
        }
        frame[name + "_extrinsic"] = {
            "dtype": "float32",
            "shape": [3, 4],
            "description": f"{name} camera 3x4 OpenCV world-to-camera extrinsics in metres, "
            f"inv(cam2world_gl @ diag(1,-1,-1,1))[:3] of {group}/extrinsics_matrix",
        }
    for arm in ARMS:
        frame[arm + "_ee_pose"] = {
            "dtype": "float32",
            "shape": [7],
            "description": f"{arm} end-effector pose, world x,y,z metres and quaternion w,x,y,z "
            f"(state/{arm}_ee_poses; the frame commanded by ee actions)",
        }
        frame[arm + "_joints"] = {
            "dtype": "float32",
            "shape": [6],
            "description": f"{arm} arm commanded joint targets in radians (state/{arm}_arm_joint_states)",
        }
        frame[arm + "_gripper"] = {
            "dtype": "float32",
            "shape": [],
            "description": f"{arm} commanded gripper opening, 0 closed to 1 open (state/{arm}_ee_joint_states)",
        }
    return declaration(frame)


def release_decoder(source):
    """The release's own image-bit decoder from the pinned benchmark snapshot."""
    path = Path(source) / "data/decode_image_bit.py"
    spec = importlib.util.spec_from_file_location("rmbench_decode_image_bit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_image_bit


def extrinsic(cam2world_gl):
    matrix = np.asarray(cam2world_gl, dtype=np.float64)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError("Expected 4x4 camera-to-world poses")
    return np.linalg.inv(matrix @ GL_TO_CV)[..., :3, :].astype(np.float32)


def crc32(path):
    value = 0
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value = zlib.crc32(chunk, value)
    return value & 0xFFFFFFFF


def sha256_file(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def release_provenance(task):
    """Release identity pinned with the training seeds (scripts/rmbench_train_metadata.py)."""
    return read(rmbench.METADATA / "train" / f"record_dataset_{task}_metadata.json")["provenance"]


def task_paths(raw, task):
    return (
        raw / "hf/data" / task / "demo_clean.zip",
        raw / "demo_clean" / task / EMBODIMENT_DIR,
    )


def member(task, relative):
    return f"{task}/{EMBODIMENT_DIR}/{relative}"


def verify_release(raw, task, records, provenance):
    """Check the preserved zip, its seed list and the extracted directory."""
    archive, directory = task_paths(raw, task)
    digest = sha256_file(archive)
    if digest != provenance["zip_sha256"]:
        raise ValueError(f"{task}: zip sha256 differs from the pinned release")
    with zipfile.ZipFile(archive) as z:
        seeds = [int(s) for s in z.read(member(task, "seed.txt")).decode().split()]
        infos = {
            info.filename: info
            for info in z.infolist()
            if info.filename.startswith(member(task, "data/")) and info.filename.endswith(".hdf5")
        }
    if (directory / "seed.txt").read_bytes().split() != [str(s).encode() for s in seeds]:
        raise ValueError(f"{task}: extracted seed.txt differs from the zip")
    if [int(r["seed"]) for r in records] != seeds or [int(r["episode"]) for r in records] != list(range(len(seeds))):
        raise ValueError(f"{task}: train metadata disagrees with the zip seed order")
    expected = {member(task, f"data/episode_{i:07d}.hdf5") for i in range(len(seeds))}
    if set(infos) != expected:
        raise ValueError(f"{task}: zip episode files differ from the seed list")
    audit = []
    for record in records:
        info = infos[member(task, f"data/episode_{int(record['episode']):07d}.hdf5")]
        path = directory / "data" / f"episode_{int(record['episode']):07d}.hdf5"
        if path.is_symlink() or path.stat().st_size != info.file_size:
            raise ValueError(f"{task}: {path.name} differs from the zip member")
        audit.append({**record, "zip_member": info.filename, "size": info.file_size, "crc32": info.CRC})
    return digest, audit


def read_episode(path, goal):
    """Validate one released episode; return its length and non-image fields."""
    with h5py.File(path, "r") as h5:
        length = len(h5["state/left_ee_poses"])
        if json.loads(h5["instructions"][()]) != [goal]:
            raise ValueError(f"{path}: instruction differs from the task instruction")
        if int(h5["additional_info/frequency"][()]) != 15:
            raise ValueError(f"{path}: unexpected save interval")
        state = {}
        for arm in ARMS:
            for key in STATE:
                name = f"{arm}_{key}"
                value, nxt = h5["state/" + name][:], h5["action/" + name][:]
                if len(value) != length or len(nxt) != length or not np.array_equal(nxt[:-1], value[1:]):
                    raise ValueError(f"{path}: action/{name} is not the next state")
                if value.dtype != np.float32:
                    raise ValueError(f"{path}: state/{name} is not float32")
                state[name] = value
        row = {}
        for arm in ARMS:
            row[arm + "_ee_pose"] = state[arm + "_ee_poses"]
            row[arm + "_joints"] = state[arm + "_arm_joint_states"]
            gripper = state[arm + "_ee_joint_states"]
            if gripper.shape != (length, 1):
                raise ValueError(f"{path}: unexpected gripper shape")
            row[arm + "_gripper"] = gripper[:, 0]
        for name, (_, group) in CAMERAS.items():
            vision = h5["vision/" + group]
            if list(vision["shape"][()]) != [HEIGHT, WIDTH, 3] or len(vision["colors"]) != length:
                raise ValueError(f"{path}: unexpected {group} frames")
            intrinsic = vision["intrinsic_matrix"][:]
            if intrinsic.dtype != np.float32 or intrinsic.shape != (length, 3, 3):
                raise ValueError(f"{path}: unexpected {group} intrinsics")
            row[name + "_K"] = intrinsic
            row[name + "_extrinsic"] = extrinsic(vision["extrinsics_matrix"][:])
        if set(h5["vision"]) != {group for _, group in CAMERAS.values()}:
            raise ValueError(f"{path}: unexpected camera groups")
    return length, row


def images(src, group, start, stop, decode):
    frames = decode(src[f"vision/{group}/colors"][start:stop])
    if frames.dtype != np.uint8 or frames.shape != (stop - start, HEIGHT, WIDTH, 3):
        raise ValueError(f"{src.filename}: undecodable {group} frames")
    return frames


def export_task(job):
    raw, output, source, task, count, videos = job
    raw, output = Path(raw), Path(output)
    adapter = get_adapter("rmbench", source)
    records = adapter.records(task, "train")
    try:
        test = adapter.records(task, "test")
    except FileNotFoundError:
        test = None  # checked again by validate_dataset and cam prepare once pinned
    provenance = release_provenance(task)
    test_seeds = {int(r["seed"]) for r in test or ()}
    if test_seeds & {int(r["seed"]) for r in records}:
        raise ValueError("Training/test seed overlap")
    digest, audit = verify_release(raw, task, records, provenance)
    selected = random_select(records, count, "rmbench", task)
    declared = fields()
    decode = release_decoder(adapter.source)
    goal = adapter.goal(task)
    archive, directory = task_paths(raw, task)
    destination = output / "h5" / f"record_dataset_{task}.h5"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".h5.partial")
    episodes = []
    origin = f"{provenance['repo']}@{provenance['revision']}:data/{task}/demo_clean.zip"
    with demonstrations.create(partial, "rmbench", task, declared, False, origin) as h5:
        for record in selected:
            episode = int(record["episode"])
            path = directory / "data" / f"episode_{episode:07d}.hdf5"
            crc = next(row["crc32"] for row in audit if int(row["episode"]) == episode)
            before = path.stat()
            if crc32(path) != crc:
                raise ValueError(f"{task}: {path.name} CRC differs from the zip member")
            length, row = read_episode(path, goal)
            writer = demonstrations.Episode(
                h5,
                declared,
                episode,
                length,
                {
                    "goal": goal,
                    "difficulty": None,
                    "action_alignment": ACTION_ALIGNMENT,
                    "step_semantics": STEP_SEMANTICS,
                },
            )
            movie = output / "rollouts" / task / f"episode_{episode}.mp4"
            video = None
            try:
                if videos:
                    movie.parent.mkdir(parents=True, exist_ok=True)
                    video = imageio.get_writer(
                        movie,
                        fps=15,
                        codec="libx264",
                        quality=7,
                        ffmpeg_params=["-threads", "1"],
                        ffmpeg_log_level="error",
                    )
                with h5py.File(path, "r") as src:
                    for start in range(0, length, 64):
                        stop = min(start + 64, length)
                        chunk = {"is_demo": np.zeros(stop - start, dtype=bool)}
                        for name, value in row.items():
                            chunk[name] = value[start:stop]
                        for name, (_, group) in CAMERAS.items():
                            chunk[name + "_rgb"] = images(src, group, start, stop, decode)
                        writer.append(chunk)
                        if video:
                            for i in range(stop - start):
                                top = np.hstack([chunk["head_rgb"][i], chunk["front_rgb"][i]])
                                bottom = np.hstack([chunk["left_wrist_rgb"][i], chunk["right_wrist_rgb"][i]])
                                video.append_data(np.vstack([top, bottom]))
            finally:
                if video:
                    video.close()
            writer.close()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Raw file changed during extraction")
            episodes.append(
                {
                    "episode": episode,
                    "seed": int(record["seed"]),
                    "frames": length,
                    "demo_frames": 0,
                    "goal": goal,
                    "video": movie.relative_to(output).as_posix() if videos else None,
                }
            )
    partial.replace(destination)
    item = {
        "task": task,
        "file": destination.relative_to(output).as_posix(),
        "sha256": sha256(destination),
        "episodes": episodes,
        "source": {
            "zip": str(archive),
            "zip_sha256": digest,
            "hf_lfs_sha256": provenance["zip_sha256"],
            "zip_size": archive.stat().st_size,
            "directory": str(directory),
            "selection_seed": selection_seed("rmbench", task),
        },
        "test_overlap_checked": test is not None,
    }
    write(output / "audit" / (task + ".json"), {"all_training_episodes": audit, **item})
    return task, item


def extract(
    raw_data,
    output,
    benchmark="rmbench",
    count=25,
    tasks=None,
    depth=False,
    videos=True,
    workers=2,
    benchmark_source=None,
    resume=False,
):
    if depth:
        from .rmbench_rgbd_data import extract as regenerate

        return regenerate(raw_data, output, benchmark, count, tasks, depth, videos,
                          workers, benchmark_source, resume)
    raw_data, output = Path(raw_data).resolve(), Path(output).resolve()
    adapter = get_adapter(benchmark, benchmark_source)
    tasks = tasks or adapter.tasks()
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(adapter.tasks()) or workers < 1:
        raise ValueError("Choose distinct RMBench tasks with pinned metadata and positive workers")
    count = "all" if count == "all" else int(count)
    release = {task: release_provenance(task) for task in tasks}
    revisions = {(p["repo"], p["revision"]) for p in release.values()}
    if len(revisions) != 1:
        raise ValueError("Tasks come from different releases")
    repo, revision = revisions.pop()
    output.mkdir(parents=True, exist_ok=resume)
    manifest = {
        "schema": "cam-dataset/1",
        "demonstration_schema": demonstrations.SCHEMA,
        "created": time.time(),
        "benchmark": benchmark,
        "benchmark_hash": source_hash(adapter),
        "raw_data": str(raw_data),
        "depth": False,
        "fields": fields(),
        "selection": {
            "count": count,
            "rule": "uniform random without replacement (cam.data.demonstrations.random_select), "
            "NumPy default_rng(int(sha256('cam/rmbench/<task>')[:8], 16)), sorted by episode",
        },
        "requested_tasks": tasks,
        "videos": videos,
        "release": {
            "repo": repo,
            "repo_type": "dataset",
            "revision": revision,
            "task_config": "demo_clean",
            "embodiment": EMBODIMENT_DIR,
            "image_encoding": "JPEG in the release (lossy at the source); decoded once with "
            "data/decode_image_bit.py to RGB uint8, no further loss",
            "excluded": list(EXCLUDED),
        },
        "tasks": {},
    }
    if resume:
        previous = read(
            output / "manifest.json" if (output / "manifest.json").exists() else output / "manifest.partial.json"
        )
        for key in ("benchmark", "benchmark_hash", "raw_data", "depth", "fields", "selection", "requested_tasks", "videos", "release"):
            if manifest[key] != previous[key]:
                raise ValueError(f"Extraction resume configuration changed: {key}")
        manifest["created"] = previous["created"]
    jobs = []
    for task in tasks:
        audit = output / "audit" / (task + ".json")
        if resume and audit.exists():
            item = read(audit)
            if sha256(output / item["file"]) != item["sha256"]:
                raise ValueError(f"Completed extraction changed: {task}")
            manifest["tasks"][task] = {k: v for k, v in item.items() if k != "all_training_episodes"}
            continue
        if resume:
            for path in [
                output / "h5" / f"record_dataset_{task}.h5",
                output / "h5" / f"record_dataset_{task}.h5.partial",
                output / "rollouts" / task,
            ]:
                if path.exists():
                    target = output / "interrupted" / str(time.time_ns()) / path.name
                    target.parent.mkdir(parents=True)
                    shutil.move(path, target)
        jobs.append((raw_data, output, adapter.source, task, count, videos))
    write(output / "manifest.partial.json", manifest)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task, item in pool.map(export_task, jobs):
            manifest["tasks"][task] = item
            write(output / "manifest.partial.json", manifest)
    manifest["tasks"] = {task: manifest["tasks"][task] for task in tasks}
    write(output / "manifest.json", manifest)
    (output / "manifest.partial.json").unlink(missing_ok=True)
    return validate_dataset(output)


def validate_dataset(root, verify_hashes=True):
    """Shared v2 validation plus RMBench identity checks against pinned metadata."""
    if read(Path(root) / "manifest.json").get("depth"):
        from .rmbench_rgbd_data import validate_dataset as validate_rgbd

        return validate_rgbd(root, verify_hashes)
    manifest = demonstrations.validate_dataset(root, verify_hashes)
    if manifest["benchmark"] != "rmbench" or manifest["depth"] or manifest["fields"] != fields():
        raise ValueError("Not an RGB-only RMBench extraction with the declared fields")
    metadata = rmbench.METADATA
    for task, item in manifest["tasks"].items():
        train = read(metadata / "train" / f"record_dataset_{task}_metadata.json")["records"]
        seeds = {int(r["episode"]): int(r["seed"]) for r in train}
        test_path = metadata / "test" / f"record_dataset_{task}_metadata.json"
        test = {int(r["seed"]) for r in read(test_path)["records"]} if test_path.exists() else set()
        for row in item["episodes"]:
            if seeds.get(int(row["episode"])) != int(row["seed"]) or int(row["seed"]) in test:
                raise ValueError(f"{task}: episode {row['episode']} identity or test overlap")
            if row["demo_frames"] != 0:
                raise ValueError("RMBench has no task-provided video frames")
    return manifest
