"""Reference videos from a run's assigned training demonstrations only.

Existing extraction videos are reused. Missing previews are encoded from the
sanitized H5 camera arrays in the website cache, without touching the dataset or
running a simulator. No expert action or intermediate-success labels are inferred.
"""

import hashlib
import math
from pathlib import Path
import tempfile

import h5py
import numpy as np
from PIL import Image

from cam.io import read


def mosaic(frames, columns):
    height = max(frame.shape[0] for frame in frames)
    width = max(frame.shape[1] for frame in frames)
    image = np.zeros((height * math.ceil(len(frames) / columns), width * columns, 3), np.uint8)
    for i, frame in enumerate(frames):
        y, x = (i // columns) * height, (i % columns) * width
        image[y:y + frame.shape[0], x:x + frame.shape[1]] = frame
    return image


def references(root, plan, task, output, asset, sources):
    """Register exactly the task's prepared subset, not its source demo pool."""
    if plan["config"]["mode"] == "interaction":
        return []
    manifest = read(root / "inputs/dataset.json")
    item = manifest["tasks"].get(task)
    if not item:
        return []
    source = root / "tasks" / task / "demonstrations" / Path(item["file"]).name
    if not source.is_file():
        return []
    dataset = Path(plan["dataset_reference"]).parent.resolve()
    cameras = (plan.get("interfaces", {}).get(task) or {}).get("video") or ["front_rgb", "wrist_rgb"]
    fps = 15 if manifest["benchmark"] == "rmbench" else 20
    result = []
    with h5py.File(source, "r") as h5:
        for number, row in enumerate(item["episodes"], 1):
            group = h5[f"episode_{int(row['episode'])}"]
            names = [name for name in cameras if name in group]
            if not names:
                continue
            columns = 2 if len(names) > 2 else len(names)
            height = max(group[name].shape[1] for name in names) * math.ceil(len(names) / columns)
            width = max(group[name].shape[2] for name in names) * columns
            key = hashlib.sha256(f"demo/1:{item['sha256']}:{row['episode']}:{names}:{fps}".encode()).hexdigest()[:24]
            preview = output / "previews" / f"demo_{key}.jpg"
            if not preview.exists():
                image = Image.fromarray(mosaic([group[name][0] for name in names], columns))
                image.thumbnail((640, 400))
                preview.parent.mkdir(parents=True, exist_ok=True)
                temporary = preview.with_suffix(".partial")
                image.save(temporary, format="JPEG", quality=85)
                temporary.replace(preview)
            existing = (dataset / row["video"]).resolve() if row.get("video") else None
            if existing and (not existing.is_relative_to(dataset) or not existing.is_file()):
                existing = None
            video = asset(existing) if existing else f"demo-video/{key}.mp4"
            if not existing:
                sources[key] = {
                    "source": str(source), "episode": int(row["episode"]),
                    "cameras": names, "columns": columns, "fps": fps,
                    "frames": int(row["frames"]),
                }
            result.append({
                "number": number, "episode": int(row["episode"]), "frames": int(row["frames"]),
                "fps": fps, "goal": row.get("goal", ""), "demo_frames": row.get("demo_frames", 0),
                "video": video, "poster": preview.relative_to(output).as_posix(),
                "cameras": names, "columns": columns,
                "width": width, "height": height,
                "preview_on_demand": not bool(existing),
            })
    return result


def encode_video(spec, output, key):
    """Encode every stored RGB frame, preserving order; one CPU encoder thread."""
    import imageio.v2 as imageio

    target = Path(output) / "media/demos" / f"{key}.mp4"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".demo_", suffix=".mp4", dir=target.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with h5py.File(spec["source"], "r") as h5:
            group = h5[f"episode_{int(spec['episode'])}"]
            cameras = [group[name] for name in spec["cameras"]]
            if any(len(camera) != spec["frames"] for camera in cameras):
                raise ValueError("Recorded demonstration frame count changed")
            with imageio.get_writer(
                temporary, fps=spec["fps"], codec="libx264", quality=7,
                macro_block_size=2, ffmpeg_params=["-threads", "1", "-movflags", "+faststart"],
                ffmpeg_log_level="error",
            ) as writer:
                for start in range(0, spec["frames"], 16):
                    arrays = [camera[start:start + 16] for camera in cameras]
                    for frames in zip(*arrays):
                        writer.append_data(mosaic(frames, spec["columns"]))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
