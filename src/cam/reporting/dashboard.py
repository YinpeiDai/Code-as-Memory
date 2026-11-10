"""Build the directory-backed Robot Memory website; previews stay lazy."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

from cam.io import read, write
from cam.reporting.catalog import index_directory


def build_dashboard(data_root, output):
    root, output = Path(data_root).resolve(), Path(output).resolve()
    content = index_directory(root)
    output.mkdir(parents=True, exist_ok=True)
    artifacts, demos, posters, logs = {}, {}, {}, {}

    def asset(relative):
        if not relative:
            return None
        # Only names discovered by the index enter the server's allowlist.
        path = root / relative
        if not path.is_file():
            return None
        key = hashlib.sha256(str(path).encode()).hexdigest()[:24] + path.suffix
        artifacts[key] = str(path)
        return "artifact/" + key

    for run in content["runs"]:
        sources = [str(root / p) for p in run["logs"] if p.endswith(".jsonl")]
        logs[run["id"]] = sources
        run["transcript"] = f"agent-log/{run['id']}.json" if sources else None
        run["logs"] = [{"name": Path(p).name, "url": asset(p)} for p in run["logs"]]
        run["code"] = [{"name": Path(p).name, "url": asset(p)} for p in run["code"]]
        for name in ("prompt", "manifest"):
            run[name] = asset(run[name])
    for episode in content["episodes"]:
        video_path = episode["video"]
        if video_path:
            key = episode["id"]
            posters[key] = {"video": str(root / video_path)}
            episode["poster"] = f"preview/{key}.jpg"
        for name in ("result", "video", "observations", "actions", "decisions_log", "stderr"):
            episode[name] = asset(episode.get(name))
    for demo in content["demos"]:
        h5 = root / demo["h5"] if demo["h5"] else None
        key = demo["id"]
        if h5 and h5.is_file():
            cameras = demo["cameras"]
            spec = {"source": str(h5), "episode": demo["episode"], "frames": demo["frames"],
                    "cameras": cameras, "columns": min(len(cameras), 2),
                    "fps": 15 if demo["benchmark"] == "rmbench" else 20}
            demos[key] = spec
            posters[key] = spec
            demo["poster"] = f"preview/{key}.jpg"
            demo["video"] = asset(demo["video"]) or f"demo-video/{key}.mp4"
        demo["h5"] = asset(demo["h5"])
    content["downloads"] = {"results": asset("dashboard/results.json"),
                            "csv": asset("dashboard/results.csv"), "progress": asset("dashboard/PROGRESS.md")}
    source = Path(__file__).resolve().parents[1] / "web"
    for name in ("dashboard.html", "dashboard.css", "dashboard.js"):
        shutil.copy2(source / name, output / ("index.html" if name.endswith(".html") else name))
    write(output / "data.json", content)
    packed = output / ".data.json.gz.partial"
    packed.write_bytes(gzip.compress((output / "data.json").read_bytes(), mtime=0))
    packed.replace(output / ".data.json.gz")
    for name, data in (("artifacts", artifacts), ("demo_sources", demos), ("preview_sources", posters), ("log_sources", logs)):
        write(output / f".{name}.json", data)
    write(output / ".sources.json", {"data_root": str(root)})
    return {"site": str(output), "data_root": str(root), "runs": len(content["runs"]),
            "episodes": len(content["episodes"]), "demos": len(content["demos"])}


def preview(spec, output, key):
    """Read one frame only when a gallery card becomes visible."""
    from PIL import Image
    from cam.reporting.reference import mosaic
    import h5py
    import imageio.v2 as imageio

    path = Path(output) / "previews" / (key + ".jpg")
    if path.is_file():
        return path
    if spec.get("video"):
        with imageio.get_reader(spec["video"]) as reader:
            frame = reader.get_data(0)
    else:
        with h5py.File(spec["source"], "r") as h5:
            group = h5[f"episode_{int(spec['episode'])}"]
            cameras = [c for c in spec["cameras"] if c in group]
            if not cameras:
                raise ValueError("No recorded RGB cameras in this episode")
            frame = mosaic([group[c][0] for c in cameras], min(2, len(cameras)))
    image = Image.fromarray(frame)
    image.thumbnail((560, 320))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    image.save(temporary, format="JPEG", quality=80)
    temporary.replace(path)
    return path


def transcript(sources):
    """Readable agent messages and tool activity, with raw transcripts linked separately."""
    entries = []
    for source in sources:
        with Path(source).open(errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                role = event.get("type", "event")
                message = event.get("message") or {}
                content = message.get("content", []) if isinstance(message, dict) else []
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                for part in content:
                    kind = part.get("type")
                    if kind in ("text", "thinking"):
                        text = part.get("text", part.get("thinking", ""))
                    elif kind == "tool_use":
                        text = part.get("name", "tool") + "\n" + json.dumps(part.get("input", {}), indent=2)
                    elif kind == "tool_result":
                        value = part.get("content", [])
                        text = value if isinstance(value, str) else "\n".join(p.get("text", "") for p in value if p.get("type") == "text")
                    else:
                        continue
                    if text:
                        entries.append({"role": role, "kind": kind, "text": text[:16000], "truncated": len(text) > 16000})
                item = event.get("item", {})
                if event.get("type") == "item.completed" and item.get("text"):
                    entries.append({"role": "assistant", "kind": item.get("type"), "text": item["text"][:16000]})
    return {"entries": entries}
