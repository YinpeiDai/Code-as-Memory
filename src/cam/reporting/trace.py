"""Frame-aligned rollout traces for the website's annotated rollout explorer.

Built host-side from a rollout's own records — ``observations.h5`` (allowlisted
sensors the policy received), ``actions.jsonl`` and ``decisions.jsonl`` — so every
overlay shows only what the policy saw and sent. Frame 0 is the reset frame and
frame i (i ≥ 1) is the observation produced by action i−1, exactly as captured.
No simulator state is read. Results are cached by source modification times.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "cam-trace/2"
POINT_SENSORS = (("tcp", "TCP"), ("left_ee_pose", "left EE"), ("right_ee_pose", "right EE"))
STATE_LIMIT = 16  # sensors with at most this many values are shown as robot state


def _round(values, digits=4):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def _project(points, K, E):
    """World points (N,3) → pixels (N,2) with OpenCV [R|t] world→camera; None if behind."""
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera = np.einsum("nij,nj->ni", E, homogeneous)
    pixels = np.einsum("nij,nj->ni", K, camera)
    out = []
    for p, z in zip(pixels, camera[:, 2]):
        out.append(None if z <= 1e-6 or not np.isfinite(p).all() else [round(float(p[0] / p[2]), 1), round(float(p[1] / p[2]), 1)])
    return out


def _changes(before, after, limit=160):
    rows = []
    for key in after:
        if before.get(key) != after.get(key):
            text = json.dumps(after[key])
            rows.append([key, text if len(text) <= limit else text[:limit] + "…"])
    return rows


def build_trace(folder, interface=None):
    import h5py

    folder = Path(folder)
    video = list((interface or {}).get("video") or ["front_rgb", "wrist_rgb"])
    action = (interface or {}).get("action") or {}
    with h5py.File(folder / "observations.h5", "r") as h5:
        n = len(h5["step"])
        step = h5["step"][:].astype(int)
        phase = h5["phase"][:].astype(int)
        tiles, x = [], 0
        for name in video:
            if name in h5:
                height, width = h5[name].shape[1:3]
                tiles.append({"name": name, "label": name.removesuffix("_rgb").replace("_", " "), "x": x, "width": int(width), "height": int(height)})
                x += int(width)
        state = {
            name: _round(h5[name][:].reshape(n, -1))
            for name in sorted(h5)
            if name not in ("step", "phase")
            and not name.endswith(("_rgb", "_depth", "_K", "_extrinsic"))
            and int(np.prod(h5[name].shape[1:])) <= STATE_LIMIT
        }
        points = {}
        for sensor, label in POINT_SENSORS:
            if sensor not in h5:
                continue
            xyz = h5[sensor][:].reshape(n, -1)[:, :3].astype(float)
            for tile in tiles:
                camera = tile["name"].removesuffix("_rgb")
                if f"{camera}_K" in h5 and f"{camera}_extrinsic" in h5:
                    K = h5[f"{camera}_K"][:].reshape(n, 3, 3).astype(float)
                    E = h5[f"{camera}_extrinsic"][:].reshape(n, 3, 4).astype(float)
                    points.setdefault(tile["name"], []).append({"label": label, "uv": _project(xyz, K, E)})
    actions = []
    if (folder / "actions.jsonl").exists():
        actions = [_round(json.loads(line)["action"]) for line in open(folder / "actions.jsonl")]
    decisions = []
    if (folder / "decisions.jsonl").exists():
        cursor = 0
        for line in open(folder / "decisions.jsonl"):
            message = json.loads(line)
            if message.get("kind") == "decision":
                row = message["row"]
                frame = next((j for j in range(cursor, n) if phase[j] == 1 and step[j] == row["step"]), cursor)
                cursor = frame
                decisions.append({
                    "frame": int(frame), "step": row["step"],
                    "tool": row["act"]["tool"], "args": row["act"]["args"],
                    "memory": _changes(row.get("memory_in") or {}, row.get("memory_written") or {}),
                    "labels": {
                        key: str(value)[:240]
                        for key, value in (row.get("memory_written") or {}).items()
                        if key in ("phase", "stage", "state", "mode", "note", "reason", "target")
                        and isinstance(value, (str, int, float, bool))
                    },
                })
            elif message.get("kind") == "outcome" and decisions:
                decisions[-1]["outcome"] = message.get("output", "")[:1200]
    result = json.loads((folder / "result.json").read_text()) if (folder / "result.json").exists() else {}
    decision_of = np.full(n, -1)
    for k, d in enumerate(decisions):
        decision_of[d["frame"]:] = k
    return {
        "schema": SCHEMA,
        "fps": 20,
        "frames": n,
        "tiles": tiles,
        "video_width": x + (x % 2),
        "video_height": max((t["height"] for t in tiles), default=0),
        "step": step.tolist(),
        "phase": phase.tolist(),
        "decision": decision_of.tolist(),
        "action_names": action.get("names") or [],
        "actions": actions,
        "state": state,
        "points": points,
        "decisions": decisions,
        "result": {k: result.get(k) for k in ("task", "episode", "split", "status", "success", "steps", "step_calls", "decisions", "goal", "seconds", "error", "difficulty", "native_metrics")},
    }


def cached_trace(folder, output, interface=None):
    """Write (or reuse) the trace JSON for one rollout; returns its path or None."""
    folder = Path(folder)
    sources = [folder / name for name in ("observations.h5", "actions.jsonl", "decisions.jsonl", "result.json")]
    if not sources[0].is_file() or not sources[3].is_file():
        return None
    stamp = "|".join(f"{p}:{p.stat().st_mtime_ns}" for p in sources if p.exists())
    key = hashlib.sha256((stamp + SCHEMA + json.dumps(interface or {}, sort_keys=True)).encode()).hexdigest()[:24]
    path = Path(output) / "traces" / f"{key}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            value = build_trace(folder, interface)
        except Exception as exc:  # a damaged rollout must not break the site
            value = {"schema": SCHEMA, "error": f"{type(exc).__name__}: {exc}"}
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
        temporary.replace(path)
    return path


def cached_poster(folder, output, interface=None):
    """Small camera preview from the recorded reset observation, cached on disk."""
    import h5py
    from PIL import Image

    source = Path(folder) / "observations.h5"
    if not source.is_file():
        return None
    cameras = (interface or {}).get("video") or ["front_rgb", "wrist_rgb"]
    key = hashlib.sha256(f"{source}:{source.stat().st_mtime_ns}:{cameras}".encode()).hexdigest()[:24]
    path = Path(output) / "previews" / f"{key}.jpg"
    if not path.exists():
        try:
            with h5py.File(source, "r") as h5:
                frames = [Image.fromarray(h5[name][0]) for name in cameras if name in h5 and len(h5[name])]
            if not frames:
                return None
            preview = Image.new("RGB", (sum(f.width for f in frames), max(f.height for f in frames)))
            offset = 0
            for frame in frames:
                preview.paste(frame, (offset, 0))
                offset += frame.width
            preview.thumbnail((640, 320))
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial")
            preview.save(temporary, format="JPEG", quality=85)
            temporary.replace(path)
        except (OSError, ValueError, IndexError):
            return None
    return path
