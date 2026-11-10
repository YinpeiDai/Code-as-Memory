"""Build annotated README videos from recorded observations and memory writes.

Run: python scripts/build_readme_media.py [--only SLUG ...]
Uses the public archive, Pillow, OpenCV, h5py and ffmpeg. No policy execution,
simulator, model calls or changes to source artifacts. Preview caches stay under
public_runs/website/cache/readme_memory; release videos and provenance go to docs/media.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tokenize

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "public_runs"
OUTPUT = ROOT / "docs/media"
CACHE = ARCHIVE / "website/cache/readme_memory"
WIDTH, HEIGHT, VIEW, HEADER, BODY_BOTTOM, FPS = 1080, 554, 480, 48, 554, 10
INK, MUTED, TEAL, GOLD = "#E5EDF6", "#91A3B8", "#55DDC1", "#F2C879"
BG, PANEL, HIGHLIGHT = "#101B2B", "#142235", "#21473F"
FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
CODE = ImageFont.truetype(str(FONT_ROOT / "DejaVuSansMono.ttf"), 32)
LABEL = ImageFont.truetype(str(FONT_ROOT / "DejaVuSans.ttf"), 22)
SMALL = ImageFont.truetype(str(FONT_ROOT / "DejaVuSans.ttf"), 18)
BOLD = ImageFont.truetype(str(FONT_ROOT / "DejaVuSans-Bold.ttf"), 23)
TRAIL_COLORS = {"tcp": (96, 178, 255), "left_ee_pose": (255, 190, 93), "right_ee_pose": (96, 178, 255)}

# Benchmark, setting, task, output slug, nominal playback multiplier.
EXAMPLES = [
    ("robomme", "demo_only_25", "PatternLock", "robomme-pattern-lock", 1),
    ("robomme", "demo_only_25", "VideoUnmaskSwap", "robomme-hidden-objects", 1.5),
    ("rmbench", "demo_only_50", "swap_blocks", "rmbench-swap-blocks", .25),
    ("rmbench", "demo_only_50", "cover_blocks", "rmbench-cover-blocks", .5),
    ("mikasa_robo", "demo_interaction_50", "BatteriesCheckerEasy-3-VLA-v0", "mikasa-batteries", 1.25),
    ("mikasa_robo", "demo_interaction_50", "SeqOfColors3-VLA-v0", "mikasa-color-sequence", 1),
    ("robomemarena", "demo_interaction_50", "T04_ButterNonEmptyDrawer", "robomemarena-drawers", 3),
    ("robomemarena", "demo_interaction_50", "T07_PourSauceFrypanTwice", "robomemarena-pour-twice", 2),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decisions(path):
    return [item["row"] for item in map(json.loads, path.read_text().splitlines())
            if item["kind"] == "decision"]


def selected_memory(task, memory):
    """Display excerpts only; no inferred outcomes or future memory writes."""
    m = memory or {}
    if task == "PatternLock":
        return {"plan": m.get("plan"), "decisions": m.get("decisions")}
    if task == "VideoUnmaskSwap":
        note = ast.literal_eval(m["note"]) if m.get("note") else {}
        return {"colors": note.get("colors"), "next_index": m.get("next_index"), "pending": m.get("pending")}
    if task == "swap_blocks":
        return {k: m.get(k) for k in ("plan", "move_idx", "phase")}
    if task == "cover_blocks":
        key = "uncover_order" if m.get("phase") == "uncover" else "colors"
        return {k: m.get(k) for k in (key, "phase", "index", "pending")}
    if task.startswith("BatteriesChecker"):
        return {k: m.get(k) for k in ("results", "queue", "stage")}
    if task.startswith("SeqOfColors"):
        return {k: m.get(k) for k in ("cue_colors", "touched", "pending")}
    if task.startswith("T04"):
        scores = m.get("scores") or [None] * 3
        colored = {key: None if score is None else score[0]
                   for key, score in zip(("top", "middle", "bottom"), scores)}
        return {"colored": colored, "target": m.get("target"), "pending": m.get("pending")}
    if task.startswith("T07"):
        return {k: m.get(k) for k in ("pours_done", "phase", "pour_xy")}
    raise ValueError(task)


def perceived_position(task, memory):
    """Select already recorded perception values, without evaluating perception again."""
    m = memory or {}
    if task == "PatternLock":
        return {"vision_plan[0] (x, y)": m["vision_plan"][0] if m.get("vision_plan") else None}
    if task == "VideoUnmaskSwap":
        target = m["targets"][m["next_index"]][:2] if m.get("targets") else None
        return {"targets[next_index] (x, y)": target}
    if task == "swap_blocks":
        return {"src_xy (x, y)": m["src_xy"][:2] if m.get("src_xy") else None}
    if task == "cover_blocks":
        index = int(m["pending"].split(":")[1]) if m.get("pending") else None
        return {"block_pos[slot] (x, y, z)": m["block_pos"][index] if index is not None else None}
    if task.startswith("BatteriesChecker"):
        return {"batteries[queue[0]] (x, y, z)": m["batteries"][int(m["queue"][0])] if m.get("queue") else None}
    if task.startswith("SeqOfColors"):
        if m.get("pending", "").startswith("touch:"):
            color = m["pending"].split(":")[1]
            return {"cube_xy[target] (x, y)": m["cube_xy"][m["cube_colors"].index(color)]}
        return {"button_xy (x, y)": m.get("button_xy") or None}
    if task.startswith("T04"):
        return {"cabinet (x, y, z)": m.get("cabinet")}
    if task.startswith("T07"):
        key = "bottle_xy" if m.get("phase") == "pick" else "pour_xy"
        return {key + " (x, y)": m.get(key)}
    raise ValueError(task)


def literal(value):
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(literal(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{0 if abs(value) < .005 else value:.2f}"
    return repr(value)


def code_lines(fields):
    lines = []
    for key, value in fields.items():
        single = f"{key} = {literal(value)}"
        if isinstance(value, dict):
            lines.append((key, f"{key} = {{"))
            for i, (name, val) in enumerate(value.items()):
                lines.append((key, f"  {literal(name)}: {literal(val)}" + ("," if i < len(value) - 1 else "")))
            lines.append((key, "}"))
        elif isinstance(value, list) and value and (key in ("plan", "results", "cue_colors", "colors", "uncover_order") or len(single) > 24):
            lines.append((key, f"{key} = ["))
            for i, val in enumerate(value):
                lines.append((key, "  " + literal(val) + ("," if i < len(value) - 1 else "")))
            lines.append((key, "]"))
        elif CODE.getlength(single) > WIDTH - VIEW - 48:
            lines.extend([(key, f"{key} = ("), (key, "  " + literal(value) + ")")])
        else:
            lines.append((key, single))
    assert len(lines) <= 8, lines
    assert all(CODE.getlength(text) <= WIDTH - VIEW - 48 for _, text in lines), lines
    return lines


def action_caption(task, row):
    if row is None:
        return "Watch the video", "Memory starts after the video"
    m, act = row["memory_written"], row["act"]
    tool = act["tool"]
    if task == "PatternLock":
        return "Retrace the stored path", "Remember the node sequence"
    if task == "VideoUnmaskSwap":
        colors = ast.literal_eval(m["note"])["colors"]
        return "Uncover " + colors[m["next_index"]], "Recall after the containers move"
    if task == "swap_blocks":
        idx = m["move_idx"]
        if idx >= len(m["plan"]):
            return "Confirm the swap", "All three moves completed"
        return ("Pick" if tool == "act_pick" else "Place") + " | " + m["plan"][idx].replace(">", " to "), "Keep the move order"
    if task == "cover_blocks":
        phase, index = m["phase"], m["index"]
        color = m["colors" if phase == "cover" else "uncover_order"][index]
        return phase.capitalize() + " " + color, "Remember what each lid hides"
    if task.startswith("BatteriesChecker"):
        battery = "b" + m["queue"][0]
        return {"pick": "Pick ", "insert": "Test ", "press": "Confirm "}[m["stage"]] + battery, "Retain each lamp result"
    if task.startswith("SeqOfColors"):
        if m["stage"] == "cue":
            return "Watch the color cue", "Store the cue order"
        if m["pending"] == "press":
            return "Confirm the sequence", "All three touches remembered"
        return "Touch " + m["pending"].split(":")[1], "Recall the next color"
    if task.startswith("T04"):
        if m["stage"] == "check":
            level = ("top", "middle", "bottom")[m["idx"]]
            return m["pending"].capitalize() + " " + level, "Keep drawer observations"
        return {"target": "Return to " + m["target"], "pick": "Pick the butter", "place": "Place in " + m["target"]}[m["stage"]], "Recall the occupied drawer"
    if task.startswith("T07"):
        if m["phase"] == "pick":
            return "Pick the sauce bottle", "Count completed pours"
        return f"Pour {m['pours_done'] + 1} of 2", "Update the count between pours"
    raise ValueError(task)


def markers(task, row):
    """Targets/remembered locations, not ground-truth object trackers."""
    if row is None:
        return [], False
    m, args = row["memory_written"], row["act"]["args"]
    if task == "PatternLock":
        return [(list(xy) + [0.0], str(i + 1)) for i, xy in enumerate(m["plan"])], True
    if task == "VideoUnmaskSwap":
        i = m["next_index"]
        return [(m["targets"][i][:2] + [.04], str(i + 1))], False
    if task == "swap_blocks":
        if "x" not in args:
            return [], False
        return [([args["x"], args["y"], .778], "target")], False
    if task == "cover_blocks":
        index = int(m["pending"].split(":")[1])
        return [(m["block_pos"][index], m["colors"][index])], False
    if task.startswith("BatteriesChecker"):
        key = {"insert": "socket", "press": "button"}.get(m["stage"])
        point = m[key] if key else m["batteries"][int(m["queue"][0])]
        return [(point, key or "b" + m["queue"][0])], False
    if task.startswith("SeqOfColors"):
        if m["stage"] == "cue":
            return [], False
        if m["pending"] == "press":
            return [(m["button_xy"] + [.05], "confirm")], False
        color = m["pending"].split(":")[1]
        index = m["cube_colors"].index(color)
        return [(m["cube_xy"][index] + [.05], color)], False
    if task.startswith("T04"):
        if "handle" in args:
            level = ("top", "middle", "bottom")[m["idx"]] if m["stage"] == "check" else m["target"]
            return [(args["handle"], level)], False
        if "butter" in args:
            return [(args["butter"], "butter")], False
        if "place" in args:
            return [(args["place"] + [args["release_z"]], m["target"])], False
        return [], False
    if task.startswith("T07"):
        return ([(m["bottle_xy"] + [.48], "bottle")] if m["phase"] == "pick"
                else [(m["pour_xy"] + [.43], "pan")]), False
    return [], False


def syntax(draw, position, text):
    x, y = position
    draw.text((x, y), text, fill=INK, font=CODE)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text.strip()).readline)
        indent = len(text) - len(text.lstrip())
        for token in tokens:
            color = GOLD if token.type == tokenize.STRING else TEAL if token.type == tokenize.NUMBER or token.string in ("None", "True", "False") else None
            if color:
                px = x + CODE.getlength(text[:token.start[1] + indent])
                draw.text((px, y), token.string, fill=color, font=CODE)
    except (tokenize.TokenError, IndentationError):
        pass  # Individual lines of a multiline list still get their token colors.


def draw_trails(canvas, poses, phase, index, K, E, scale, offset, size):
    """Only recorded history, restarting at phase boundaries; no future trajectory."""
    start = index
    while start > 0 and phase[start - 1] == phase[index]:
        start -= 1
    start = max(start, index - 159)
    overlay = Image.new("RGBA", canvas.size)
    draw = ImageDraw.Draw(overlay)
    for key, positions in poses.items():
        color = TRAIL_COLORS[key]
        points = []
        for world in positions[start:index + 1, :3]:
            pc = E[:, :3] @ world + E[:, 3]
            uv = K @ pc
            x, y = uv[:2] / uv[2] * scale + np.asarray(offset)
            valid = pc[2] > 0 and np.isfinite([x, y]).all() and offset[0] <= x <= offset[0] + size[0] and offset[1] <= y <= offset[1] + size[1]
            points.append((float(x), float(y)) if valid else None)
        for j, (a, b) in enumerate(zip(points, points[1:])):
            if a is not None and b is not None:
                alpha = round(40 + 180 * (j + 1) / max(1, len(points) - 1))
                draw.line([a, b], fill=(*color, alpha), width=4)
        if points and points[-1] is not None:
            x, y = points[-1]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(*color, 255))
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"))


def render(rgb, task, row, fields, position, changed, camera, h5, frame_index, success, poses, phases):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((VIEW, 0, WIDTH, HEIGHT), fill=PANEL)
    demo = not bool(phases[frame_index])
    phase_color = GOLD if demo else TEAL
    draw.text((18, 10), "Demo phase" if demo else "Execution phase", font=BOLD, fill=phase_color)
    draw.text((VIEW + 24, 10), "ROBOT MEMORY", font=LABEL, fill=TEAL)
    frame = Image.fromarray(rgb)
    scale = min((VIEW - 12) / frame.width, (BODY_BOTTOM - HEADER) / frame.height)
    size = (round(frame.width * scale), round(frame.height * scale))
    offset = ((VIEW - size[0]) // 2, HEADER + (BODY_BOTTOM - HEADER - size[1]) // 2)
    canvas.paste(frame.resize(size, Image.Resampling.LANCZOS), offset)
    caption, _ = action_caption(task, row)
    points, connect = markers(task, row)
    projected = []
    K, E = h5[camera + "_K"][frame_index], h5[camera + "_extrinsic"][frame_index]
    draw_trails(canvas, poses, phases, frame_index, K, E, scale, offset, size)
    draw = ImageDraw.Draw(canvas)
    if demo:
        draw.rectangle((offset[0], offset[1], offset[0] + size[0] - 1, offset[1] + size[1] - 1), outline=GOLD, width=6)
        caption = "Watch the video"
    for world, label in points:
        pc = E[:, :3] @ np.asarray(world) + E[:, 3]
        if pc[2] <= 0:
            continue
        uv = K @ pc
        x, y = uv[:2] / uv[2]
        x, y = float(x * scale + offset[0]), float(y * scale + offset[1])
        if offset[0] <= x <= offset[0] + size[0] and offset[1] <= y <= offset[1] + size[1]:
            projected.append((x, y, label))
    if connect and len(projected) > 1:
        draw.line([(x, y) for x, y, _ in projected], fill=TEAL, width=3)
    for x, y, label in projected:
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline=TEAL, width=3)
        width = SMALL.getlength(label) + 14
        tx, ty = min(max(x + 12, 4), VIEW - width - 4), max(HEADER + 4, y - 29)
        draw.rounded_rectangle((tx, ty, tx + width, ty + 25), radius=5, fill=BG)
        draw.text((tx + 7, ty + 2), label, font=SMALL, fill=TEAL)
    if success:
        caption = "TASK COMPLETE"
    badge_width = BOLD.getlength(caption) + 28
    assert badge_width <= VIEW - 20, caption
    draw.rounded_rectangle((12, BODY_BOTTOM - 45, 12 + badge_width, BODY_BOTTOM - 9), radius=8, fill=BG)
    draw.text((26, BODY_BOTTOM - 40), caption, font=BOLD, fill=TEAL if success else GOLD if demo else INK)
    legend = [("Left EEF", "left_ee_pose"), ("Right EEF", "right_ee_pose")] if "left_ee_pose" in poses else [("EEF trail", "tcp")]
    lx = 14
    for text, key in legend:
        draw.rounded_rectangle((lx, HEADER + 5, lx + SMALL.getlength(text) + 27, HEADER + 31), radius=4, fill=BG)
        draw.line((lx + 6, HEADER + 18, lx + 18, HEADER + 18), fill=TRAIL_COLORS[key], width=3)
        draw.text((lx + 23, HEADER + 7), text, font=SMALL, fill=INK)
        lx += SMALL.getlength(text) + 39
    for i, (key, text) in enumerate(code_lines(fields)):
        y = 76 + i * 39
        if key in changed:
            draw.rounded_rectangle((VIEW + 13, y - 2, WIDTH - 12, y + 37), radius=5, fill=HIGHLIGHT)
            draw.rectangle((VIEW + 13, y + 3, VIEW + 17, y + 32), fill=TEAL)
        syntax(draw, (VIEW + 24, y), text)
    for name, value in position.items():
        draw.text((VIEW + 24, 414), "PERCEIVED POSITION · METERS", font=SMALL, fill=MUTED)
        draw.text((VIEW + 24, 440), name, font=LABEL, fill=INK)
        syntax(draw, (VIEW + 24, 472), literal(value))
    return canvas


def build(example, episodes):
    benchmark, setting, task, name, speed = example
    candidates = [row for row in episodes if row["benchmark"] == benchmark and row["setting"] == setting
                  and row["task"] == task and row["kind"] == "test" and row.get("success") and row.get("video")]
    source = min(candidates, key=lambda row: int(row["episode"]))
    video, result = ARCHIVE / source["video"], ARCHIVE / source["result"]
    decision_path = result.parent / "decisions.jsonl"
    rows = decisions(decision_path)
    outcome = json.loads(result.read_text())
    assert outcome["success"] and len(rows) == outcome["decisions"]
    camera = "head" if benchmark == "rmbench" else "agentview" if benchmark == "robomemarena" else "front"
    work = CACHE / name
    work.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    preview = OUTPUT / f"{name}.mp4"
    poster = OUTPUT / f"{name}.jpg"
    timeline, schedule = [], []
    with h5py.File(result.parent / "observations.h5") as h5:
        phases, steps = h5["phase"][:], h5["step"][:]
        poses = {key: h5[key][:, :3] for key in TRAIL_COLORS if key in h5}
        assert poses and all(np.isfinite(value).all() for value in poses.values())
        assert len(steps) == frame_count == len(h5[camera + "_rgb"])
        labels = [("E" if phase else "D") + str(step) for phase, step in zip(phases, steps)]
        assert len(set(labels)) == len(labels)
        starts = [labels.index(row["frames"][-1]) for row in rows]
        assert starts == sorted(set(starts))
        assert all(labels[index] == "E" + str(row["step"]) for index, row in zip(starts, rows))
        boundaries = sorted(set([0, *starts, frame_count]))
        previous = {}
        for a, b in zip(boundaries, boundaries[1:]):
            available = [i for i, start in enumerate(starts) if start <= a]
            row_index = available[-1] if available else -1
            row = rows[row_index] if row_index >= 0 else None
            memory = row["memory_written"] if row else rows[0]["memory_in"]
            fields = selected_memory(task, memory)
            position = perceived_position(task, memory)
            code_lines(fields)
            changed = {key for key, value in fields.items() if previous.get(key) != value}
            count = max(14 if changed else 1, math.ceil((b - a) / source_fps / speed * FPS))
            timeline.append({"output_frame": len(schedule), "source_frame": a, "source_label": labels[a],
                             "decision_index": row_index, "fields": fields, "perceived_position": position,
                             "phase": "execution" if phases[a] else "demo", "highlighted_fields": sorted(changed)})
            for j, index in enumerate(np.linspace(a, b - 1, count).astype(int)):
                schedule.append((int(index), row_index, fields, position, changed if j < 10 else set()))
            previous = fields
        schedule.extend([(frame_count - 1, len(rows) - 1, previous, position, set())] * 14)
        poster_row = min(len(rows) - 1, len(rows) * 2 // 3)
        poster_index = next(e["output_frame"] for e in timeline if e["decision_index"] == poster_row) + 6
        command = ["ffmpeg", "-v", "error", "-y", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                   "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "1",
                   "-movflags", "+faststart", "-an", str(preview)]
        log_path = work / "encode.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            decoded_index, bgr = -1, None
            snapshots = {event["output_frame"] for event in timeline} | {len(schedule) - 1}
            try:
                for output_index, (index, row_index, fields, position, changed) in enumerate(schedule):
                    while decoded_index < index:
                        ok, bgr = capture.read()
                        if not ok:
                            raise ValueError(f"Missing video frame {index}: {name}")
                        decoded_index += 1
                    raw_width = h5[camera + "_rgb"].shape[2]
                    rgb = cv2.cvtColor(bgr[:, :raw_width], cv2.COLOR_BGR2RGB)
                    row = rows[row_index] if row_index >= 0 else None
                    frame = render(rgb, task, row, fields, position, changed, camera, h5, index,
                                   index == frame_count - 1, poses, phases)
                    process.stdin.write(frame.tobytes())
                    if output_index in snapshots:
                        frame.save(work / f"state_{output_index:04d}.png")
                    if output_index == poster_index:
                        still = frame.copy()
                        draw = ImageDraw.Draw(still, "RGBA")
                        x, y = VIEW // 2, (HEADER + BODY_BOTTOM) // 2
                        draw.ellipse((x - 33, y - 33, x + 33, y + 33), fill=(16, 27, 43, 210), outline=(255, 255, 255, 235), width=2)
                        draw.polygon([(x - 9, y - 17), (x - 9, y + 17), (x + 19, y)], fill=(255, 255, 255, 255))
                        still.save(poster, quality=92, subsampling=0)
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(log_path.read_text())
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                capture.release()
        calibration = hashlib.sha256()
        for key in ("phase", "step", camera + "_K", camera + "_extrinsic", *poses):
            calibration.update(key.encode())
            calibration.update(np.ascontiguousarray(h5[key][:]).tobytes())
    record = {
        **{key: source.get(key) for key in ("benchmark", "setting", "task", "episode", "run_id", "checkpoint",
                                          "success", "program_hash", "harness_hash", "benchmark_hash")},
        "source_video": source["video"], "source_video_sha256": digest(video),
        "source_result": source["result"], "source_result_sha256": digest(result),
        "source_decisions": str(decision_path.relative_to(ARCHIVE)), "source_decisions_sha256": digest(decision_path),
        "source_observations": str((result.parent / "observations.h5").relative_to(ARCHIVE)),
        "alignment_calibration_and_pose_sha256": calibration.hexdigest(), "camera": camera,
        "goal": outcome["goal"], "phase_labels": {"0": "Demo phase", "1": "Execution phase"},
        "demo_frames": int(np.count_nonzero(phases == 0)), "eef_pose_fields": list(poses), "trail_history_frames": 160,
        "preview": preview.name, "preview_sha256": digest(preview), "preview_dimensions": [WIDTH, HEIGHT],
        "preview_format": "video/mp4", "poster": poster.name, "poster_sha256": digest(poster), "poster_output_frame": poster_index,
        "preview_fps": FPS, "nominal_playback_multiplier": speed, "preview_duration_seconds": len(schedule) / FPS,
        "source_frames": frame_count, "sampled_source_frames": [r[0] for r in schedule], "memory_timeline": timeline,
    }
    shutil.copyfile(preview, work / "presentation.mp4")
    shutil.copyfile(poster, work / "poster.jpg")
    (work / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{name}: {len(schedule) / FPS:.1f}s, {preview.stat().st_size / 1048576:.2f} MiB, {len(rows)} recorded decisions", flush=True)
    return record


def write_gallery(clips, output, cached=False):
    cards = []
    for clip in clips:
        slug = Path(clip["preview"]).stem
        video = f"{slug}/presentation.mp4" if cached else clip["preview"]
        poster = f"{slug}/poster.jpg" if cached else clip["poster"]
        cards.append(f'<article><h2>Task: {html.escape(clip["task"])}</h2><p><strong>Task Goal:</strong> {html.escape(clip["goal"])}</p>'
                     f'<video controls playsinline preload="none" poster="{poster}" src="{video}"></video></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rollout Examples · CaM</title><style>
body{margin:0;background:#0e1725;color:#e5edf6;font:17px/1.6 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:32px 20px}
h1{font-size:32px}h2{font-size:21px;margin:0}p{color:#aebed1;margin:12px 0 18px}article{margin:32px 0 48px}video{width:100%;display:block;border-radius:12px;background:#101b2b}
</style><main><h1>Rollout Examples</h1><p>Recorded memory, perceived positions, and end-effector motion.</p>'''
    output.write_text(page + "\n".join(cards) + "</main></html>\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=[item[3] for item in EXAMPLES])
    args = parser.parse_args()
    catalog = json.loads((ARCHIVE / "dashboard/index.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = [item for item in EXAMPLES if not args.only or item[3] in args.only]
    with ThreadPoolExecutor(max_workers=3) as pool:
        clips = list(pool.map(lambda item: build(item, catalog["episodes"]), selected))
    if args.only:
        print("Partial render complete; run without --only to update the release manifest.")
    else:
        manifest = {"schema": "cam-readme-media/3", "source_root": "public_runs",
                    "selection": "Lowest numbered successful held-out episode for each selected task.",
                    "presentation": "H.264 MP4, front/overview video with recorded memory excerpts and perceived positions. Changed fields highlight for one second; changed states last at least 1.4 seconds. Final observation holds for 1.4 seconds. Timing is illustrative, not real-time robot latency.",
                    "memory_format": "Original selected values; positions rounded to two decimals in world-frame meters. Position panels select retained estimates from logged memory, not hidden object state or newly executed perception. VideoUnmaskSwap colors are parsed from memory.note; T04 colored maps drawer names to the first component of memory.scores (colored-pixel counts). No memory is inferred between decisions or after episode termination.",
                    "annotations": "Demo/execution labels follow recorded observation phase, with a gold border and Watch the video caption for demo frames. EEF trails project at most 160 past recorded poses and reset at phase boundaries; two-arm poses use separate colors. Point markers project remembered locations or action targets. Display surface heights: PatternLock 0.0m, VideoUnmaskSwap 0.04m, swap_blocks 0.778m, T07 bottle 0.48m and pan 0.43m. Goal text comes verbatim from the selected result.json and appears only in the surrounding README/gallery text, not in video frames.",
                    "builder_sha256": digest(Path(__file__)), "clips": clips}
        (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        write_gallery(clips, OUTPUT / "index.html")
        write_gallery(clips, CACHE / "index.html", cached=True)
