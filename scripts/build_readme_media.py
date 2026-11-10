"""Build README previews from successful, archived evaluation recordings.

Run from the repository root: python scripts/build_readme_media.py
Requires public_runs/dashboard/index.json, its recorded videos, and ffmpeg.
No simulator, model, demonstration H5, or original artifact is modified.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "public_runs"
OUTPUT = ROOT / "docs/media"
# Benchmark, setting, task, filename, playback multiplier, camera crop.
EXAMPLES = [
    ("robomme", "demo_only_25", "PatternLock", "robomme-pattern-lock", 1, None),
    ("robomme", "demo_only_25", "VideoUnmaskSwap", "robomme-hidden-objects", 1.5, None),
    ("rmbench", "demo_only_50", "swap_blocks", "rmbench-swap-blocks", 0.25, "640:240:0:0"),
    ("rmbench", "demo_only_50", "cover_blocks", "rmbench-cover-blocks", 0.5, "640:240:0:0"),
    ("mikasa_robo", "demo_interaction_50", "BatteriesCheckerEasy-3-VLA-v0", "mikasa-batteries", 1.25, None),
    ("mikasa_robo", "demo_interaction_50", "SeqOfColors3-VLA-v0", "mikasa-color-sequence", 1, None),
    ("robomemarena", "demo_interaction_50", "T04_ButterNonEmptyDrawer", "robomemarena-drawers", 3, None),
    ("robomemarena", "demo_interaction_50", "T07_PourSauceFrypanTwice", "robomemarena-pour-twice", 2, None),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(example, episodes):
    benchmark, setting, task, name, speed, crop = example
    candidates = [row for row in episodes if row["benchmark"] == benchmark
                  and row["setting"] == setting and row["task"] == task
                  and row["kind"] == "test" and row.get("success") and row.get("video")]
    source = min(candidates, key=lambda row: int(row["episode"]))
    video = ARCHIVE / source["video"]
    result = ARCHIVE / source["result"]
    outcome = json.loads(result.read_text())
    if not outcome.get("success"):
        raise ValueError(f"Selected recording is not a successful evaluation: {task}")
    recording = OUTPUT / f"{name}.mp4"
    preview = OUTPUT / f"{name}.gif"
    shutil.copyfile(video, recording)
    filters = ([f"crop={crop}"] if crop else []) + [
        f"setpts=(PTS-STARTPTS)/{speed}", "fps=10",
        "scale=384:192:force_original_aspect_ratio=decrease:flags=lanczos",
        "pad=384:192:(ow-iw)/2:(oh-ih)/2:color=0x111827",
    ]
    graph = ",".join(filters) + (
        ",split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    )
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "1", "-i", str(video),
                    "-filter_complex_threads", "1", "-filter_complex", graph,
                    "-loop", "0", "-an", str(preview)], check=True)
    original_hash = digest(video)
    assert digest(recording) == original_hash
    print(f"{name}: {preview.stat().st_size / 1024:.0f} KiB GIF", flush=True)
    return {
        **{key: source.get(key) for key in ("benchmark", "setting", "task", "episode", "run_id",
              "checkpoint", "success", "program_hash", "harness_hash", "benchmark_hash")},
        "source_video": source["video"], "source_video_sha256": original_hash,
        "source_result": source["result"], "source_result_sha256": digest(result),
        "recording": recording.name, "recording_sha256": original_hash,
        "preview": preview.name, "preview_sha256": digest(preview),
        "preview_playback_multiplier": speed, "preview_crop": crop,
        "preview_fps": 10, "preview_dimensions": [384, 192],
    }


if __name__ == "__main__":
    catalog = json.loads((ARCHIVE / "dashboard/index.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        clips = list(pool.map(lambda item: build(item, catalog["episodes"]), EXAMPLES))
    manifest = {"schema": "cam-readme-media/1", "source_root": "public_runs",
                "selection": "Lowest numbered successful held-out episode for each selected task.",
                "recordings": "Byte-identical copies of the archived MP4s; previews adjust cameras, size and speed.",
                "clips": clips}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
