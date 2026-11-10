# Recorded Robot Memory examples

The main README shows eight selected **successful held-out evaluation episodes**
from the published settings. These illustrations do not replace the aggregate
[benchmark results](../results.md). Every example uses episode 0, the lowest
numbered successful episode for its selected task.

| Benchmark | Task | Full recording | Preview speed |
|---|---|---|---:|
| RoboMME | PatternLock | [MP4](robomme-pattern-lock.mp4) | 1× |
| RoboMME | VideoUnmaskSwap | [MP4](robomme-hidden-objects.mp4) | 1.5× |
| RMBench | swap_blocks | [MP4](rmbench-swap-blocks.mp4) | 0.25× |
| RMBench | cover_blocks | [MP4](rmbench-cover-blocks.mp4) | 0.5× |
| MIKASA-Robo | BatteriesCheckerEasy-3-VLA-v0 | [MP4](mikasa-batteries.mp4) | 1.25× |
| MIKASA-Robo | SeqOfColors3-VLA-v0 | [MP4](mikasa-color-sequence.mp4) | 1× |
| RoboMemArena | T04_ButterNonEmptyDrawer | [MP4](robomemarena-drawers.mp4) | 3× |
| RoboMemArena | T07_PourSauceFrypanTwice | [MP4](robomemarena-pour-twice.mp4) | 2× |

MP4s are byte-identical copies of the archived recordings, including all stored
camera views and frames. GIF previews cover the same recordings at 384×192 and
10 fps, with resized views and adjusted playback speed. RMBench previews show
the first two of its four camera views. Its original recording samples action
boundaries; playback speed is not a measurement of real-time robot performance.

[manifest.json](manifest.json) records each source path relative to `public_runs`,
setting, episode, policy/harness identity, original result hash, media hashes,
and preview transform. MIKASA examples come from the initial ten-task batch,
which has the full recorded evidence.

To rebuild with the original public archive available:

```bash
python scripts/build_readme_media.py
```

The builder reads existing MP4s and result JSONs. It never starts learning or a
simulator and leaves the source archive unchanged.
