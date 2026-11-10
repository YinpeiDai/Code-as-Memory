# Successful held-out episodes

Eight selected **successful held-out episodes** pair the main camera with large,
readable excerpts of the policy's recorded memory. Each is episode 0, the lowest
numbered successful episode for its task. Aggregate scores are in [results](../results.md).

The README links compact H.264 MP4s through JPEG previews. For a local gallery
with playback controls, open [index.html](index.html) in a browser after cloning.

| Benchmark | Video | Memory shown |
|---|---|---|
| RoboMME | [PatternLock](robomme-pattern-lock.mp4) | Node list and decision count |
| RoboMME | [VideoUnmaskSwap](robomme-hidden-objects.mp4) | Remembered colors and next target |
| RMBench | [swap_blocks](rmbench-swap-blocks.mp4) | Move list, move index and phase |
| RMBench | [cover_blocks](rmbench-cover-blocks.mp4) | Stored colors, uncover order and progress |
| MIKASA-Robo | [BatteriesCheckerEasy-3](mikasa-batteries.mp4) | Lamp results, remaining queue and stage |
| MIKASA-Robo | [SeqOfColors3](mikasa-color-sequence.mp4) | Cue sequence and completed touches |
| RoboMemArena | [ButterNonEmptyDrawer](robomemarena-drawers.mp4) | Drawer observations and chosen target |
| RoboMemArena | [PourSauceFrypanTwice](robomemarena-pour-twice.mp4) | Completed-pour count and phase |

Recorded demo frames have a gold border, **Demo phase**, and **Watch the video**;
other frames show **Execution phase**. Only the two RoboMME examples contain a
separate video-demo phase. Each official task goal is copied verbatim from the
episode result, including native wording. SeqOfColors3 also has the documented
[cue-order clarification](../../src/cam/prompts/mikasa_robo/seq_of_colors3.md).

Memory changes only at recorded decision boundaries, synchronized through the
observation frame labels. The panels select relevant fields and round coordinates
to two decimals. VideoUnmaskSwap's `colors` is extracted from its stored `note`;
the drawer `colored` dictionary names the first component of each stored `scores`
entry: colored-pixel counts, not simulator occupancy labels. No intermediate or
terminal memory writes are invented. The pour episode ends during its second
pour, before another decision can record `pours_done = 2`.

The position panel selects relevant estimates retained in recorded memory, labeled
with their field names and coordinates in world-frame meters. No perception is
rerun and no hidden object state is used. Missing estimates appear as `None`.

EEF trails project up to 160 past recorded poses through the camera calibration,
resetting at phase boundaries. RMBench's two arms use separate colors; its sparse
action-boundary recordings yield connecting segments. Teal markers show remembered
positions or action targets. RMBench uses its wide head camera; RoboMME and MIKASA
use `front`, and RoboMemArena uses `agentview`. The wrist view is replaced by memory.

Videos are 1080×650 at 10 fps with 32-pixel code text. Changed fields highlight for
one second, each changed state stays visible for at least 1.4 seconds, and the
terminal frame holds for 1.4 seconds. Playback speed varies for readability;
RMBench's source records action boundaries. These are illustrations, not timing
measurements.

[manifest.json](manifest.json) includes source paths relative to `public_runs`,
policy identity, input/output hashes, camera choice, sampled source frames, and
every displayed memory state and position. Original MP4s, observations and decision logs remain
in the archive. MIKASA examples use the initial ten-task batch with full evidence.

Rebuild with the original public archive, the installed root Python environment,
`ffmpeg`, and system DejaVu fonts:

```bash
python scripts/build_readme_media.py
```

The builder reads videos, decision logs, recorded poses and camera calibration.
It does not execute policies, start simulators, or call models. A playable gallery,
preview snapshots, encoding logs and provenance are also saved under
`public_runs/website/cache/readme_memory/`.
