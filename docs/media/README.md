# Robot Memory GIFs

Eight selected **successful held-out episodes** pair the main camera with large,
readable excerpts of the policy's recorded memory. Each is episode 0, the lowest
numbered successful episode for its task. Aggregate scores are in [results](../results.md).

| Benchmark | GIF | Memory shown |
|---|---|---|
| RoboMME | [PatternLock](robomme-pattern-lock.gif) | Node list and decision count |
| RoboMME | [VideoUnmaskSwap](robomme-hidden-objects.gif) | Remembered colors and next target |
| RMBench | [swap_blocks](rmbench-swap-blocks.gif) | Move list, move index and phase |
| RMBench | [cover_blocks](rmbench-cover-blocks.gif) | Stored colors, uncover order and progress |
| MIKASA-Robo | [BatteriesCheckerEasy-3](mikasa-batteries.gif) | Lamp results, remaining queue and stage |
| MIKASA-Robo | [SeqOfColors3](mikasa-color-sequence.gif) | Cue sequence and completed touches |
| RoboMemArena | [ButterNonEmptyDrawer](robomemarena-drawers.gif) | Drawer observations and chosen target |
| RoboMemArena | [PourSauceFrypanTwice](robomemarena-pour-twice.gif) | Completed-pour count and phase |

Memory changes only at recorded decision boundaries, synchronized through the
observation frame labels. The panels select relevant fields and round coordinates
to two decimals. VideoUnmaskSwap's `colors` is extracted from its stored `note`;
the drawer `colored` dictionary names the first component of each stored `scores`
entry: colored-pixel counts, not simulator occupancy labels. No intermediate or
terminal memory writes are invented. The pour episode ends during its second
pour, before another decision can record `pours_done = 2`.

Scene markers project remembered positions or current action targets using the
recorded camera calibration. They are explanatory overlays, not ground-truth
tracking. RMBench uses its wide head camera; RoboMME and MIKASA use `front`, and
RoboMemArena uses `agentview`. The wrist view is replaced by the memory panel.

GIFs are 960×480 at 10 fps with 32-pixel code text. Changed fields highlight for
one second, each changed state stays visible for at least 1.4 seconds, and the
terminal frame holds for 1.4 seconds. Playback speed varies for readability;
RMBench's source records action boundaries. These are illustrations, not timing
measurements.

[manifest.json](manifest.json) includes source paths relative to `public_runs`,
policy identity, input/output hashes, camera choice, sampled source frames, and
every displayed memory state. Original MP4s, observations and decision logs remain
in the archive. MIKASA examples use the initial ten-task batch with full evidence.

Rebuild with the original public archive, the installed root Python environment,
`ffmpeg`, and system DejaVu fonts:

```bash
python scripts/build_readme_media.py
```

The builder reads videos, decision logs, and observation calibration. It does not
execute policies, start simulators, or call models. Preview snapshots, encoding
logs and provenance are saved under `public_runs/website/cache/readme_memory/`.
