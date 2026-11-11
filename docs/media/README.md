# Rollout examples

Eight successful held-out episodes pair the main camera with recorded policy
memory. All are episode 0, the lowest-numbered successful episode for their task.
See [results](../results.md) for aggregate scores and failures.

The README embeds GitHub-hosted MP4s with playback controls. Open the
[local gallery](index.html) after cloning for the same videos.
[Attachment metadata](github_assets.json) records URLs and matching video hashes.

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


## Reading the videos

Gold-bordered **Demo phase** frames are task-provided videos in the two RoboMME
examples. **Execution phase** shows policy control. Memory changes follow recorded
decision boundaries; the first update is revealed line by line for readability.
Playback speed and pauses are edited, so these videos do not measure latency.

Panels show selected memory fields, remembered positions in world-frame metres,
action arguments and recorded action feedback. Coordinates are rounded; long
values are abbreviated. Camera trails use recorded end-effector poses and
calibration. No perception is rerun or hidden simulator state added.

VideoUnmaskSwap's colors come from its stored note; drawer labels summarize
recorded pixel scores. The pour example ends during its second pour, before
another memory update records completion. The terminal success label comes from
the episode result. MIKASA examples illustrate full-benchmark tasks beyond the
five-task VLA comparison.

[manifest.json](manifest.json) retains source paths, policy identities, hashes,
frames and every displayed memory state. Original recordings remain in the archive.

## Rebuild

With the public archive, root Python environment, `ffmpeg` and DejaVu fonts:

```bash
python scripts/build_readme_media.py
```

The builder reads recorded videos and logs without running policies, simulators
or models. Preview caches and encoding logs go under
`public_runs/website/cache/readme_memory/`.
