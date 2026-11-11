# Rollout Examples

Eight selected **successful held-out episodes** pair the main camera with large,
readable excerpts of the policy's recorded memory. Each is episode 0, the lowest
numbered successful episode for its task. Aggregate scores are in [results](../results.md).

The README embeds compact H.264 MP4s as GitHub video attachments with playback
controls. Each attachment URL is on its own line so GitHub generates the player
and signs private-media URLs for the viewer. Custom `<video>` tags can leave
private attachment references unresolved. [github_assets.json](github_assets.json) records each attachment URL and
the matching local video hash. For an offline gallery with the same videos, open
[index.html](index.html) in a browser after cloning.

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
separate video-demo phase. Task goals appear only in the surrounding README/gallery
text, not inside videos. They match the recorded episode goals verbatim.
The six RMBench, MIKASA-Robo and RoboMemArena goals were also checked against
their official repositories at the pinned revisions:

- RMBench: the `seen` instructions for [swap_blocks](https://github.com/RoboTwin-Platform/RMBench/blob/7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7/description/task_instruction/swap_blocks.json) and [cover_blocks](https://github.com/RoboTwin-Platform/RMBench/blob/7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7/description/task_instruction/cover_blocks.json), including the original spelling.
- MIKASA-Robo: `LANGUAGE_INSTRUCTION` in [BatteriesCheckerEasy](https://github.com/CognitiveAISystems/MIKASA-Robo/blob/509b875f3d207c287497c0a897661062de928bb0/mikasa_robo_suite/vla/memory_envs/batteries_checker_easy_vla.py) and [SeqOfColors](https://github.com/CognitiveAISystems/MIKASA-Robo/blob/509b875f3d207c287497c0a897661062de928bb0/mikasa_robo_suite/vla/memory_envs/seq_of_colors_vla.py).
- RoboMemArena: `TASK_PROMPTS["task4"]` and `TASK_PROMPTS["task7"]` in [eval_common.py](https://github.com/OpenHelix-Team/RoboMemArena/blob/830a31ae7c08fb99f514843275cd264a2bf36f16/evaluation_benchmark/scripts/eval_common.py).

SeqOfColors3 also has the separately documented
[cue-order clarification](../../src/cam/prompts/mikasa_robo/seq_of_colors3.md).

At the first execution frame, the camera pauses on the recorded initial memory
(selected fields are `None` in these examples). The first returned memory is
revealed line by line, followed by the chosen action, before motion resumes.
This reveal presents one saved return; it does not reconstruct assignments inside
the function or measure its runtime.

Subsequent memory changes follow recorded decision boundaries, synchronized through
the observation frame labels. The panels select relevant fields and round coordinates
to two decimals. VideoUnmaskSwap's `colors` is extracted from its stored `note`;
the drawer `colored` dictionary names the first component of each stored `scores`
entry: colored-pixel counts, not simulator occupancy labels. No intermediate or
terminal memory writes are invented. The pour episode ends during its second
pour, before another decision can record `pours_done = 2`.

The position panel selects relevant estimates retained in recorded memory, labeled
with their field names and coordinates in world-frame meters. No perception is
rerun and no hidden object state is used. Missing estimates appear as `None`.

The function-output panel shows selected action arguments returned by
`perceive_decide()` and actual feedback from the preceding action, such as
`grasped`, `lamp_lit`, or drawer movement. Long values use an ellipsis; full values
remain in the logs. Internal perception-helper returns were not separately
recorded. The terminal success label comes from the episode result.

EEF trails project up to 160 past recorded poses through the camera calibration,
resetting at phase boundaries. RMBench's two arms use separate colors; its sparse
action-boundary recordings yield connecting segments. Teal markers show remembered
positions or action targets. RMBench uses its wide head camera; RoboMME and MIKASA
use `front`, and RoboMemArena uses `agentview`. The wrist view is replaced by memory.

Videos are 1080×650 at 10 fps with 32-pixel memory text and 26-pixel function-output
text. The initial memory holds for 1.5 seconds; the first update reveals one line
every 0.3 seconds, holds for 0.8 seconds, then shows its action for 1.2 seconds.
Changed fields highlight for
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
