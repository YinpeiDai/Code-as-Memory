# Simulator interface adaptations

CaM wraps each official benchmark behind `Observations`, `robot.step(...)`, and
a common trajectory format. A trusted broker owns the simulator, resets and
scoring; the isolated policy receives declared camera and robot-state arrays.
The shared API preserves each benchmark's action units, pose conventions,
gripper behavior and native step accounting.

| Benchmark | Preserved from the original benchmark | Adaptation for CaM |
|---|---|---|
| [RoboMME](../src/cam/interface/robomme/adapter.py) | Native episode/seed/difficulty resolution, absolute end-effector pose actions, task-provided reset videos, control-step accounting and scoring. | Converts native observation lists into front/wrist RGB-D frames, camera calibration and robot state. Reset-video frames are distinguished from execution frames. Released demonstration recordings are retained. |
| [MIKASA-Robo](../src/cam/interface/mikasa_robo/adapter.py) | VLA task definitions, canonical wrappers with overlays disabled, normalized delta-pose actions, task horizons and native success tests. | Renders front/wrist cameras at **256×256 RGB-D** and retains depth, calibration and robot state before canonical observation flattening. Backend selection is described below. |
| [RMBench](../src/cam/interface/rmbench/adapter.py) | `demo_clean` dual-arm setup, absolute end-effector targets with quaternions, native action counting and `eval_success` scoring. | Exposes head, front and both wrist cameras at native **320×240**, with RGB-D, calibration and per-arm state. Motion feedback uses public post-action poses, as described below. |
| [RoboMemArena](../src/cam/interface/robomemarena/adapter.py) | LIBERO/MuJoCo setup, seeded reset and ten settling steps, normalized `OSC_POSE` actions, official stage checks and counting-task monitoring. | Keeps the official **256×256** policy-image preprocessing from 480×640 renders; adds aligned depth, calibrated camera geometry and robot state. Ends at the official scoring decision point, omitting post-goal steps that cannot change the result. |

## Demonstrations and public observations

We replay released MIKASA-Robo and RoboMemArena expert trajectories to record
**256×256 RGB-D demonstrations through the same public observation interface used
online**. RoboMME uses its released RGB-D recordings. RMBench regenerates official
expert trajectories at the released training seeds and records native 320×240
RGB-D. The downloadable pools already contain these prepared demonstrations;
installation and normal experiment preparation do not replay them.

Frames use RGB `uint8`, aligned depth in optical-axis millimetres (`int16`, zero
for invalid depth), camera intrinsics/extrinsics and the declared robot state.
Raw simulator dictionaries, object poses, segmentation and hidden task/stage state
stay inside the broker. Benchmark-specific action semantics remain in each
public interface guide, including differing gripper signs and rotation formats.

## Execution differences

**MIKASA physics.** Training interactions and evaluation default to CPU physics
and CPU IK, except `ShellGameTouch-VLA-v0`, which uses GPU physics and GPU IK.
Rendering uses the GPU independently. The runtime selects and records the backend;
the coding agent does not choose it. Our primary comparison is the five-task
`mikasa_robo_vla_suite`, following prior publications; the full 90-task solutions
use the same adapter. See the [MIKASA guide](benchmarks/mikasa_robo.md).

**RMBench motion feedback.** Native planner-status flags remain private. After a
nonterminal action, CaM compares the observed end-effector pose with the requested
target and raises `MotionError` above **0.02 m** position error or **0.1 rad**
rotation error. This uses public observations and does not change native scoring.
Public robot-only IK/limit helpers support offline development where available.

Each run pins its interface, metadata and source version. CaM additionally
enforces policy decision, action-call and wall-clock limits, recording policy
failures separately from unscorable infrastructure errors. Historical runs retain
their original interfaces and results.

[Benchmark guides](benchmarks/README.md) · [Data setup](getting_started.md) ·
[Experiment protocol](reproducing.md)
