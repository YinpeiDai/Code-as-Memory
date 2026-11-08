## RMBench data and simulator protocol

Under `api_reference/runtime/`, read `observations.py` for `Frame` and
`Observations`, and `robot.py` for `Robot.now()`, `frames()` and `step()`.
When demonstrations are supplied, `trajectory.py` documents `Trajectory` and
`training_episodes`. These are reference source files; offline scripts import
the installed `cam.runtime` package as shown below. Generated policy functions
use their supplied `obs` and `robot` arguments without imports.

Only your selected extracted H5 is mounted at `/demonstrations` when allowed.
Use `from cam.runtime.trajectory import Trajectory, training_episodes` in offline
command scripts. A trajectory exposes `T`, `goal`, `action_alignment`,
`step_semantics` and per-frame fields named like the online sensors. Recorded
frames are a scripted dual-arm expert's states, saved about every 15 simulator
steps; they contain no commands and are not CaM actions. Camera images were JPEG
compressed in the released data. Expert H5 is absent during policy execution.

The robot is a two-arm aloha-agilex in RMBench's `demo_clean` setting.
With RGB-D enabled, each camera also provides aligned int16 optical-axis depth
in millimetres (`*_depth`, 0 invalid/out of range), at the same 320x240 resolution.
RGB-D demonstrations are regenerated native-expert recordings; their RGB and
depth are captured together. They remain sampled states, not action commands.

`obs.frames` holds one frame after each action (one at reset); `obs.current` is
latest. Cameras give 240×320 RGB, 3×3 OpenCV intrinsics in pixels and 3×4
OpenCV world-to-camera extrinsics; wrist cameras move with the arms. World units
are metres, z up; poses are x, y, z and quaternion w, x, y, z. Each
`*_ee_pose` is that arm's end-effector frame; the gripper's grasp centre lies
0.12 m along its local +x axis. Joint and gripper values are commanded targets.
Depth is available only when enabled in the run's sensor interface.
No masks, object poses, subtask labels, rewards or simulator internals
are available to the coding model or generated policy.

`robot.step(...)` sends one native end-effector action for both arms: absolute
world targets in the `*_ee_pose` convention and gripper openings (0 closed,
1 open). Each arm is planned separately, avoiding only its own links and the
table (not objects or the other arm), then executed to completion while its
gripper moves to the target. A target that cannot be planned leaves that arm
still for 50 simulator steps and raises `MotionError`; the action still counts.
Repeating the current pose holds an arm. Each act call has a wall-clock limit,
so split long action sequences across decisions.
Success is checked every simulator step and ends the episode at once, even
mid-action; otherwise the episode ends at the task's native action limit. There
is no early failure signal. CaM does not modify RMBench scoring.
