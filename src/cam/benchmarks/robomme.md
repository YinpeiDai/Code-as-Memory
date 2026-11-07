## RoboMME data and simulator protocol

Under `api_reference/runtime/`, read `observations.py` for `Frame` and
`Observations`, and `robot.py` for `Robot.now()`, `frames()` and `step()`.
When demonstrations are supplied, `trajectory.py` documents `Trajectory` and
`training_episodes`. These are reference source files; offline scripts import
the installed `cam.runtime` package as shown below. Generated policy functions
use their supplied `obs` and `robot` arguments without imports.

Only your selected extracted H5 is mounted at `/demonstrations` when allowed.
Use `from cam.runtime.trajectory import Trajectory, training_episodes` in offline
command scripts. A trajectory exposes `T`, `goal`, `difficulty`, `is_demo`,
`exec_start`, `action_alignment`, RGB, optional depth, calibration, measured native
TCP, joints, fingers and recorded expert actions. Read `action_alignment` before
pairing commands with observations. Recorded frame index is not a control counter.
Missing commands remain missing. Expert H5 is absent during policy execution.
The trajectory reader lists offline demo IDs; the assignment separately lists
the smaller allowed simulator training subset when configured.

Online `obs.frames` contains all frames since the previous action; `obs.current`
is latest. Native task-provided video observations precede execution when enabled.
They are distinct from offline expert demonstrations. Frames expose front/wrist
RGB, optional aligned depth (int16 optical-axis millimetres, 0 unknown), 3×3
intrinsics, 3×4 world-to-camera extrinsics, measured native TCP XYZ metres and
XYZ roll/pitch/yaw radians, joints, finger widths, phase, step and label.
No masks, object poses, oracle subgoals, success labels from demos or simulator
internals are available to the coding model or generated policy.

`robot.now()` returns the latest frame. `robot.frames()` returns the action's
observed frames. `robot.step(x,y,z,roll,pitch,yaw,gripper)` sends one absolute world
TCP target for one native control step. -1 closes and +1 opens the gripper; stick
tasks ignore it. Use gradual motion and observed feedback. Unreachable commands
may incur native penalties. Native success/failure/timeout/error ends the episode.
The assignment specifies the decision and per-action call caps. RoboMME uses
the original wrapper and `max_steps=1300`; CaM does not modify scoring.
