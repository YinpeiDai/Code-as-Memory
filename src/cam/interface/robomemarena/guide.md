## RoboMemArena data and simulator protocol

Under `api_reference/runtime/`, read `observations.py` for `Frame` and
`Observations`, and `robot.py` for `Robot.now()`, `frames()` and `step()`.
When demonstrations are supplied, `trajectory.py` documents `Trajectory` and
`training_episodes`. These are reference source files; offline scripts import
the installed `cam.runtime` package as shown below. Generated policy functions
use their supplied `obs` and `robot` arguments without imports.

Only your selected extracted H5 is mounted at `/demonstrations` when allowed.
Use `from cam.runtime.trajectory import Trajectory, training_episodes` in offline
command scripts. Demonstrations are simulator replays of the released expert
actions, recorded with the online sensors: a trajectory exposes `T`, `goal`,
`action_alignment`, `step_semantics`, every per-frame sensor below, static
`agentview_K`/`wrist_K` (`traj.static`) and the replayed `action`. Read
`action_alignment` before pairing commands with frames. Frame 0 is the first
online frame; a demonstration ends at its scored success. Episode IDs refer to
fixed training identities; follow the assignment's separate demonstration and
interaction lists. Expert H5 is absent during policy execution.

The simulator is the official LIBERO/MuJoCo environment. Episodes start after 10
uncounted settling steps (no motion, gripper opening); the first frame has step 0
and each `robot.step` returns one frame. `obs.frames` holds all frames since the
previous action; `obs.current` is latest. Images are the official 256×256 policy
view, upright (row 0 at the top), resized non-uniformly from a 480×640 render;
the 3×3 intrinsics include that resize (OpenCV pixel centres). Extrinsics are 3×4
world-to-camera `[R|t]`, OpenCV axes (x right, y down, z forward), metres; the
wrist camera moves with the hand. `tcp` is the measured end-effector XYZ (world
metres) plus robosuite `quat2axisangle` of its xyzw quaternion in radians, angle
not wrapped (norm can exceed π), as in the demonstrations. `joints` are 7 arm
angles (rad); `fingers` are 2 finger joint positions (m). Depth, when enabled, is
int16 optical-axis millimetres aligned with the RGB (0 unknown). No masks, object
poses, stage labels or simulator internals are available.

`robot.step(dx, dy, dz, drx, dry, drz, gripper)` is one native 20 Hz OSC_POSE
delta step, components clipped to [-1, 1]: position offsets the controller target
by up to 0.05 m (about 1 cm of actual motion per full-scale step) and rotation turns
the target by an axis-angle up to 0.5 rad, both in the world frame. Only the
gripper's sign acts: +1 closes and -1 opens (opposite to some benchmarks); 0 holds.
Fingers take several steps to move.

The official checkers determine task success (TSR); completion can require a
monitoring period after the requested actions. Episodes stop once their result
is settled, or after 2500 policy steps. Stage labels and partial completion
scores (CSR) are private and never learner feedback. A native simulator exception
ends the episode as `error`. The assignment specifies decision, step, and time caps.
