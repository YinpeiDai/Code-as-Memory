## MIKASA-Robo data and simulator protocol

Under `api_reference/runtime/`, read `observations.py` for `Frame` and
`Observations`, and `robot.py` for `Robot.now()`, `frames()` and `step()`.
When demonstrations are supplied, `trajectory.py` documents `Trajectory` and
`training_episodes`. These are reference source files; offline scripts import
the installed `cam.runtime` package as shown below. Generated policy functions
use their supplied `obs` and `robot` arguments without imports.

Only your selected extracted H5 is mounted at `/demonstrations` when allowed.
Use `from cam.runtime.trajectory import Trajectory, training_episodes` in offline
command scripts. A trajectory exposes `T`, `goal` (the language instruction),
`is_demo` (always false), `action_alignment`, `step_semantics` and per-frame
`base_rgb`, `wrist_rgb`, `tcp`, `gripper_opening` and `action`, the recorded
expert command in `robot.step` units; `traj.get(name, t)` reads frame `t`.
Frame t is the observation after t commands; `action[t]` was issued after frame
t. Recordings end at the first native success; the final observation is not
recorded. Recorded frames can differ slightly in rendering from online frames.
Expert H5 is absent during policy execution. The trajectory reader lists
offline demo IDs; the assignment separately lists the allowed simulator
training identities and whether they are disjoint from demonstrations.

Online `obs.frames` contains all frames since the previous action, one per native
control step; `obs.current` is latest. Every frame has phase `exec`; any memory
cue is part of the observed scene, with no flag marking it. The goal is the
native language instruction. Cameras are 128x128, without depth or calibration.
No object poses, task cues, rewards, success labels or simulator internals are
available to the coding model or generated policy.

`robot.now()` returns the latest frame. `robot.frames()` returns the action's
observed frames. `robot.step(dx,dy,dz,drx,dry,drz,gripper)` sends one native
control step with the semantics declared below. Offsets are relative to the
measured pose at each step, and one step moves the TCP only part of the
commanded offset; close the loop on measured `tcp`. In some tasks commands are
replaced by zero during an initial phase; observations continue. The episode
ends with `success` at the first native success signal, `fail` if the task
terminates otherwise, or `timeout` at the task's native step limit. CaM uses the
official MIKASA-Robo-VLA environment, wrappers and seeds; scoring is unchanged.

`robot.dry_run(..., mode="absolute" or "delta")` can check a candidate's next
arm joint target without stepping. Read `api_reference/runtime/mikasa_robo/dry_run.md`
for units, returned values, and limits of this mechanical check.
