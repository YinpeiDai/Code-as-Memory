## MIKASA-Robo data and simulator protocol

Under `api_reference/runtime/`, read `observations.py` for `Frame` and
`Observations`, and `robot.py` for `Robot.now()`, `frames()`, `step()` and
`dry_run()`. When demonstrations are supplied, `trajectory.py` documents
`Trajectory` and `training_episodes`. These are reference source files; offline
scripts import the installed `cam.runtime` package as shown below. Generated
policy functions use their supplied `obs` and `robot` arguments without imports.

Only your selected H5 is mounted at `/demonstrations` when allowed. Use
`from cam.runtime.trajectory import Trajectory, training_episodes` in offline
command scripts. Each demonstration is a released MIKASA-Robo expert episode
replayed in the simulator and recorded with the online sensors: per-frame
`front_rgb`, `wrist_rgb`, `front_depth`, `wrist_depth`, `front_extrinsic`,
`wrist_extrinsic`, `tcp`, `joints`, `fingers` and `action` (the expert command
in `robot.step` units), plus static `front_K` and `wrist_K`; `traj.get(name, t)`
reads frame `t`. Frame t is the observation after t commands; `action[t]` was
issued after frame t. Recordings end at the first native success; the final
observation is not recorded. Expert H5 is absent during policy execution. The
assignment lists the demonstration IDs and, separately, the simulator training
identities and seed schedule. They may be the same identities or disjoint,
according to the pinned assignment; the learner cannot select or override seeds.

Online `obs.frames` contains all frames since the previous action, one per native
control step; `obs.current` is latest. `front_*` is a fixed camera viewing the
scene and robot; `wrist_*` is mounted on the hand. Both cameras provide native
256x256 RGB and aligned 256x256 depth on every observation, during development
and held-out evaluation. Images are not resized or temporally subsampled by CaM.
Depth is int16 millimetres
along each camera's optical axis (0 unknown), pixel-aligned with its RGB image;
`K` is the OpenCV intrinsic matrix and `extrinsic` maps world points to camera
coordinates (x right, y down, z forward). `tcp` is in the same world frame. Every
frame has phase `exec`; any memory cue is part of the observed scene, with no flag
marking it. The goal is the native language instruction. No object poses, task
cues, rewards, success labels or simulator internals are available.

`robot.now()` returns the latest frame. `robot.frames()` returns the action's
observed frames. `robot.step(dx,dy,dz,drx,dry,drz,gripper)` sends one native
control step with the semantics declared below. Offsets are relative to the
measured pose at each step, and one step moves the TCP only part of the
commanded offset; close the loop on measured `tcp`. In some tasks commands are
replaced by zero during an initial phase; observations continue. The episode
ends with `success` at the first native success signal, `fail` if the task
terminates otherwise, or `timeout` at the task's native step limit (stated in the
action description). Every observed step counts toward that limit, including
cue and delay phases spent waiting: measure in the demonstrations how many steps
each phase takes, budget the whole episode, and prefer decisive motions that
finish with margin over slow creeping approaches or long blind waits.

### Kinematic dry run

`robot.dry_run(x, y, z, roll, pitch, yaw)` checks an absolute TCP pose (world
frame, same convention as `tcp`) against the Panda arm's kinematics and joint
limits from the current measured joints, without moving and without using a step.
It returns a dict: `reachable` (a joint solution within limits exists),
`path_clear` (the straight TCP path from the current pose stays within limits),
`path_blocked_at`, `joint_solution`, `joint_margin`, `joints_at_limit`,
`nearest_limits`, `exact_solution_violates`, `position_error_m` and
`rotation_error_rad` of the closest pose within limits,
`position_reachable_with_rpy` (an orientation that reaches the position when the
requested one cannot) and a `message`. It checks kinematics only, not
collisions. It takes about 0.01-0.1 s for reachable poses and about 1 s
otherwise, so use it when choosing targets rather than at every step. The
controller does not avoid limits: commanding offsets toward an unreachable pose
drives joints into their limits, where the arm stalls or moves erratically.
Joint limits are listed with the `joints` sensor.

### Approaching MIKASA-Robo tasks

Each task is scored by the simulator's own success test, which is not shown to
you. Demonstrations show one way to satisfy it, and experts usually stop at the
first moment it holds, so they reveal little about its exact conditions. In this
benchmark the conditions are often more specific than the instruction suggests.
Treat the following as hypotheses to test, not as facts about your task:
- **What counts as an interaction.** A "touch", "reach" or "press" may be judged
  by where the TCP is relative to some reference point of the object, not by
  physical contact. Which point counts, how close is close enough, and from which
  direction the approach works are all open questions.
- **Settling.** Success may be checked only while the arm is at rest. Arriving
  and moving on, or jittering around the target, may never register.
- **Side effects.** Disturbing other objects, displacing the target, or touching
  a wrong candidate may make success impossible without ending the episode.
- **Multi-step actions.** A press may count only after a release and a clear
  lift. Repeated, merged or premature actions may be ignored or fatal. Order may
  matter even when the instruction does not stress it.
- **Timing.** Cue, delay and response phases. Acting early may be ignored or
  penalized, and step counts must start from the right observable event.
- **Completeness.** Every target, item or checkpoint may need handling before the
  final action counts.

Use the demonstrations to bound these hypotheses: measure approach heights,
distances and directions, pause lengths, lift heights and phase durations.
When the assigned mode permits interaction, design rollouts that discriminate
between remaining candidates. Change one factor at a time, such as settle duration, approach point,
lift height or timing, and compare outcomes. A timeout carries no reason, so make
the policy record measured quantities and stage progress in memory and in act
outcomes; a failed rollout should then show which stage failed and why.
Represent task memory explicitly. Record cue observations as soon as they are
visible, since they may vanish, together with the step and your confidence, and
track per-stage progress so the policy never repeats or skips a stage.
Use `robot.dry_run` to confirm that chosen targets and approach poses are
reachable before committing steps to them.

For demo-only development, use the recordings and synthetic checks to compare
hypotheses; no simulator rollout or live dry-run is available. The separate
`cam.runtime.mikasa_robo.safety_check` API accepts supplied robot state and
candidate actions entirely offline. Use it to check joint-position margins and,
when the required inputs are available, estimated drive-torque demand. Read
`api_reference/runtime/mikasa_robo/safety_check.md`; torque demand is not measured
contact force or a guarantee of stability.

CaM uses the official MIKASA-Robo-VLA environment, wrappers, seeds and step
limits; only the cameras are rendered at 256x256 with depth and calibration,
which does not change dynamics or scoring.
