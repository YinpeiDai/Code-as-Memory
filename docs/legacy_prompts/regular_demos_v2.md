# Original RoboMME: demonstrations-only Code-as-Policies

Start a fresh general Python policy. Your only task evidence is the assigned
extracted training H5 file in read-only /demonstrations. The assignment lists the
number and original episode IDs. There are no host file tools, browser, network,
benchmark source, full recordings, test data, other task datasets, previous
solutions or other agents' policies.

ZERO simulator attempts are allowed, including previews, calibration runs,
debugging resets and direct environment creation. The rollout tool is disabled.
No language model is used during generated policy execution.

## Evidence and development

Use `from cap_agent.trajectory import Trajectory, training_episodes` in command
scripts. Inspect every assigned episode: its goal, difficulty, task-video phase,
both camera views and expert execution. Summarize variation across the full
selection, then inspect relevant events and frames more closely as useful.
You may reuse the fixed recordings without limit.

Infer a policy responding to the current goal and observations. Do not replay
recorded action sequences, embed training images, or hardcode a seed-to-solution
lookup. Derived general constants and algorithms may be saved in Python.

`cap.command` runs Bash/Python in writable /workspace, with the read-only assigned
demonstrations. `cap.read_image` displays derived images. `cap.usage`,
`cap.snapshot` and `cap.validate` provide accounting, source snapshots and offline
validation. All commands, outputs, images, revisions and API usage are archived.
Keep analysis artifacts in /workspace/scratch. Write task_analysis.md and update
CHANGELOG.md after complete iterations. Use offline checks and synthetic inputs;
finish with validated code and a candid account of its evidence and limitations.

Before finishing, check the generated perception and goal parsing against
recorded examples from every assigned episode where applicable. Check geometry
against the recorded RGB-D, calibration and robot state when depth is supplied.
Use synthetic inputs to exercise decision and control logic without a simulator.
Record which episodes, frames and behaviors were checked, the observed errors,
and what remains unverified. Then produce the general code solution and run
offline policy validation. Offline checks do not establish simulator success.

## Recorded data format

Each `Trajectory(task, episode)` is a context manager. `T`, `goal`, `difficulty`,
`is_demo` and `exec_start` describe the recording. `is_demo` separates the task's
RGB video instruction from expert execution. Every recorded frame is retained.

- `front_rgb(t)`, `wrist_rgb(t)`: uint8 RGB images.
{{DEPTH_RECORDED}}
- `front_K`, `wrist_K`: recorded 3x3 camera intrinsics.
- `front_extrinsic(t)`, `wrist_extrinsic(t)`: recorded 3x4 world-to-camera matrices.
- `tcp(t)`: recorded native TCP XYZ metres and XYZ roll/pitch/yaw radians, using
  the same TCP frame as online observations. This is a measured pose.
- `tcp_quaternion_wxyz(t)`: the recorded TCP orientation as wxyz quaternion.
- `joints(t)`, `fingers(t)`: seven joint positions and two finger widths.
- `eef_action(t)`: recorded Cartesian command [XYZ, RPY, gripper].
- `waypoint_action(t)`: recorded waypoint command, or None for missing waypoints.
- `joint_action(t)`: seven joint targets and gripper command.
- `step(t)`: recording frame index, not a native control-step counter.
- `events()`: measured gripper-command transitions, without oracle labels.

Read `action_alignment` before pairing actions with frames. The source declares
no action/observation alignment; correspondence is unspecified. Missing waypoint
values are preserved, not synthesized. There is no segmentation, object
pose, oracle subgoal, completion label or simulator internals in this data.
The demonstrations mount is absent during policy execution.

## Generated policy interface

Write task.json, policy.md and tools/*.py. task.json contains task, runtime
"python", and memory declarations [{name,type,description}]. See
api_reference/workspace.py for supported memory types; fields start as None.

Define `perceive_decide(obs, goal, memory, last)` with a docstring and return
exactly `{"memory": {"every_declared_field": value}, "act": {"tool": "act_name", "args": {...}}}`.
`obs.frames` contains observations since the last action, `obs.current` is latest,
and `last` contains the prior tool outcome. Each decision gets a fresh namespace;
retain episode history in declared memory.

Online Frame fields are front_rgb/wrist_rgb, {{DEPTH_ONLINE}} front_K/wrist_K,
front_extrinsic/wrist_extrinsic, tcp, joints, fingers, phase ("demo" or "exec"),
step and label. Task-provided RGB instruction frames precede execution when the
task has them. No masks, object IDs/poses, subgoal labels or internals are
available online.

An act_* tool takes robot first. `robot.now()` is the latest Frame and
`robot.frames()` returns frames produced by this action. Call
`robot.step(x,y,z,roll,pitch,yaw,gripper)` for one native control step with an
absolute world TCP target. Gripper -1 closes and +1 opens; stick robots ignore
it. Use gradual motion and observed TCP feedback. IK can reject unreachable
targets. Limits: 1500 step calls per action tool, 40 decisions per episode and
native max_steps=1300. Terminal success/failure/timeout/error ends the episode;
do not catch terminal exceptions.

Generated tools/*.py contains no imports. np, math and safe builtins are supplied.
Private helpers start with _; public functions start with perceive_ or act_. Read
api_reference/tools.py and sandbox.py for the supported language.
