# Demonstrations + interaction Code-as-Policies

Write a general Python policy for the assigned task, using the assigned released expert demonstrations and your own counted training
interactions. Start fresh, without previous learned policies. No benchmark source,
test episodes, object poses or segmentation are available.
{{DEPTH_RESTRICTION}}

You have **at most 20 simulator attempts for this task in total**. The rollout tool
automatically cycles the task's training episodes in order. Three seeds receive at
most 7/7/6 attempts; MoveCube's five seeds receive at most four each. Failures,
timeouts, early stops and initialization errors count. No free probes or resets.
The initial observation is available only inside a counted attempt.

Only these model tools are available:
- `command`: isolated Bash/Python, writing in `/workspace`, reading your own
  `/feedback` and your assigned read-only `/demonstrations`. No network, host home,
  simulator implementation or other task data.
- `read_image`: return an image you created or captured to the coding model.
- `rollout`: execute your current policy on the next training seed, then save video,
  observations, executed actions, decisions and final status in `/feedback/RNNN_epN`.
- `usage`, `validate`, `snapshot`: accounting and offline code checks/history.

Use `python3.11` with numpy, scipy, Pillow, OpenCV, h5py and imageio. Use the command
tool to inspect files or create plots; use read_image to see those plots. Feedback
H5 datasets are indexed by recorded frame, with `phase` (0 demonstration, 1 execution)
and native `step`. `actions.npy` contains only YOUR executed policy commands.

Released expert recordings are a separate, read-only input in /demonstrations.
Read manifest.json and api_reference/trajectory.py. Use
`from cap_agent.trajectory import Trajectory, training_episodes` in offline scripts.
Each recording includes both {{SENSORS}} views, calibration, robot states, expert actions,
goal, phase and native step. Every recorded frame is available; select informative
windows/contact sheets yourself. Do not load all 50 videos into the prompt.
You only receive this task's three recordings (five for MoveCube). Clearly separate
its task-provided video demo from expert execution using traj.is_demo/exec_start.
Check action_alignment before associating an expert command with an observed pose.
No subgoal annotations, object ground truth or task source are included.
The demonstration files are NEVER mounted during live policy execution; generalize
visual/feedback rules from them rather than replaying an episode's trajectory.

Every command, tool response, code revision and development file is archived. The
driver also logs filesystem/network syscalls. Do not attempt to escape the sandbox,
use the internet, look for benchmark implementations or remove rollout evidence.

## Generated policy interface

Write `task.json`, `policy.md`, and `tools/*.py` inside `/workspace`.
`task.json` has `task`, `runtime: "python"`, and `memory: []` (or declared fields).
Memory fields use `{name,type,description}`; supported types are documented in
`api_reference/workspace.py`. Every field starts at None for every episode.

Define `perceive_decide(obs, goal, memory, last)` with a docstring, returning exactly:
`{"memory": {"every_declared_field": value}, "act": {"tool": "act_name", "args": {...}}}`.
The current task instruction is `goal`; `last` holds the previous tool's outcome.
`obs.frames` are the observations since the previous action and `obs.current` is
the latest one. Task-provided observations at reset follow the assignment's demo rule.

Each Frame provides only:
- `front_rgb`, `wrist_rgb`: uint8 RGB images.
{{DEPTH_FIELDS}}
- `front_K`, `wrist_K`: 3×3 camera intrinsics.
- `front_extrinsic`, `wrist_extrinsic`: 3×4 world-to-camera transforms.
- `tcp`: world XYZ metres and roll/pitch/yaw radians.
- `joints`, `fingers`: robot joint state and gripper finger widths.
- `phase`, `step`, `label`: observation timing.

There are NO object identities, object poses, masks, oracle actions,
task internals or evaluation seeds in the online policy interface. Released expert
actions are available only for offline development, in /demonstrations. Estimate scene state from {{SENSORS}}.

An `act_*` function takes `robot` first. `robot.now()` returns the latest Frame;
`robot.frames()` returns frames produced by this action. Each
`robot.step(x,y,z,roll,pitch,yaw,gripper)` advances one control step. Gripper -1 closes,
+1 opens; ignored for stick tasks. The action is a TCP target, not an instantaneous
teleport: use observed TCP feedback and gradual waypoints. Errors may consume native
step penalties. Each action tool may send at most 1500 commands, and each episode
has at most 40 decisions plus its native simulator step limit. Do not catch terminal
exceptions. Native terminal success/failure/timeout ends the rollout.

All generated Python goes in `tools/*.py`, without imports. `np`, `math` and safe
builtins are available. Private helpers start with `_`; public tools start with
`perceive_` or `act_`. Calls get fresh namespaces; keep history in declared memory.
The generated Python runs directly, without a language model during execution.

## Development procedure

1. Inspect all assigned demos at a useful overview level. Identify the goal,
   timing, perception cues and successful motion phases. Use measured {{SENSORS}}, robot
   feedback and expert actions to form an initial policy; record evidence by
   episode and frame interval in task_analysis.md. Do not reuse previous solutions.
2. Interleave experience sources throughout development. Before each counted
   attempt, state the hypothesis, what the demos already tell you, and what needs
   simulator feedback. Where possible, check perception on saved images or test
   action functions with an explicitly approximate mock before spending a rollout.
   A successful mock is not evidence of successful physical interaction.
3. Validate, snapshot and run the next round-robin training attempt. Inspect its
   video/observations and compare the relevant failure or uncertainty with expert
   demo windows. Revisit demonstrations after feedback instead of treating them
   as a one-time warm-up. They remain available throughout all R20 iterations.
4. Revise the general policy and run focused offline checks. In CHANGELOG.md log
   attempt number, training episode, demo frames consulted, observations, tested
   hypothesis, result and code change. State when a demo does not answer a question.
   Avoid identical blind retries; every trial should test a concrete uncertainty.
5. Call snapshot after each complete iteration. All commands, returned outputs,
   images, scratch scripts and policy versions are archived. Before attempt N+1,
   the driver freezes R_N; it cannot include later feedback. R5/R10/R20 therefore
   refer to policies developed with at most those numbers of simulator attempts.
6. Stop at 20 attempts or earlier if the policy is robust across the training
   seeds. You may make final offline edits and validation after the last attempt.
   Report remaining limitations and actual attempts used. Never evaluate test data.

The driver seals the final policy. There is no inference-time language model.
