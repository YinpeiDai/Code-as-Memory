# Coding agent: Code-as-Policies baseline

Generate a Python policy for the single assigned RoboMME-Plus task. At evaluation,
your Python code runs directly. There is no Haiku or other language model deciding
actions. Generalize to unseen instructions, appearances, layouts and timing.

## Allowed experience

Your assignment states the adaptation category, allowed training episode indices,
and R, the maximum number of simulator attempts **for this task in total** (up to 50).
R is not per seed. Every failure, timeout, early stop, crash, repeated seed, and
extra reset counts. Simulator debugging and controller probes share the budget.

- `demos`: read and reuse only the task's released demonstrations; no fresh
  simulator interaction, including resets, previews and debugging (R0).
- `interaction`: use only simulator observations from the permitted training
  seeds; do not open training HDF5 recordings or expert actions.
- `both`: use the recordings and at most R training simulator attempts.

Read only your workspace and the public harness API reference supplied with your
assignment, plus `Trajectory` for your own task when demonstrations are allowed.
Never read benchmark task implementations, metadata files, oracle information,
released solutions, other workspaces, previous studies, or test results. Never
write training-specific coordinates, colors, orders, timing or counts into code.
Do not download solutions or query an external policy model.

Write only task.json, policy.md, tools/*.py, task_analysis.md, CHANGELOG.md and your
own scratch files in this workspace. Do not modify study_task.json, the budget
ledger, harness, environment variables that identify the study, or simulator
configuration. Do not remove or overwrite rollout evidence. Run foreground jobs.

## Python interface

`task.json` contains `{"task": "<Task>", "runtime": "python", "memory": [...]}`.
Memory field types and names follow workspace.py. `policy.md` documents the code;
it is not an inference-time prompt. All Python goes in `tools/*.py`.

Define `perceive_decide(obs, goal, memory, last)` with a docstring. Return exactly:

```python
{"memory": {"every_declared_field": value},
 "act": {"tool": "act_name", "args": {"argument": value}}}
```

The goal is the current episode's instruction. `obs.frames` contains every frame
since the last act; at the first decision it includes the task's own video demo,
if provided, followed by the first execution observation. `obs.current` is the
latest frame. Both cameras, depth, calibration and robot state are available as
documented in observations.py. This task-provided demo is an evaluation observation,
distinct from the released expert training recordings.

`last` is None initially, otherwise the previous act's name and textual outcome.
Keep within-episode history in the returned memory. Every memory field starts at
None for each episode, and every function call gets a fresh namespace. There is
no cross-episode state. Decide in Python using the instruction, images and memory.

An `act_*` function takes `robot` first, followed by your chosen arguments. It may
use robot.now(), robot.frames(), and robot.step(x,y,z,roll,pitch,yaw,gripper) to run
closed-loop motion. Check its effect and return measured outcomes. The harness
stops at the official success, failure or timeout. Act calls have a 1500-step
safety cap; split longer routines across decisions. At most 40 decisions are made.
Do not catch the environment's terminal exception. Specific recoverable motion
errors may be handled. There are no imports: np, math and safe Python builtins are
available. Helpers begin with `_`; public functions are `perceive_*` or `act_*`.

## Development

1. Inspect your allowed experience and write task_analysis.md. In interaction-only
   mode, the initial training rollout is also the opportunity to observe the task.
2. Replace the deliberately empty starter policy, then run the assignment's
   validation command. Offline validation does not consume simulator attempts.
   After every complete coding iteration, run the assignment's snapshot command.
   This saves the code even when it has not yet been validated. Validation and
   rollout boundaries also save revisions; intermediate file edits are watched.
3. If R > 0, choose a training seed and run the provided rollout command with a
   fresh label. Inspect the saved video, decisions and results; revise the code.
   Direct simulator probes must also use cap_agent.env.make_env(task, episode,
   'train'), which uses the same ledger. Never import the benchmark builder directly.
   Run one attempt at a time. Finish analyzing its feedback and revising the policy
   before requesting the next attempt. The harness freezes R_N immediately before
   attempt N+1 begins; it cannot include feedback from the later attempt.
4. Record each revision and its evidence in CHANGELOG.md. Check the usage command
   before requesting more rollouts. Stop simulator work when the budget is exhausted;
   you may finish offline edits and validation. There are no free retries.
5. Finish with a valid policy and a short report of evidence, limitations and usage.
   Do not run the test split. The experiment driver freezes and evaluates the policy.

The blank starter is an interface example, not a task-solving policy. Categories
and replicates start fresh. Budget checkpoints within one run share its development
history. Do not import released solutions or policies from other runs/categories.
