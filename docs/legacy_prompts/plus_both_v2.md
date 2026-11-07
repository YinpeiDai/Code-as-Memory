# Demonstrations + interaction v2

Build a general Python policy for the assigned task, starting fresh. Learn mainly
from expert demonstrations; use simulator rollouts to verify the solution or
resolve specific gaps. You decide when interaction is useful, without a required
demo/trial alternation. The final policy runs directly, without an inference-time LLM.

## Evidence and access

Inspect every assigned recording in read-only `/demonstrations`. Read its goal,
separate task video-demo frames from expert execution using `is_demo/exec_start`,
and check `action_alignment`. Use `cap_agent.trajectory.Trajectory` and
`training_episodes`; see `manifest.json` and `api_reference/trajectory.py`.

The 50 released demos use the same training seeds as interaction: three episodes
per task, five for MoveCube. Within your task, demo episode N matches rollout
episode N. Note questions against their demo episode/frame intervals, finish the
initial code solution offline, then verify those questions through interaction.
Trials cycle the aligned episodes automatically; you cannot choose their order.

Inspect front and wrist RGB-D, correlating visual events with expert actions,
robot states and camera calibration. Use `command` to extract PNGs/contact sheets
into `/workspace`, then `read_image` to view them. Sample for an overview and
inspect all frames around important events as needed. Reuse recordings without
limit; offline analysis consumes no simulator attempts.

Do not replay recorded actions or hard-code a demonstration's objects,
coordinates, counts or timing. Use current observations and the current goal.
Separate observed facts from assumptions: shared training properties need not be
task invariants. Keep perception, instruction parsing and control testable, with
progress checks and bounded recovery.

Use only cap MCP tools. `command` provides isolated Bash/Python with python3.11,
numpy, scipy, Pillow, OpenCV, h5py and imageio. Read only your workspace, supplied
API references, assigned demonstrations and your own `/feedback`. No internet,
simulator source, test data, other studies, previous solutions or object ground
truth is available. Expert recordings/actions are offline development inputs;
they are not mounted during live execution. The assignment specifies whether
the task's own video demo is supplied at reset.

Never initialize a simulator directly, including for previews or debugging.
Only the counted `rollout` tool provides simulator access after `begin_interaction`.
Commands, outputs, images, files, revisions, access traces and selections are
archived. Never delete evidence; call `snapshot` after each coding iteration.

## Development loop

1. **Analyze demos.** In `task_analysis.md`, record useful episode/frame intervals,
   task logic, measured geometry/motion, assumptions and unresolved questions.
2. **Check a solution offline.** Write tests in `scratch` and run `offline_check`
   covering `instruction`, `perception` and `control`. Exercise actual generated
   code through the documented toolkit/sandbox; assert meaningful outcomes.
   Checks run read-only, with temporary outputs in `/tmp`. One command may cover
   several kinds. State mock assumptions; mock success is not physical success.
3. **Freeze R0.** Register `candidate(role="solution")` with a summary and passing
   check IDs for its executable behavior. Then call `begin_interaction` with its
   ID, evidence from every assigned demo and remaining uncertainties. R0 is
   the completed demo-derived solution, checked offline but unverified in simulation.
4. **Request useful verification.** Call `rollout` with a frozen `candidate_id`,
   `purpose` (verification/diagnostic/refinement/confirmation), `hypothesis`,
   `evidence`, `offline_gap` and `expected_outcomes`. Explain what each plausible
   outcome would change. The tool executes that snapshot, not working-file edits.
   Diagnostic `probe` candidates may omit offline checks, must use purpose
   `diagnostic`, and cannot be selected for evaluation.
5. **Review and repair.** Inspect feedback, then call `review_rollout` with the
   observation, conclusion and next action before another trial or selection.
   Revisit relevant expert windows. Prefer focused changes that preserve working
   behavior; use saved evidence before repeating a trial. Register revised code
   as a new candidate, with current checks for any solution.
6. **Select explicitly.** Use `select_candidate` with a training-only rationale.
   Keep the initial solution, current selection and experiments separate. A
   replacement must be a checked solution with simulator evidence, cover every
   seed tested for the current selection, and preserve its successful seed outcomes.
   This uses each candidate's latest outcome per seed. Ties require justification;
   partial coverage does not establish generalization. Untested edits never
   automatically replace the selected policy.

`usage` reports budget, selected IDs and per-seed evidence. `restore_candidate`
restores editable policy files without changing the selection. Comments/docstrings
and policy prose preserve executable evidence; behavior/configuration changes do not.
`validate` checks the policy interface without a reset.

Feedback lives in `/feedback/RNNN_epN`: video, RGB-D/calibration/robot-state H5,
actions and results. H5 indices are recorded frames; `phase=0` is task demo and
`phase=1` execution. Native steps are separate. Actions here are yours, not experts'.

## Budget and completion

At most **20 attempts total per task**, on automatically cycled training seeds.
Probes, confirmation, initialization failures, timeouts and early stops all count.
The final seed cycle is reserved for `confirmation` using one unchanged solution
candidate ID: the last three attempts normally, or five for MoveCube. Plan this
candidate in advance, review its results and select it only if justified.

The driver seals the current selection before the next attempt: R5 contains at
most feedback from attempts 1–5, before attempt 6. Never use test results for
selection. You may stop early; the last selection carries forward with actual
attempts recorded. Finish with a valid selected policy and a short report of its
ID, supporting evidence, uncertainties, seed coverage and unused budget.

## Python policy interface

Write `task.json`, `policy.md` and `tools/*.py` in `/workspace`. Keep the assigned `task`, set
`runtime="python"`; declare memory fields as documented in `api_reference/workspace.py`.
Every field starts at None each episode. No cross-episode state is available.

Define `perceive_decide(obs, goal, memory, last)` with a docstring, returning exactly:
`{"memory": {"every_declared_field": value}, "act": {"tool": "act_name", "args": {...}}}`.
`obs.frames` holds observations since the last action, `obs.current` the latest;
`last` is the previous action outcome and `goal` the current instruction.

Frames expose front/wrist RGB (uint8), aligned depth (H×W int16 millimetres along
the optical axis; 0 unknown), 3×3 intrinsics, 3×4 world-to-camera extrinsics, TCP
XYZ metres/roll-pitch-yaw radians, joints, fingers, phase, step and label.
See `api_reference/observations.py` for field names.

An `act_*` tool takes `robot` first. Use `robot.now()`, `robot.frames()` and
`robot.step(x,y,z,roll,pitch,yaw,gripper)`. Each step advances one control step;
-1 closes the gripper, +1 opens it (ignored for stick tasks). TCP targets require
feedback and gradual waypoints; movement is not instantaneous. Errors may consume
native step penalties. Limits: 1500 commands per action, 40 decisions per episode,
plus the native task step limit. Never catch terminal success/failure/timeout exceptions.

No imports in generated `tools/*.py`: `np`, `math` and safe builtins are available.
Private helpers start with `_`; public functions with `perceive_` or `act_`.
Calls get fresh namespaces; preserve within-episode history in declared memory.
