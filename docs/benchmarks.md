# Benchmark adapters

RoboMME and MIKASA-Robo share the isolated coding harness, logs and frozen-policy
evaluation. MIKASA uses a separate simulator environment. Other registered
benchmarks remain planned in this release.

| Benchmark | Tasks / held-out episodes per task | Evidence and selection | Policy action | Setup |
|---|---|---|---|---|
| RoboMME | 16 / 50 | Modern H5; 25 demos balanced 9/8/8 by difficulty; RGB-D | Absolute TCP XYZ/RPY + gripper | [First run](getting_started.md) |
| MIKASA-Robo | 90 / 50 | Lossless RLDS; 25 uniformly sampled identities; RGB + proprioception | 7-D normalized delta pose + gripper | [Environment](../envs/mikasa_robo/README.md) |
| MIKASA-Robo v2 (`mikasa_robo_v2`) | 90 / 50 (same official env, seeds, limits, scoring) | Released expert commands replayed under each collector's simulator setup, recorded with the v2 sensors; 50 per task in seeded random order, native-success replays only; RoboMME-style 256x256 RGB-D, calibration, TCP, joints, fingers; `robot.dry_run` | 7-D normalized delta pose + gripper | [MIKASA-Robo v2](#mikasa-robo-v2) |
| Benchmark | Required raw downloads | Extracted tasks | Dataset |
|---|---|---|---|
| MIKASA-Robo | 3,665/3,665 pinned files | 90/90 | `data/mikasa_robo_25` |
| Benchmark | Released training demos/task | Prepared pool target | Initial assignment |
|---|---:|---|---|
| RoboMME | 100 | Existing balanced 25-demo protocol | 25 |
| MIKASA-Robo | 250 | 100 | 25 |

## MIKASA-Robo v2

`mikasa_robo_v2` keeps MIKASA-Robo-VLA's official environment, wrappers, control
mode, seeds, step limits and success signal, and changes only the observations:
`obs_mode="rgbd"` with both cameras at 256x256. A pass-through wrapper reads the
raw observation beneath the canonical wrappers, which drop depth and calibration.
Policy frames use RoboMME's names: `front_*` (MIKASA `base_camera`) and `wrist_*`
(`hand_camera`) RGB, int16 optical-axis depth (mm), K and world-to-camera
extrinsics, plus `tcp`, `joints` (with documented Panda limits) and `fingers`.
Replaying TakeItBack training rollouts through v2 reproduces the v1 trajectories
exactly; the sensors do not change physics.

**Demonstrations** (`data/mikasa_robo_v2_50`, built by
`cam.benchmarks.mikasa_robo_v2_data`): the release has 128x128 RGB only, and
single-environment replay at the official evaluation settings does not reproduce
it (2/25 pilot successes). The collectors' own setups do reproduce it: frame 0
pixel-identical, TCP within millimetres, 25/25 successes.
- MP tasks: one environment, default CPU physics, the motion-planning
  collector's wrapper chain.
- PPO tasks: batched GPU environments of 10 with seeds `10k..10k+9` and the
  collector's wrappers, with batch-mates replaying their released commands.

Each task takes released episodes in a seeded random order and keeps replays
that reach native success, until 50. Rejections and shortfalls are recorded in
the manifest. Workers use `CAM_REPLAY_GPUS` (default `0`) and
`CAM_REPLAY_THREADS` (default 3).

**Interaction.** `configs/mikasa_robo_v2/both_v2_claude_code.json` sets 50
attempts on 50 distinct training seeds (`disjoint_once`), excluding the 50 demo
identities and the test seeds.

**Robot checks.** `robot.dry_run(..., mode="delta" or "absolute")` checks the
next native controller joint target without advancing an active episode; an
absolute request is converted to one bounded command toward the pose. It does
not check collisions or whole-path reachability. Calls are logged in
`dry_runs.jsonl`. The older v2 call without a mode retains its absolute IK query
for compatibility. See the [live API](../src/cam/benchmarks/mikasa_robo_api/api.md).

For demo-only development, `cam.runtime.mikasa_robo.safety_check(state, action)`
uses only supplied state and public robot math. It reports position margins,
supplied torque margins and, when velocities and a target are available,
estimated initial PD demand. It does not predict actual future torque safety.
See the [offline API](../src/cam/benchmarks/mikasa_robo_api/safety_check.md) and
[50-demo setup guide](mikasa_v2_setup.md).

**Guide.** The v2 guide adds step-budget advice, the dry-run documentation, and
one general strategy paragraph, added at the user's request. It names hypothesis
families that MIKASA success tests commonly use (proximity-judged contact,
settling, side effects, press-release-lift, timing, completeness) and asks the
learner to bound them from demos and discriminate them with rollouts. It contains
no task-specific thresholds or contracts.

## All-task demo pools

Prepare demonstrations for every task, then use `cam subset` to assign fewer
examples. Never pad missing recordings. Original MIKASA uses RGB-only RLDS;
v2 uses replay-derived RGB-D. The final v2 pool contains 90 tasks with 50 demos
each. See [setup with shared NFS](mikasa_v2_setup.md).

The learner receives public sensors/actions and its assigned examples only.
Scorer code, hidden state, test identities and historical solutions stay private.
