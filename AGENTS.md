# Reproducing CaM experiments

Read [docs/reproducing.md](docs/reproducing.md) before preparing or running an
experiment. It describes the canonical run hierarchy, demonstration validation,
coding-agent filesystem access, and the separation of learning from held-out tests.

- Use the requested coding client/model and learning setting. Prepare through
  `cam prepare`; every new task gets its own timestamped run and generated README.
- Prepare demos for every task in each benchmark, not just pilot tasks. For the
  three additional benchmarks, retain up to 100 distinct training demos per task
  and start experiments with separate 25-demo subsets (`cam subset`). Record
  release/replay shortfalls; never pad counts or expose the full pool to a
  learner assigned fewer demos. RoboMME retains its existing 25-demo protocol.
- For `mode: "both"`, require an explicit `both_protocol: "v1"` or `"v2"`.
  V1 allows free demo/interaction choice; v2 instructs demo-first development and
  interaction for verification and focused exploration. Use v2 with at most 25
  attempts for current demo-plus-interaction experiments. With distinct training
  seeds, use all remaining attempts to verify and strengthen the solution even
  when it seems ready for testing; successful verification needs no code change.
  For MIKASA-Robo, RMBench, and RoboMemArena, assign 25 offline demos and 25
  distinct new training seeds with `training_seed_policy: "disjoint_once"`;
  exclude demo and test seeds, and never recycle failed attempts.
  Both use the same six-tool harness. See
  [docs/demo_interaction.md](docs/demo_interaction.md); do not relabel historical
  unversioned runs or add the historical v2 candidate machinery to this comparison.
- Keep model-generated code development inside the CaM tool interface. The host
  orchestrator may inspect the repository and repair infrastructure, but must not
  supply hidden simulator state, benchmark implementation, historical solutions,
  or test feedback to the learner.
- Preserve original datasets, logs, revisions, and scored evaluation outcomes.
  Never edit a prepared source snapshot to bypass its integrity checks. Prepare
  a new timetag after changing the harness, configuration, prompt, or data.
- Finish learning and freeze the policy before evaluating held-out episodes.
  Record actual scored/expected counts and distinguish infrastructure failures.
- RoboMemArena (user-requested, 2026-11-07): develop all 26 tasks in demo-only
  mode with exactly 25 replay-derived RGB-D demos per task (user revised the
  earlier 50-demo request), Claude Code / Opus
  5.5, zero training rollouts, then evaluate all 50 held-out episodes per task.
  Keep the earlier four-task 25-demo results separate; report official TSR.
  User-requested GPU sharing (2026-11-07): do not wait for the entire MIKASA
  preparation queue; admit RoboMemArena with a free-memory check and bounded
  workers (four replay workers on each of GPUs 0 and 1, eight total; increased
  from two per GPU at the user's request; at least
  16 GiB free on the assigned GPU before each native replay job).
  Cap the RoboMemArena pool at 50 successful replay-derived demos per task,
  stopping new verification once enough successes are available. Preserve
  earlier outcomes above the cap and report release/replay shortfalls.
  T04 diagnostic follow-up was stopped at the user's request after 16 completed
  retries recovered no additional successes (32 distinct successes remain).
  Preserve its separate logs and captures; do not restart the extra worker
  without a new user request. The main eight-worker replay preparation continues.
  Reuse the existing four-task RGB-D captures after integrity checks; copy them
  into the new preparation and replay only missing captures. Those files have
  25 demos per task; report successful-replay shortfalls explicitly. Preserve
  the ongoing replay pool, and use a fresh 25-demo all-task experiment/subset
  without changing historical 50-demo preparation plans or snapshots.
- Keep the generated run README and reproduction guide accurate when changing
  directory layout, tool exposure, or sandbox mounts.
- Diagnose weak policies with `scripts/dev_evaluate.py` on held-out training
  identities, never test episodes. Repair infrastructure and public interface
  documentation; do not add scorer-derived task contracts, success thresholds,
  solution hints, or unassigned diagnostic feedback to learner prompts. Keep
  `prompts/core.md` and `prompts/modes/` identical across benchmarks, re-prepare
  under a new timetag, and report every run.
- User-authorized exception (2026-11-07): only MIKASA-Robo
  `SeqOfColors3-VLA-v0` (both `mikasa_robo` and RGB-D `mikasa_robo_v2` adapters)
  receives the static cue-order clarification in
  `src/cam/prompts/clarifications/mikasa_robo_seq_of_colors3.md`, overriding the
  native goal's "in any order" wording. Record it in each new run's prompt,
  manifest, and README; keep shared core/mode prompts and historical runs intact.
- MIKASA-Robo v2 (`mikasa_robo_v2`, user-requested): RoboMME-style 256x256
  RGB-D/calibration/joints on the unchanged official env and scoring; 50 replayed
  demos and 50 `disjoint_once` interaction seeds per task. Its guide's general
  strategy paragraph (hypothesis families, no thresholds) is the one sanctioned
  exception to the no-scorer-hints rule; keep it task-agnostic. `robot.dry_run` is
  a shared opt-in kinematic query (no step, no collisions).
- Current MIKASA follow-up: the same ten selected tasks, demo-only with 50 RGB-D
  demos each, Claude Code / Opus 5.5, eight coding jobs, zero training rollouts.
  Ask the learner to infer memory representation and candidate completion
  conditions from demos, documenting uncertainty. The MIKASA-only offline
  `safety_check(state, action=None)` API is authorized: publish only sanitized
  robot geometry/controller parameters and pure robot math. It may compare
  supplied torques or estimate initial PD demand from joint state, velocity and
  action; distinguish URDF effort limits from simulator drive caps. It must not
  create a simulator, use hidden state, or claim actual future torque safety.
- Avoid GPU oversubscription: one RMBench evaluation per GPU (≤6 workers), not
  alongside MIKASA GPU-simulation workers. Archive resulting infrastructure
  failures with `cam recover-test` and retry identically.
- User-authorized RGB-D collection exception (2026-11-07): run five RMBench
  native-expert capture jobs on each of GPUs 0 and 1 (ten total; increased from
  two per GPU), sharing existing workloads after checking free memory before
  each new job. Preserve captures and active workers when resizing under a new
  scheduler timetag. This exception is for dataset generation.
