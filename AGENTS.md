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
- Earlier MIKASA follow-up: the same ten selected tasks, demo-only with 50 RGB-D
  demos each, Claude Code / Opus 5.5, eight coding jobs, zero training rollouts.
  Ask the learner to infer memory representation and candidate completion
  conditions from demos, documenting uncertainty. The MIKASA-only offline
  `safety_check(state, action=None)` API is authorized: publish only sanitized
  robot geometry/controller parameters and pure robot math. It may compare
  supplied torques or estimate initial PD demand from joint state, velocity and
  action; distinguish URDF effort limits from simulator drive caps. It must not
  create a simulator, use hidden state, or claim actual future torque safety.
- User-requested MIKASA demo+interaction guidance (2026-11-07): new v2-protocol
  runs for both MIKASA adapters include the task-independent supplement
  `src/cam/prompts/benchmarks/mikasa_robo_both_v2.md`. Build from demos first,
  then use counted training rollouts for motion verification and further
  revisions, including rotation-command oscillation and settling checks.
  Explain offline `safety_check` versus live `robot.dry_run` without claiming
  dynamic safety. Keep shared core/mode prompts identical across benchmarks;
  supply no scorer thresholds, prior diagnostic feedback or task-specific
  controller gains. Pin the supplement in the manifest, prompt and run README;
  use fresh timetags and preserve historical prompts and policies.
- Current MIKASA development (2026-11-07): the same ten selected tasks, 50 RGB-D
  demos plus all 50 training attempts on those exact demo identities, Claude Code / Opus 5.5,
  eight coding jobs, and CPU physics for both development and later evaluation.
  Use `configs/experiments/mikasa_v2_cpu_50_demoseeds_10tasks.json` with
  `training_seed_policy: "demo_cycle"`, count 50 and budget 50 (one pass, no repeats).
  The harness selects the next identity, rejects seed/episode overrides and
  blocks attempt 51; preparation checks exact demo/rollout identity equality and
  excludes test seeds. This supersedes new-seed selection for this follow-up.
  GPU rendering
  stays 256x256 RGB-D. Assigned demos may be reread/analyzed without an access
  count limit or simulator charge. Use every remaining rollout for verification
  even when the policy appears ready. Explore release/lift, timing and motion
  hypotheses using assigned evidence only; keep paths within checked joint
  margins without prescribing hidden thresholds. Start fresh blank policies;
  keep previous demo-only and GPU results separate. The development launcher
  does not start held-out evaluation.
- Remaining MIKASA tasks on another machine: use
  `configs/experiments/mikasa_v2_cpu_50_demoseeds_remaining80.json` (exactly the
  other 80 tasks). Replay data are at
  `/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/data/mikasa_robo_v2_50`:
  `manifest.json` and `h5/record_dataset_<Task-ID>.h5`, 90 tasks / 4,500 episodes.
  Keep CPU physics, 50 demos and one rollout per demo seed, Claude Code / Opus
  5.5. The latest user request assigns ten concurrent coding jobs to these 80
  tasks and automatic full held-out evaluation after each task finishes learning
  and freezes at R050. Use `--evaluate-on-freeze --evaluation-workers-per-gpu 2`
  with GPUs 0 and 1: four evaluation episode workers total, one task's test
  evaluation at a time, alongside at most one training simulator per GPU.
  Evaluate all 50 test episodes per task (4,000 total); keep test feedback out
  of every learner. New preparations use the concise MIKASA prompt;
  the active ten-task batch retains its original prompt. Record the second
  machine's batch tag in `docs/mikasa_v2_progress.md`; do not launch duplicates here.
- User-requested MIKASA artifact storage: this checkout's `runs/`, `outputs/`,
  `data/`, and `sites/` link to matching directories under
  `/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/mikasa_robo`.
  Keep new development batches, workspaces, policies, logs, rollouts, evaluation
  outputs, generated data and dashboard caches there. Preserve the existing
  replay pool at `/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/data/mikasa_robo_v2_50`.
  Source code and Python environments stay local. See the setup guide for the
  storage layout; do not replace existing artifact directories when configuring
  another checkout.
- Avoid GPU oversubscription: one RMBench evaluation per GPU (≤6 workers), not
  alongside MIKASA GPU-simulation workers. Archive resulting infrastructure
  failures with `cam recover-test` and retry identically.
- User-authorized RGB-D collection exception (2026-11-07): run five RMBench
  native-expert capture jobs on each of GPUs 0 and 1 (ten total; increased from
  two per GPU), sharing existing workloads after checking free memory before
  each new job. Preserve captures and active workers when resizing under a new
  scheduler timetag. This exception is for dataset generation.
- Current RMBench follow-up (user-requested, 2026-11-07 local): use up to 50
  successful regenerated RGB-D demos per task across all ten tasks, with the
  shared demo-only prompt and zero training rollouts. This replaces the proposed
  25-demo assignment for this follow-up. Preserve the actual shortfalls: 49 for
  battery_try, 49 for blocks_ranking_try, 48 for observe_and_pickup, and 50 for
  each of the other seven tasks (496 total). Use fresh RGB-D preparations and
  keep earlier RGB experiments and the separate 25-demo subset intact.
  Latest user-authorized held-out follow-up: as soon as a task's own learner
  finishes and its policy is frozen, evaluate all 100 test episodes for that
  task; do not wait for the other learners (1,000 episodes total across ten tasks).
  Use one RMBench evaluation per GPU with at most six episode workers, avoiding
  MIKASA GPU simulation. Preserve scored outcomes, archive infrastructure
  failures with `cam recover-test` before identical retries, and report complete
  scored/expected coverage. Never send test feedback back to a learner.
  User-requested local-storage/progressive follow-up: assemble the new task H5s
  on the local `/data` SSD, preserving original NFS captures and interrupted
  assembly evidence; copy finished data to NFS later with the user. Start each
  task once its own assigned demos are ready instead of waiting for all ten.
  Inspect five seeded random successful episodes per task in detail, retaining
  file hashes, schema, success and train/test identity checks for every demo.
  Do not repeat native simulator replay just for these file inspections.
- User-requested RMBench comparison (2026-11-07 local): after all ten current
  up-to-50-demo RGB-D demo-only learners finish and all 1,000 held-out tests are
  scored, automatically start a separate 25-demo demo-only experiment on all
  ten tasks, then evaluate the same 100 test episodes per task. Use deterministic
  25-demo subsets of the same successful RGB-D assignments (250 total), local
  SSD H5s, the identical pinned harness/simulator/config/shared prompts and
  Claude Code / Opus 5.5, eight coding slots and zero training rollouts. Start
  fresh policies and timetags; expose only the assigned 25 demos. Preserve the
  496-demo baseline, original captures and results. Keep test outcomes and prior
  solutions out of learners. Queue with scripts/queue_rmbench_comparison.py;
  record subset identities before launch and report actual baseline shortfalls
  (49/49/48), full scored coverage and per-task comparison results.
- User-requested repair and prompt revision (2026-11-08): call Claude Code /
  Opus 5.5 once more to repair the existing 50-demo RMBench swap_blocks policy.
  This explicitly permits reuse of that same policy in a fresh demo-only run,
  with identical assigned demos/configuration and zero simulator training
  attempts. Use `cam prepare --initial-policy-run SOURCE_RUN` to pin the copied
  program and provenance before learning. Do not copy test feedback or another
  task's solution. Preserve the original 0/100 score and report the repair as a
  separate user-requested follow-up. The shared demo-only prompt now requires
  a final offline code review and execution checks, including first-decision
  memory fields and `last` initialized to None. Apply this task-independent
  wording to future preparations across benchmarks; never edit old prompts.
- Latest RMBench swap_blocks revision (2026-11-08): rerun from scratch using
  the same 50 RGB-D demos, Claude Code / Opus 5.5, the revised shared demo-only
  prompt, zero training rollouts and a blank initial policy; do not import the
  old or repaired solution. Evaluate all 100 held-out episodes after freezing.
  After complete scoring, make this new development/rollout run the website's
  default replacement regardless of score. Latest display revision: remove the
  original and repair swap_blocks development/rollout entries from the website,
  including its historical selector; use the fresh run as the 50-demo entry.
  Preserve original files and scored outcomes on disk. Keep the 25-demo task
  entry and original 25-versus-50 comparison; record the revised current selection.
