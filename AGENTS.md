# Working on CaM

Read [docs/reproducing.md](docs/reproducing.md) before preparing or running an experiment.

## Current public layout

- The four simulator interfaces are `src/cam/interface/{robomme,mikasa_robo,rmbench,robomemarena}/`.
  Each has its own environment in `envs/`; the root environment is the shared harness.
- Active plans and client configurations live in `configs/<benchmark>/`.
  `configs/solutions/` is required for immutable policy schemas, checksums and provenance.
- Historical sources, plans and documentation are retained locally in `legacy/`.
  The full preceding instructions are preserved in
  `legacy/20261110_public_reorganization/AGENTS.md`; their paths describe historical runs.
  Never relaunch an old queued experiment merely because it appears there.
- Use Robot Memory wording in current documentation. Keep public guides concise;
  technical detail belongs in `docs/` and the relevant interface/environment README.

## Current user-authorized defaults

- This workspace uses `/home/daiyp/Code-as-Memory/runs` for development logs,
  rollout data, and related run artifacts. Pass this root with `--output` when a
  published plan defaults to `public_runs` or `validate_runs`; keep existing
  artifacts in place.
- Eight concurrent development jobs. Normal evaluation uses five episode workers
  per GPU, with GPU admission and unchanged episode scoring.
- The release sanity check covers every published task, setting and solution:
  five complete episodes per policy, ten workers per GPU on GPUs 0 and 1.
  Store logs and JSON only in `runs/eval_sanity_check/`; use recording `results`.
  Preserve failures and archive unscorable infrastructure attempts before identical retries.
  Task failure is a valid execution outcome; policy/runtime errors must be reported.
- Publish demonstration pools to `Yinpei/<benchmark>-<count>-demos`: RoboMME's
  existing 25/task and the other three benchmarks' up-to-50/task packages.
  Include H5, available training-demonstration videos and exact selected-25 JSON.
  Materialize the selected subset during experiment preparation. A learner assigned
  25 demonstrations must never receive the full 50-demo H5 or a mount of the pool.
- All website inputs, progress, consolidated results and preview caches belong
  under the selected run directory (`runs` in this workspace). Gallery cards
  open a separate page; detail episode navigation stays in that page.

## Experiment integrity

- Preserve original datasets, scored outcomes, prompts, code, checkpoints and
  source snapshots. Published executable policy bytes must remain unchanged.
  Harness, prompt, configuration or data changes require a fresh timestamped
  preparation; never edit a prepared source snapshot to bypass integrity checks.
- Use `cam prepare` and the requested client/model. Fresh development starts
  with blank policies unless the user explicitly authorizes a repair import.
- Hosts may repair infrastructure and public interfaces. Learners use only the
  CaM tools, assigned demonstrations and counted training feedback. Never expose
  benchmark implementations, hidden state, held-out diagnostics, prior solutions,
  task-specific fixes or unassigned identities to a learner.
- Freeze before testing. Evaluation makes no model calls. Keep current-interface
  reruns separate from historical scores, with actual scored/expected coverage.
- Shared core/mode prompts stay identical across benchmarks. Benchmark additions
  are public interface guides and the documented task-independent supplements.
  MIKASA `SeqOfColors3-VLA-v0` retains its authorized static cue-order clarification.
- RMBench uses shared demo-only guidance without the retired motion supplement.
  MotionError uses public post-action poses with 0.02 m / 0.1 rad tolerances;
  native planner flags remain unavailable. The earlier catalogs and scores remain.
- Published MIKASA has 90 policies and 4,444/4,500 successes. The imported 80
  contribute 3,970/4,000 results but lack full development/rollout evidence.
  Never describe their compatibility targets as verified historical provenance.
- Current MIKASA interaction reproduction uses CPU physics except
  `ShellGameTouch-VLA-v0`, which uses GPU physics, with 50 demos and one
  attempt per assigned demo identity. RoboMemArena uses 50 demos/attempts except
  T04's actual 32. RMBench's larger pool has 49/49/48 demos for battery_try,
  blocks_ranking_try and observe_and_pickup; the other seven have 50.
  Never pad counts or reuse consumed identities; exclude held-out seeds.
- Preparation trusts released manifests. Do not replay or fully revalidate pools
  by default. Explicit subset preparation copies only assigned groups; publishing
  verifies transfer hashes without generating new simulator data.
