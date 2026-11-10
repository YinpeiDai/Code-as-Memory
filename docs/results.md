# Published results and provenance

These counts were consolidated from the selected public archive's original
result JSONs. They describe frozen historical experiments, not the new sanity
checks or a current-harness reproduction.

| Benchmark | Setting | Successful / scored / expected* | Success rate |
|---|---|---:|---:|
| MIKASA-Robo | demo_interaction_50 | 4,444/4,500 | 98.76% |
| RMBench | demo_only_25 | 927/1,000 | 92.70% |
| RMBench | demo_only_50 | 1,000/1,000 | 100.00% |
| RoboMemArena | demo_interaction_50 | 1,271/1,300 | 97.77% |
| RoboMemArena | demo_only_25 | 1,070/1,300 | 82.31% |
| RoboMemArena | demo_only_50 | 1,179/1,300 | 90.69% |
| RoboMME | demo_only_25 | 795/800 | 99.38% |

*Every selected episode is scored; the denominator is both scored and expected.
There are 10,400 held-out results across these seven settings. Rates use official
whole-task success (TSR for RoboMemArena). Do not pool different settings into a
single benchmark headline or count repeated checks as new published evidence.

## Figures and comparison scope

The [four-benchmark figure](figures/benchmark_comparison.png) compares selected
published reports with CaM on matching task lists. It uses all 16 RoboMME tasks,
nine RMBench tasks (excluding `place_block_mat`), five MIKASA task identities,
and all 26 RoboMemArena tasks. The MIKASA subset is **225/250 = 90.0%**:

| MIKASA task | CaM successful / evaluated |
|---|---:|
| ShellGameTouch | 25 / 50 |
| InterceptMedium | 50 / 50 |
| RememberColor3 | 50 / 50 |
| RememberColor5 | 50 / 50 |
| RememberColor9 | 50 / 50 |

Sources are the [RoboMME leaderboard](https://robomme.github.io/leaderboard.html),
[SimpleMemVLA v2, Figure 3 and Tables 1/3](https://arxiv.org/pdf/2609.05533v2),
and the [RoboMemArena leaderboard](https://robomemarena.github.io/leaderboard.html).
Training data, sensing, interaction budgets, simulator versions and evaluation
counts differ; MIKASA matches task identities across legacy and current VLA
releases. RoboMemArena baselines are listed as literature-reported/unverified by
its maintainers. These are contextual comparisons, not controlled ablations.

The [RoboMME study](figures/robomme_efficiency.png) reports **4.37 ms median**
perception/memory decision time (**476.8 ms P95**) and **$29.44 development API
cost**, with zero evaluation API cost. Timing excludes action-tool execution and
simulation; cost excludes local compute and data preparation.
[Exact measurements, sources, and figure reproduction](figures/README.md).

## How to interpret the settings

**RoboMME.** The published 25-demo experiment used its original runner and prompt. A current-harness rerun is separate evidence. Development accepts the released 25-demo package through a metadata-checked compatibility reader, preserving every H5 byte.

**MIKASA-Robo.** Each task uses 50 demos and one interaction on each of those exact identities, with CPU physics. All 90 policies and 4,500 result records are available. Full development/video evidence covers the initial ten tasks; the other 80 are imported results. Their missing historical source snapshots and reset-seed evidence are not independently verified.

**RMBench.** The larger assignment has 49 demos for battery_try, 49 for blocks_ranking_try, 48 for observe_and_pickup, and 50 for the other seven tasks. The selected 50-demo swap_blocks policy is the fresh rerun. Earlier outcomes remain archived. Current evaluation uses observation-derived MotionError; historical scores used their pinned interfaces.

**RoboMemArena.** The larger assignment has 50 demos per task except T04, which has 32. Interaction uses those same identities exactly once (50/32 attempts). The 25-demo arm uses the revised offline-review prompt; the interaction arm adds manipulation/recovery guidance. These comparisons change prompts as well as data or interaction.

Shared demo-only prompts gained a final offline code review after some baseline
runs. The RoboMemArena interaction supplement followed host review of the
baseline. Learners received their assigned demonstrations and counted training
feedback, without held-out diagnostics or historical solutions. These differences
must be disclosed in comparisons.

## Evidence locations

The archive is rooted at `public_runs/` (locally `/data/daiyp/CaM_public`):
`demos/<benchmark>/`, `dev_logs/<benchmark>/<setting>/`, and
`eval/<benchmark>/<setting>/`. The generated `dashboard/results.json` and CSV
contain per-task counts. Setting READMEs link to unchanged
`configs/solutions/<benchmark>/<setting>.json` catalogs.

Release receipts are `ROBOMME_RELEASE.json`, `RMBENCH_RELEASE.json`,
`ROBOMEMARENA_RELEASE.json`, and the initial-ten `MIKASA_RELEASE.json`.
`MIKASA_EVAL_STATISTICS_IMPORT.json` records the additional 3,970/4,000 imported
outcomes, giving 4,444/4,500 overall. Only the initial ten have full local
learning/video/trajectory evidence. Missing source snapshots, prompts, or reset
metadata for imports are not inferred from the current interface.

Original data, snapshots, intermediate revisions, scores, and verification
receipts are preserved. The old chronological working documents live locally
under ignored `legacy/20261110_public_reorganization/docs/`; the concise guides
here describe the supported public workflows.
