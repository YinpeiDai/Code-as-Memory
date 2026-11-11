# Results and provenance

Results are reported separately for each benchmark, setting and evaluation scope.
Every denominator below is both the scored and expected episode count. Per-task
tables, solution links and evidence limitations are on each benchmark's result page.
New sanity checks and current-interface reruns never replace original scores.

## RoboMME

| Setting | Successful / scored / expected | Success rate |
|---|---:|---:|
| 25 demos only | 795 / 800 / 800 | 99.38% |

[All 16 tasks and provenance](results/robomme.md).

## MIKASA-Robo

We ran both the full 90-task benchmark and the five-task selection used in prior
VLA publications. The latter follows [SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533v2)
and is named `mikasa_robo_vla_suite` in code. It uses the same simulator and public
interface; the separate namespace keeps its policies and scores distinct.

| Scope | Setting / result version | Successful / scored / expected | Success rate |
|---|---|---:|---:|
| Full 90 tasks | Original published 50 demos + interaction | 4,444 / 4,500 / 4,500 | 98.76% |
| Full 90 tasks | Aggregate with GPU ShellGameTouch update | 4,469 / 4,500 / 4,500 | 99.31% |
| Five publication tasks | 25 demos only | 153 / 250 / 250 | 61.2% |
| Five publication tasks | 50 demos only | 175 / 250 / 250 | 70.0% |
| Five publication tasks | 50 demos + 50 interactions | 250 / 250 / 250 | 100% |

The five-task campaign contains 15 fresh frozen policies and **750/750 scored
held-out episodes**, with zero policy or infrastructure errors. ShellGameTouch
uses GPU physics; the other four use CPU physics. Full-90's updated aggregate
replaces only the historical ShellGameTouch entry with an earlier policy's GPU
evaluation; it is not a fresh 90-task rerun. Its 80 imported tasks still lack full
historical development/video evidence. Neither full-90 aggregate includes the
fresh five-task campaign's results.

[Both scopes, per-task results, code and provenance](results/mikasa_robo.md).

## RMBench

| Setting | Successful / scored / expected | Success rate |
|---|---:|---:|
| 25 demos only | 927 / 1,000 / 1,000 | 92.70% |
| Up to 50 demos only | 1,000 / 1,000 / 1,000 | 100.00% |

[All ten tasks, exact demo counts and interface provenance](results/rmbench.md).

## RoboMemArena

| Setting | Successful / scored / expected | Whole-task success (TSR) |
|---|---:|---:|
| 25 demos only | 1,070 / 1,300 / 1,300 | 82.31% |
| Up to 50 demos only | 1,179 / 1,300 / 1,300 | 90.69% |
| Up to 50 demos + matching interactions | 1,271 / 1,300 / 1,300 | 97.77% |

[All 26 tasks, budgets and prompt differences](results/robomemarena.md).

## Figures and comparison scope

The [four-benchmark figure](figures/benchmark_comparison.png) compares selected
published reports with CaM on matching task lists. It uses all 16 RoboMME tasks,
nine RMBench tasks (excluding `place_block_mat`), five MIKASA task identities,
and all 26 RoboMemArena tasks. MIKASA now uses the separate five-task VLA-suite
archive at `public_runs/dev_logs/mikasa_robo_vla_suite/demo_interaction_50/`:
**250/250 = 100%**. ShellGameTouch uses GPU physics; the other four tasks use CPU
physics. The earlier five-task result remains **225/250 = 90.0%**. These new runs
do not replace either full-90 aggregate above. Their exact policies and coverage
are pinned in the [immutable catalog](../configs/solutions/mikasa_robo_vla_suite/demo_interaction_50.json).

| MIKASA task | Earlier successful / evaluated | VLA suite successful / evaluated |
|---|---:|---:|
| ShellGameTouch | 25 / 50 | 50 / 50 |
| InterceptMedium | 50 / 50 | 50 / 50 |
| RememberColor3 | 50 / 50 | 50 / 50 |
| RememberColor5 | 50 / 50 | 50 / 50 |
| RememberColor9 | 50 / 50 | 50 / 50 |

Sources are the [RoboMME leaderboard](https://robomme.github.io/leaderboard.html),
[SimpleMemVLA v2, Figure 3 and Tables 1/3](https://arxiv.org/pdf/2609.05533v2),
and the [RoboMemArena leaderboard](https://robomemarena.github.io/leaderboard.html).
Training data, sensing, interaction budgets, simulator versions and evaluation
counts differ; MIKASA matches task identities across legacy and current VLA
releases. RoboMemArena baselines are listed as literature-reported/unverified by
its maintainers. These are contextual comparisons, not controlled ablations.

The main figure also shows total development API cost: **$22.72** for RMBench,
**$33.81** for the five-task MIKASA VLA suite, **$29.44** for RoboMME, and
**$278.57** for RoboMemArena, with **$0 inference API cost** throughout.
RMBench's cost covers all ten developed tasks; its success comparison covers nine.
Costs are reported API usage estimates, excluding local compute and data preparation.

The separate [RoboMME study](figures/robomme_efficiency.png) measures **1.413 ms
mean policy computation per control step**, including generated action helpers
and excluding simulation, IPC, H5 decoding and startup.
[Exact measurements, sources, and figure reproduction](figures/README.md).

## How to interpret the settings

**RoboMME.** The published 25-demo experiment used its original runner and prompt. A current-harness rerun is separate evidence. Development accepts the released 25-demo package through a metadata-checked compatibility reader, preserving every H5 byte.

**MIKASA-Robo.** The original full-90 setting used 50 demos and one interaction on each of those exact identities, with CPU physics. Current preparations use GPU physics for ShellGameTouch and CPU by default for other tasks; the five-task VLA comparison is reported separately above. All 90 policies and 4,500 result records are available. Full development/video evidence covers the initial ten tasks; the other 80 are imported results. Their missing historical source snapshots and reset-seed evidence are not independently verified.

**RMBench.** The larger assignment has 49 demos for battery_try, 49 for blocks_ranking_try, 48 for observe_and_pickup, and 50 for the other seven tasks. The selected 50-demo swap_blocks policy is the fresh rerun. Earlier outcomes remain archived. Current evaluation uses observation-derived MotionError; historical scores used their pinned interfaces.

**RoboMemArena.** The larger assignment has 50 demos per task except T04, which has 32. Interaction uses those same identities exactly once (50/32 attempts). The 25-demo arm uses the revised offline-review prompt; the interaction arm adds manipulation/recovery guidance. These comparisons change prompts as well as data or interaction.

Shared demo-only prompts gained a final offline code review after some baseline
runs. The RoboMemArena interaction supplement followed host review of the
baseline. Learners received their assigned demonstrations and counted training
feedback, without held-out diagnostics or historical solutions. These differences
must be disclosed in comparisons.

## Evidence locations

The original seven settings contain 11,200 held-out results under `public_runs/`
(locally `/data/daiyp/CaM_public`): `demos/<benchmark>/`,
`dev_logs/<benchmark>/<setting>/`, and `eval/<benchmark>/<setting>/`.
Release receipts are `ROBOMME_RELEASE.json`, `RMBENCH_RELEASE.json`,
`ROBOMEMARENA_RELEASE.json`, and the initial-ten `MIKASA_RELEASE.json`.
`MIKASA_EVAL_STATISTICS_IMPORT.json` records the other 80 MIKASA tasks.

The new MIKASA comparison is under
`runs/mikasa_robo_vla_suite/20261110T190922_491466Z/`; its 750 episodes are reported
by setting rather than pooled into a benchmark headline. Published code and
immutable schemas/checksums live in `solutions/` and `configs/solutions/`.
All 15 development histories also have a checksum-verified copy under
`public_runs/dev_logs/mikasa_robo_vla_suite/` for upload, including training rollouts.
Original data, snapshots, prompts, intermediate revisions and scores are preserved.
Exact source provenance is retained; missing historical evidence is not inferred
from current compatibility targets.
