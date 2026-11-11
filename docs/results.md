# Results and provenance

Results are reported separately for each benchmark, setting and evaluation scope.
Tables report **successful/total** held-out episodes; every listed evaluation is complete. Per-task
tables, solution links and evidence limitations are on each benchmark's result page.
New sanity checks and current-interface reruns never replace original scores.

## RoboMME

The original release provides [**100 training demos/task**](https://github.com/RoboMME/robomme_benchmark).
CaM uses **25/task** with zero interaction.

| Setting | Successful/total | Success rate |
|---|---:|---:|
| 25 demos only | 795/800 | 99.38% |

[All 16 tasks and provenance](results/robomme.md).

## MIKASA-Robo

Our primary comparison uses the five-task selection from prior VLA publications;
we also retain full 90-task results for broad coverage.
The subset follows [SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533v2)
and is named `mikasa_robo_vla_suite` in code. It uses the same simulator and public
interface; the separate namespace keeps its policies and scores distinct.

The original VLA release provides [**250 training demos/task**](https://mikasarobo.github.io/datasets.html).
CaM uses **25, 50 or up to 250/task**, with interaction budgets shown below.

| Scope | Setting / result version | Successful/total | Success rate |
|---|---|---:|---:|
| Five publication tasks | 25 demos only | 153/250 | 61.2% |
| Five publication tasks | 50 demos only | 175/250 | 70.0% |
| Five publication tasks | Up to 250 demos only | 197/250 | 78.8% |
| Five publication tasks | 50 demos + 50 interactions | 250/250 | 100% |
| Full 90 tasks | Original published 50 demos + interaction | 4,444/4,500 | 98.76% |
| Full 90 tasks | Aggregate with GPU ShellGameTouch update | 4,469/4,500 | 99.31% |

The up-to-250 arm, `mikasa_robo_vla_suite_250`, uses 237/56/250/250/246 successful
RGB-D demos for ShellGameTouch, InterceptMedium and RememberColor3/5/9, respectively,
with zero interaction. Its updated demo-only guidance and provenance are recorded
on the [MIKASA results page](results/mikasa_robo.md#five-task-publication-comparison).

The GPU-updated aggregate is not a new full-suite rerun. Full-90 scores remain
separate from the five-task comparison.

[Both scopes, per-task results, code and provenance](results/mikasa_robo.md).

## RMBench

The original simulation protocol uses [**50 training demos/task**](https://arxiv.org/html/2603.01229v1#S5.SS1).
CaM uses **25 or up to 50/task** with zero interaction; the larger assignment has
48–50 successful demos per task.

| Setting | Successful/total | Success rate |
|---|---:|---:|
| 25 demos only | 927/1,000 | 92.70% |
| Up to 50 demos only | 1,000/1,000 | 100.00% |

[All ten tasks, exact demo counts and interface provenance](results/rmbench.md).

## RoboMemArena

The original dataset provides [**100 full-task training demos/task**](https://arxiv.org/html/2605.10921v1#S3.SS3).
CaM uses **25 or up to 50/task**; T04 has 32 in the larger assignment. Interaction
uses one attempt per assigned demo (50/task, or 32 for T04).

| Setting | Successful/total | Whole-task success (TSR) | Cumulative success (CSR) |
|---|---:|---:|---:|
| 25 demos only | 1,070/1,300 | 82.31% | 91.69% |
| Up to 50 demos only | 1,179/1,300 | 90.69% | 94.34% |
| Up to 50 demos + matching interactions | 1,271/1,300 | 97.77% | 98.15% |

CSR averages the native completed-stage percentage across episodes, giving
partial credit for progress within a task.

[All 26 tasks, budgets and prompt differences](results/robomemarena.md).

## Figures and comparison scope

The [comparison figure](figures/benchmark_comparison.png) uses all 16 RoboMME tasks,
nine RMBench tasks (excluding `place_block_mat`), the five-task MIKASA VLA suite,
and all 26 RoboMemArena tasks. Sources are the
[RoboMME leaderboard](https://robomme.github.io/leaderboard.html),
[SimpleMemVLA, Figure 3 and Tables 1/3](https://arxiv.org/pdf/2609.05533v2), and the
[RoboMemArena leaderboard](https://robomemarena.github.io/leaderboard.html).
Training budgets, sensing, simulator versions and evaluation protocols differ.
RoboMemArena baseline entries are literature-reported/unverified by its maintainers.

The figure separates demo-only learning (25 demos for RoboMME, up to 250 for
MIKASA and up to 50 for RMBench/RoboMemArena) from demos + interaction on MIKASA
and RoboMemArena. Each setting shows its own success and development API cost.
MIKASA's interaction arm uses 50 demos + 50 attempts.
See [exact budgets, costs and sources](figures/README.md). Evaluation makes zero
API calls; development costs exclude compute and data preparation.

The [RoboMME computation study](figures/robomme_efficiency.png) measures **1.413 ms
mean policy computation per control step**, including action helpers and excluding
simulation, IPC, H5 decoding and startup. See [measurements and figure reproduction](figures/README.md).

## Evidence and reproduction

[Published code](../solutions/README.md) and [immutable catalogs](../configs/solutions/)
retain policy bytes, schemas, checksums and source provenance. Download
[development logs](getting_started.md#development-logs) and
[demonstrations](getting_started.md#demonstration-packages) separately.

Each benchmark's result page records exact demo counts and material prompt or
interface differences. Published scores use their pinned runs; current-interface
reruns are separate evidence. Historical prompts and available logs remain intact,
and current compatibility checks do not establish missing historical provenance.
