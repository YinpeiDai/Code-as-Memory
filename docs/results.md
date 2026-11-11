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

Our primary comparison uses the five-task selection from prior VLA publications;
we also retain full 90-task results for broad coverage.
The subset follows [SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533v2)
and is named `mikasa_robo_vla_suite` in code. It uses the same simulator and public
interface; the separate namespace keeps its policies and scores distinct.

| Scope | Setting / result version | Successful / scored / expected | Success rate |
|---|---|---:|---:|
| Five publication tasks | 25 demos only | 153 / 250 / 250 | 61.2% |
| Five publication tasks | 50 demos only | 175 / 250 / 250 | 70.0% |
| Five publication tasks | 50 demos + 50 interactions | 250 / 250 / 250 | 100% |
| Full 90 tasks | Original published 50 demos + interaction | 4,444 / 4,500 / 4,500 | 98.76% |
| Full 90 tasks | Aggregate with GPU ShellGameTouch update | 4,469 / 4,500 / 4,500 | 99.31% |

The GPU-updated aggregate is not a new full-suite rerun. Full-90 scores remain
separate from the five-task comparison.

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

The [comparison figure](figures/benchmark_comparison.png) uses all 16 RoboMME tasks,
nine RMBench tasks (excluding `place_block_mat`), the five-task MIKASA VLA suite,
and all 26 RoboMemArena tasks. Sources are the
[RoboMME leaderboard](https://robomme.github.io/leaderboard.html),
[SimpleMemVLA, Figure 3 and Tables 1/3](https://arxiv.org/pdf/2609.05533v2), and the
[RoboMemArena leaderboard](https://robomemarena.github.io/leaderboard.html).
Training budgets, sensing, simulator versions and evaluation protocols differ.
RoboMemArena baseline entries are literature-reported/unverified by its maintainers.

Development API costs are **$22.72** for RMBench, **$33.81** for the MIKASA VLA suite,
**$29.44** for RoboMME and **$278.57** for RoboMemArena; evaluation makes zero API
calls. These are usage estimates excluding compute and data preparation.
RMBench's cost covers all ten developed tasks; its figure compares nine.

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
