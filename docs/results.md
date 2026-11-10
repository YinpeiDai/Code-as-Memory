# Results and provenance

Results are reported separately for each benchmark, setting and evaluation scope.
Every denominator below is both the scored and expected episode count. Per-task
tables, solution links and evidence limitations are on each benchmark's result page.
New sanity checks and current-interface reruns never replace original scores.

## RoboMME

| Setting | Successful / scored / expected | Success rate |
|---|---:|---:|
| 25 demos only | 795 / 800 / 800 | 99.38% |

[All 16 tasks and provenance](benchmarks/robomme/results.md).

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

[Both scopes, per-task results, code and provenance](benchmarks/mikasa_robo/results.md).

## RMBench

| Setting | Successful / scored / expected | Success rate |
|---|---:|---:|
| 25 demos only | 927 / 1,000 / 1,000 | 92.70% |
| Up to 50 demos only | 1,000 / 1,000 / 1,000 | 100.00% |

[All ten tasks, exact demo counts and interface provenance](benchmarks/rmbench/results.md).

## RoboMemArena

| Setting | Successful / scored / expected | Whole-task success (TSR) |
|---|---:|---:|
| 25 demos only | 1,070 / 1,300 / 1,300 | 82.31% |
| Up to 50 demos only | 1,179 / 1,300 / 1,300 | 90.69% |
| Up to 50 demos + matching interactions | 1,271 / 1,300 / 1,300 | 97.77% |

[All 26 tasks, budgets and prompt differences](benchmarks/robomemarena/results.md).

## Figures and comparison scope

The [four-benchmark figure](figures/benchmark_comparison.png) uses all 16 RoboMME
tasks, nine RMBench tasks (excluding `place_block_mat`), the five MIKASA publication
tasks, and all 26 RoboMemArena tasks. Its MIKASA CaM result now uses the fresh
`mikasa_robo_vla_suite/demo_interaction_50` result, 250/250, with an immutable
catalog. The older five-task slice, 225/250, remains in the figure's source JSON
as historical evidence.

Baseline sources are the [RoboMME leaderboard](https://robomme.github.io/leaderboard.html),
[SimpleMemVLA v2](https://arxiv.org/pdf/2609.05533v2), and the
[RoboMemArena leaderboard](https://robomemarena.github.io/leaderboard.html).
Matching task identities supports comparison with prior publications, but
training data, interaction budgets, sensing, simulator versions and evaluation
counts differ. RoboMemArena baselines are listed as literature-reported/unverified
by its maintainers. These are contextual comparisons, not controlled ablations.

The [RoboMME timing study](figures/robomme_efficiency.png) reports **4.37 ms median**
perception/memory decision time (**476.8 ms P95**) and **$29.44 development API
cost**, with zero evaluation API cost. Timing excludes action execution and
simulation; cost excludes local compute and data preparation.
[Figure sources and measurement details](figures/README.md).

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
