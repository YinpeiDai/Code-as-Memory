# Publication figures

| Figure | Preview | Vector exports | Measurements |
|---|---|---|---|
| Four benchmarks | [PNG](benchmark_comparison.png) | [SVG](benchmark_comparison.svg) · [PDF](benchmark_comparison.pdf) | [JSON](benchmark_results.json) |
| RoboMME performance, computation and API cost | [PNG](robomme_efficiency.png) | [SVG](robomme_efficiency.svg) · [PDF](robomme_efficiency.pdf) | [JSON](robomme_comparison.json) |

Regenerate the exports from the installed root environment:

```bash
python scripts/plot_public_results.py
```

## Comparison scope

Figures use whole-task success, equal task weights and CaM's best published
setting. JSON files retain exact counts, per-task measurements and source hashes.

| Benchmark | CaM figure coverage | Baseline source |
|---|---|---|
| RMBench | 900/900; nine tasks, up to 50 demos | [SimpleMemVLA v2](https://arxiv.org/pdf/2609.05533v2), Figure 3 / Table 1 |
| MIKASA-Robo | 250/250; five-task VLA suite, 50 demos + 50 interactions | [SimpleMemVLA v2](https://arxiv.org/pdf/2609.05533v2), Figure 3 / Table 3 |
| RoboMME | 795/800; 16 tasks, 25 demos | [Official leaderboard](https://robomme.github.io/leaderboard.html) |
| RoboMemArena | 1,271/1,300; 26 tasks, up to 50 demos + matching interactions | [Official leaderboard](https://robomemarena.github.io/leaderboard.html) |

RMBench excludes `place_block_mat` to match the paper's task list; CaM's full
suite is 1,000/1,000. EventVLA uses the paper's reported 67.8% overall.
MIKASA uses ShellGameTouch, InterceptMedium and RememberColor3/5/9; its full-90
results remain separate. Demo counts are per task, including the
[documented smaller-pool exceptions](../getting_started.md#learning-settings).

RoboMME's MEMI entry is MEMI-Astra-Act; the oracle-planner variant is excluded.
RoboMemArena baselines come from its literature-reported/unverified leaderboard
and use whole-task TSR. Training budgets, sensing, action interfaces, simulator
versions and evaluation counts differ across methods. See [per-benchmark results](../results.md).

## API costs

| Benchmark | Development scope | Development API cost | Evaluation API cost |
|---|---|---:|---:|
| RMBench | Ten tasks, `demo_only_50` | $22.72 | $0 |
| MIKASA-Robo | Five-task VLA suite, `demo_interaction_50` | $33.81 | $0 |
| RoboMME | 16 tasks, `demo_only_25` | $29.44 | $0 |
| RoboMemArena | 26 tasks, `demo_interaction_50` | $278.57 | $0 |

Costs are finalized coding-client usage estimates, excluding compute, data
preparation and other settings. RMBench's cost covers ten tasks while its success
figure covers nine. Evaluation makes zero model calls. The
[cost manifest](benchmark_costs.json) retains exact totals, per-task costs,
source hashes and evaluation coverage.

With the public archive and its referenced evaluation artifacts, repeat the audit:

```bash
python scripts/collect_figure_costs.py --archive public_runs \
  --output "runs/figure_studies/costs/$(date -u +%Y%m%dT%H%M%SZ).json"
```

## RoboMME policy computation

CaM averages **1.413 ms of policy computation per native control step**. This
includes perception, memory updates and generated action helpers; it excludes
simulation, waiting inside `robot.step`, IPC, H5 decoding, imports and startup.

The study replays the first five recorded test episodes per task, without outcome
filtering: **80 episodes, 16 policies and 19,080 control steps**. Replayed decisions,
memory, commands and action outcomes match their recordings. Each episode has one
warm-up and three timed repeats on an AMD EPYC 9334 CPU with single-threaded math
libraries. The mean divides summed per-episode median computation by recorded steps.
It is amortized computation: an initial perception segment reached 881.3 ms.

Baseline timing comes from [Table 2 of the RoboMME blog](https://robomme.github.io/blog/),
measured on **RoboMME-Plus** with 2× RTX 6000 Ada GPUs and a 32-core AMD EPYC CPU.
It is an approximate reference across different protocols and hardware.
MEMI-Astra-Act has no reported timing there. Unreported measurements are **N/R**.
The leaderboard's API costs cover evaluation; baseline development spending is
unreported. The [comparison JSON](robomme_comparison.json) records each scope.

Reproduce the CPU replay study in a fresh directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/profile_robomme_policy.py --archive public_runs \
  --output "runs/figure_studies/robomme_policy/$(date -u +%Y%m%dT%H%M%SZ)" \
  --episodes 5 --repeats 3
```

This uses recorded observations and frozen policies, without simulators or models,
and produces timing measurements rather than new success scores.
