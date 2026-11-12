# Publication figures

| Figure | Preview | Vector exports | Measurements |
|---|---|---|---|
| Four benchmarks | [WebP](benchmark_comparison.webp) · [PNG](benchmark_comparison.png) | [SVG](benchmark_comparison.svg) · [PDF](benchmark_comparison.pdf) | [JSON](benchmark_results.json) |
| RoboMME performance, computation and API cost | [PNG](robomme_efficiency.png) | [SVG](robomme_efficiency.svg) · [PDF](robomme_efficiency.pdf) | [JSON](robomme_comparison.json) |

The README uses a lossless WebP preview with the same pixels and dimensions as the PNG.

Regenerate the exports from the installed root environment:

```bash
python scripts/plot_public_results.py
```

## Comparison scope

The four-benchmark figure uses horizontal bars for whole-task success, with
**offline demos only** in blue and **demos + interaction** in green.
The four columns run left to right: RMBench, RoboMME, MIKASA-Robo, RoboMemArena.
The color-matched keys show nominal per-task demo budgets (25, 50 or 250) and
total development API costs. Task weights are equal; actual demo counts and
interaction budgets follow. RoboMemArena's CSR is in its [results table](../results/robomemarena.md).

| Benchmark | CaM setting (per task) | Successful/total | Development API cost |
|---|---|---:|---:|
| RMBench | Up to 50 demos only | 898/900 | $22.72 |
| RoboMME | 25 demos only | 795/800 | $29.44 |
| MIKASA-Robo | Up to 250 demos only | 197/250 | $20.01 |
| MIKASA-Robo | 50 demos + 50 attempts | 250/250 | $33.81 |
| RoboMemArena | Up to 50 demos only | 1,179/1,300 | $77.85 |
| RoboMemArena | Up to 50 demos + matching attempts | 1,271/1,300 | $278.57 |

Baseline sources are [SimpleMemVLA v2](https://arxiv.org/pdf/2609.05533v2),
Figure 3 / Tables 1 and 3 for RMBench and MIKASA, and the official
[RoboMME](https://robomme.github.io/leaderboard.html) and
[RoboMemArena](https://robomemarena.github.io/leaderboard.html) leaderboards.
JSON files retain exact counts, per-task measurements and source hashes.

RMBench excludes `place_block_mat` to match the paper's task list; CaM's full
suite is 998/1,000 in the November 11 held-out evaluation of the released policies.
Both scopes round to 99.8%. EventVLA uses the paper's reported 67.8% overall.
MIKASA uses ShellGameTouch, InterceptMedium and RememberColor3/5/9; its full-90
results remain separate. Its demo-only arm uses 237/56/250/250/246 successful RGB-D
demos, respectively. It differs from the interaction arm in both demo budget and
prompt, so the comparison does not isolate the effect of interaction.
RoboMemArena's larger assignment has 32 demos/attempts for T04 and 50 for other
tasks; RMBench has 48–50 demos/task. RoboMME and RMBench show only demo-only learning.

RoboMME's MEMI entry is MEMI-Astra-Act; the oracle-planner variant is excluded.
RoboMemArena baselines come from its literature-reported/unverified leaderboard
and use whole-task TSR. Training budgets, sensing, action interfaces, simulator
versions and evaluation counts differ across methods. See [per-benchmark results](../results.md).

## API costs

Costs above total all developed tasks in each setting: ten for RMBench, five for
MIKASA, 16 for RoboMME and 26 for RoboMemArena. They are finalized coding-client
usage estimates, excluding compute, data preparation and other settings.
RMBench's cost covers ten tasks while its success figure covers nine.
**Evaluation costs $0 in every CaM setting** and makes zero model calls. The
[cost manifest](benchmark_costs.json) retains exact totals, per-task costs,
source hashes and evaluation coverage.

With the public archive and its referenced evaluation artifacts, repeat the audit:

```bash
python scripts/collect_figure_costs.py --archive public_runs \
  --output "runs/figure_studies/costs/$(date -u +%Y%m%dT%H%M%SZ).json"
```

The up-to-250 MIKASA evidence is under
`public_runs/dev_logs/mikasa_robo_vla_suite/demo_only_250`. The optional
`--mikasa-250` argument supports the original experiment root. Rendering needs
only the committed JSON files; their original audit paths remain provenance.

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
