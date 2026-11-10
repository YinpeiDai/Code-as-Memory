# Publication figures

| Figure | Preview | Vector exports | Measurements |
|---|---|---|---|
| Four benchmarks | [PNG](benchmark_comparison.png) | [SVG](benchmark_comparison.svg) · [PDF](benchmark_comparison.pdf) | [JSON](benchmark_results.json) |
| RoboMME timing and cost | [PNG](robomme_efficiency.png) | [SVG](robomme_efficiency.svg) · [PDF](robomme_efficiency.pdf) | [JSON](robomme_efficiency.json) |

From the installed root environment, regenerate all six exports without a
simulator, archive download, or model call:

```bash
python scripts/plot_public_results.py
```

## Baseline sources and scope

The comparison uses whole-task success, equal task weights, and the best published
CaM setting for each benchmark. Values are rounded to one decimal for display.
The JSON retains source URLs, snapshot hashes, available per-task values, and
CaM's exact success/episode counts. Sources were checked on November 10, 2026.

| Benchmark | CaM figure coverage | Baseline source |
|---|---|---|
| RoboMME | 795/800; 16 tasks | [Official leaderboard](https://robomme.github.io/leaderboard.html), mean of 16 per-task values |
| RMBench | 900/900; nine tasks | [SimpleMemVLA v2](https://arxiv.org/pdf/2609.05533v2), Figure 3 and Table 1 |
| MIKASA-Robo | 250/250; fresh five-task publication comparison, 50 demos + 50 interactions | [SimpleMemVLA v2](https://arxiv.org/pdf/2609.05533v2), Figure 3 and Table 3 |
| RoboMemArena | 1,271/1,300; 26 tasks | [Official leaderboard](https://robomemarena.github.io/leaderboard.html), reported TSR |

**RoboMME:** MEMI is the **MEMI-Astra-Act** variant. The separate
MEMI-Astra-Plan result uses an oracle action planner and is not plotted.

**RMBench:** The shared list excludes `place_block_mat`. The remaining tasks are
`observe_and_pickup`, `rearrange_blocks`, `put_back_block`, `swap_blocks`, `swap_T`,
`battery_try`, `blocks_ranking_try`, `cover_blocks`, and `press_button`.
CaM's full ten-task suite is 1,000/1,000. EventVLA retains the paper's reported
67.8% overall; its displayed integer per-task values average to 67.89%.

**MIKASA-Robo:** The shared identities are ShellGameTouch, InterceptMedium,
RememberColor3, RememberColor5, and RememberColor9. The figure now uses the fresh
`mikasa_robo_vla_suite/demo_interaction_50` result, **250/250**, verified against
its [immutable catalog](../../configs/solutions/mikasa_robo_vla_suite/demo_interaction_50.json).
Each policy uses 50 assigned demos, 50 training attempts and 50 held-out episodes,
with 256×256 RGB-D. ShellGameTouch uses GPU physics/IK; the others use CPU physics/IK.
The JSON retains the original 225/250 slice as `archived_result` and the earlier
author-reported physics correction as `previous_figure_result`.
The paper quotes VLA baselines from MemoryVLA++ and GMP from its per-task training
with tuned horizons. Task identities match across legacy and current releases,
but simulator versions, sensing and training budgets differ.
The full 90-task historical and GPU-updated aggregates remain separate from this
fresh five-task comparison. [Both MIKASA scopes and per-task results](../results/mikasa_robo.md).

**RoboMemArena:** The plotted baselines come from the leaderboard's
literature-reported/unverified section, not independently verified submissions.
Evaluation counts and protocols vary across methods. CaM reports whole-task
success (TSR), not partial-progress success (PSR).

Across all panels, these are selected published reports rather than a controlled
comparison: demonstrations, interaction budgets, sensing, action interfaces,
simulator revisions and evaluation counts differ. See [results and provenance](../results.md)
for CaM's historical evidence and imported MIKASA results.

## RoboMME timing and cost

The timing study replays the **first five recorded test episodes per task** in
numeric order, without filtering by outcome: 80 episodes and 225 decisions across
all 16 frozen policies. It creates no new success scores. Every replayed action
specification and memory update matched the original recorded decision within
numerical tolerance.

Each decision has one warm-up and five timed repetitions; its median is one
point in the empirical cumulative distribution. Across those 225 points the
median is **4.365912 ms**, mean **79.703524 ms**, and P95 **476.848693 ms**.
Decisions have equal weight; tasks do not. The CPU is an AMD EPYC 9334, with
OpenMP/OpenBLAS/MKL thread limits set to one. Runtime, policy and input hashes,
hardware metadata, per-task summaries and all timing samples accompany the JSON.

Timing covers `cam.core.code_policy.decide`: perception and memory computation,
validation, and its internal memory copies. It excludes observation loading,
H5 decoding, imports/startup, action-tool execution, robot I/O, rendering and
simulation. Some policies do substantial work inside action tools. This is a
measurement of the decision stage, not total controller latency or a hardware
speed comparison with another method.

**$29.4412816** is the sum of all 16 finalized task-level
`costs/api_cost.json` reports from the original development run. Each has final
session usage and matching query/session token totals. These are Claude CLI API
cost estimates, not audited bills; local compute, data preparation and model
pretraining are excluded. All 800 original evaluation result files record zero
API calls and zero API cost. The historical live aggregate omitted the last
completed task, so the figure uses final task reports rather than that incomplete
aggregate. Original reports remain unchanged.

Raw timing artifacts are under
`validate_runs/figure_studies/robomme_efficiency/20261110T_public_figures/`:
`study.json`, `samples.json`, `tasks.json`, `results.json`, `PROGRESS.md`, and the
exact profiling source. The committed JSON is sufficient to redraw the figure.
With the original archive available, collect an independent timing study in a
fresh output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/profile_robomme_decisions.py --archive public_runs \
  --output "validate_runs/figure_studies/robomme_efficiency/$(date -u +%Y%m%dT%H%M%SZ)" \
  --episodes 5 --repeats 5
```

This runs frozen code on recorded observations. It does not run simulators,
execute action tools, call models, or update policies.
