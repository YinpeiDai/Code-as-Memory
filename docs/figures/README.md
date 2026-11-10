# Publication figures

| Figure | Preview | Vector exports | Measurements |
|---|---|---|---|
| Four benchmarks | [PNG](benchmark_comparison.png) | [SVG](benchmark_comparison.svg) · [PDF](benchmark_comparison.pdf) | [JSON](benchmark_results.json) |
| RoboMME performance, latency and API cost | [PNG](robomme_efficiency.png) | [SVG](robomme_efficiency.svg) · [PDF](robomme_efficiency.pdf) | [JSON](robomme_comparison.json) |

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
| MIKASA-Robo | 100%; five task identities, author-reported CPU-physics correction | [SimpleMemVLA v2](https://arxiv.org/pdf/2609.05533v2), Figure 3 and Table 3 |
| RoboMemArena | 1,271/1,300; 26 tasks | [Official leaderboard](https://robomemarena.github.io/leaderboard.html), reported TSR |

**RoboMME:** MEMI is the **MEMI-Astra-Act** variant. The separate
MEMI-Astra-Plan result uses an oracle action planner and is not plotted.

**RMBench:** The shared list excludes `place_block_mat`. The remaining tasks are
`observe_and_pickup`, `rearrange_blocks`, `put_back_block`, `swap_blocks`, `swap_T`,
`battery_try`, `blocks_ranking_try`, `cover_blocks`, and `press_button`.
CaM's full ten-task suite is 1,000/1,000. EventVLA retains the paper's reported
67.8% overall; its displayed integer per-task values average to 67.89%.

**MIKASA-Robo:** The shared identities are ShellGameTouch, InterceptMedium,
RememberColor3, RememberColor5, and RememberColor9. The figure uses the author's
reported 100% correction after identifying a CPU physics issue. Corrected episode
counts and rerun artifacts have not been attached; the JSON retains the original
225/250 result separately as `archived_result`. The archived experiment evaluates their current
`-VLA-v0` implementations with 50 episodes each, 50 assigned demonstrations plus
50 training interactions, CPU physics, and its public observation/action
interface. The paper quotes VLA baselines from MemoryVLA++ and GMP from its
per-task training with tuned horizons. Task identities match across legacy and
current releases, but simulator versions, sensing and training budgets do not.
CaM's full 90-task
suite is 4,444/4,500 (98.76%); it is not the five-task comparison score.

**RoboMemArena:** The plotted baselines come from the leaderboard's
literature-reported/unverified section, not independently verified submissions.
Evaluation counts and protocols vary across methods. CaM reports whole-task
success (TSR), not partial-progress success (PSR).

Across all panels, these are selected published reports rather than a controlled
comparison: demonstrations, interaction budgets, sensing, action interfaces,
simulator revisions and evaluation counts differ. See [results and provenance](../results.md)
for CaM's historical evidence and imported MIKASA results.

## RoboMME performance, latency and API cost

The comparison covers CaM, MEMI-Astra-Act, Embodied Astra, SimpleMemVLA,
PonderPounce and FrameSamp+Modul. Its [JSON](robomme_comparison.json) retains
source URLs, download hashes, coverage, hardware, per-task timing and raw-study
paths. Unreported values remain `null` and are displayed as **N/R**.

**Performance** uses original [RoboMME leaderboard](https://robomme.github.io/leaderboard.html)
results. CaM achieves **795/800 (99.375%)**. Different methods have different
training, sensing, action interfaces and evaluation protocols.

**Policy computation** is expressed as mean milliseconds per native control step.
The baseline values come from [Table 2 of the official blog](https://robomme.github.io/blog/),
which measures **RoboMME-Plus**, not original RoboMME. They are approximate
references: FrameSamp+Modul 6.59 ms, SimpleMemVLA 69.12 ms, PonderPounce 199.39 ms,
and Embodied Astra 343.80 ms. That table uses 2× RTX 6000 Ada GPUs and a 32-core
AMD EPYC CPU. MEMI-Astra-Act has no timing entry there. We do not infer its latency
from a different Astra system or claim a controlled speedup across these setups.

CaM's new measurement includes both **perception/memory decisions and generated
action helpers**. These helpers can run perception, track objects, fit motion,
and compute subsequent commands. Simulator work and waiting inside `robot.step`
are excluded, along with IPC, H5 decoding, imports and startup.

The profiler replays the first five recorded test episodes per task, without
filtering by outcome: **80 episodes, 16 policies, 19,080 native control steps and
19,000 emitted commands**. Every replayed decision, memory write, command and
completed action outcome matches its original recording. Each episode has one
warm-up and three timed repetitions on an AMD EPYC 9334 CPU, with
OpenMP/OpenBLAS/MKL limited to one thread. The total is the sum of each episode's
median complete policy computation time, divided by its recorded native steps.

This gives **1.412990 ms per control step**, or **337.0 ms of policy computation
per episode on average**. The figure labels this mean **<2 ms**; the bar and JSON
retain the exact measurement. This is amortized computation, not a guaranteed response
deadline: a measured initial perception segment reached **881.3 ms**. Samples are
the first five episode identities per task, not all 800 test episodes. README
videos also change playback speed and hold memory states for readability.

The earlier [decision-only study](robomme_efficiency.json) remains available:
225 decisions, median 4.365912 ms, mean 79.703524 ms, P95 476.848693 ms. It excludes
action helpers and uses a different denominator. Its 4.37 ms median must not be
presented as the complete policy latency or compared directly with per-step means.

**API cost** separates development from evaluation. CaM's **$29.4412816** sums the
16 finalized development `costs/api_cost.json` reports; all 800 original evaluation
results record **$0 and zero API calls**. These are API usage estimates, not
audited bills. The leaderboard reports $706 for MEMI-Astra-Act's RoboMME result
and $310 for Embodied Astra's RoboMME evaluation; its metric notes define API
cost over the 800 test episodes. The open-model blog evaluations make no API
calls. None of these sources separately reports the baselines' total development
API spending, so those cells are **N/R**, never assumed zero. GPU/CPU compute,
data preparation and foundation-model pretraining are excluded from every API
cost cell; zero API cost does not mean free training or inference.

Full timing traces, progress and the exact profiler source are saved under
`validate_runs/figure_studies/robomme_full_policy/20261110T210600Z/`.
Source snapshots are under
`validate_runs/figure_studies/robomme_baseline_comparison/20261110T210538Z/`.
Reproduce the CPU study in a fresh directory with the original public archive:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/profile_robomme_policy.py --archive public_runs \
  --output "validate_runs/figure_studies/robomme_full_policy/$(date -u +%Y%m%dT%H%M%SZ)" \
  --episodes 5 --repeats 3
```

This executes frozen code against recorded observations. It starts no simulator,
uses no model, changes no policy, and produces no new success scores.
