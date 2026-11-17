# RoboMME results

The [original benchmark release](https://github.com/RoboMME/robomme_benchmark)
provides **100 training demonstrations per task** (1,600 across 16 tasks).
CaM uses **25 demonstrations per task** and no training interaction. These are the
original frozen regular-test runs: **795/800 successes (99.38%)**. Evaluation is complete.
The published experiment used its original runner and prompt; current-harness
reruns are separate evidence.

[Code solutions](../../solutions/robomme/demo_only_25/README.md)

## Regular test results

Successful/total held-out episodes.

| Task | 25 demos only |
|---|---:|
| `BinFill` | 50/50 |
| `ButtonUnmask` | 50/50 |
| `ButtonUnmaskSwap` | 50/50 |
| `InsertPeg` | 50/50 |
| `MoveCube` | 50/50 |
| `PatternLock` | 50/50 |
| `PickHighlight` | 50/50 |
| `PickXtimes` | 50/50 |
| `RouteStick` | 50/50 |
| `StopCube` | 47/50 |
| `SwingXtimes` | 50/50 |
| `VideoPlaceButton` | 50/50 |
| `VideoPlaceOrder` | 48/50 |
| `VideoRepick` | 50/50 |
| `VideoUnmask` | 50/50 |
| `VideoUnmaskSwap` | 50/50 |
| **Total** | **795/800 (99.38%)** |

## OOD-test results

The [OOD-test split](https://github.com/RoboMME/robomme_benchmark/blob/eefb2044066153e1cc2484e28678d254342246b1/doc/ood_split.md)
probes extrapolation of regular RoboMME's task parameters while preserving its
mechanisms and success criteria. It contains **20 expert-verified episodes per
task (320 total)**. CaM reuses the published 25-demo-only solutions unchanged;
the three baselines use their released RoboMME checkpoints. No method receives
OOD training or adaptation.

Successful/total episodes; baseline names link to their source repositories.

| Task | CaM (25 demos only) | [FrameSamp+Modul](https://github.com/RoboMME/robomme_policy_learning/tree/ac846a9a4aec841e483b1dc5d620dfe426bde853) | [SimpleMemVLA](https://github.com/OpenBMB/SimpleMemVLA) | [PonderPounce](https://github.com/worv-ai/PonderPounce/tree/723df35762bb641e1d520e4fa9359b98644adc21) |
|---|---:|---:|---:|---:|
| `BinFill` | 20/20 | 0/20 | 0/20 | 0/20 |
| `ButtonUnmask` | 19/20 | 0/20 | 19/20 | 18/20 |
| `ButtonUnmaskSwap` | 19/20 | 1/20 | 5/20 | 2/20 |
| `InsertPeg` | 18/20 | 0/20 | 12/20 | 0/20 |
| `MoveCube` | 20/20 | 13/20 | 17/20 | 10/20 |
| `PatternLock` | 16/20 | 0/20 | 0/20 | 0/20 |
| `PickHighlight` | 19/20 | 0/20 | 3/20 | 1/20 |
| `PickXtimes` | 20/20 | 0/20 | 0/20 | 16/20 |
| `RouteStick` | 20/20 | 0/20 | 10/20 | 1/20 |
| `StopCube` | 19/20 | 0/20 | 0/20 | 0/20 |
| `SwingXtimes` | 20/20 | 0/20 | 4/20 | 5/20 |
| `VideoPlaceButton` | 20/20 | 5/20 | 9/20 | 9/20 |
| `VideoPlaceOrder` | 17/20 | 4/20 | 6/20 | 4/20 |
| `VideoRepick` | 20/20 | 0/20 | 0/20 | 7/20 |
| `VideoUnmask` | 18/20 | 0/20 | 19/20 | 17/20 |
| `VideoUnmaskSwap` | 20/20 | 3/20 | 2/20 | 6/20 |
| **Total** | **305/320 (95.3%)** | **26/320 (8.1%)** | **106/320 (33.1%)** | **96/320 (30.0%)** |

This is an exploratory comparison with different execution-step limits. CaM has
a 1,300-step limit throughout. SimpleMemVLA has a 1,300-step limit for 217 episodes
and 1,500 for 103; PonderPounce has 1,300 for 156 and 1,500 for 164. FrameSamp+Modul
has a 1,500-step limit throughout. All 320 episodes count for each method,
including four native IK errors counted as CaM failures.

## Evidence

Regular-test counts were checked against the original `result.json` records under
`public_runs/eval/robomme/`.
Catalogs under `configs/solutions/robomme/` retain code hashes and source provenance.
New sanity checks and current-interface reruns do not replace these outcomes.
The separate OOD table was checked against the completed CaM and baseline
evaluation records; it does not replace the regular-test results.

[Run or develop this benchmark](../benchmarks/robomme.md) · [All benchmark results](../results.md)
