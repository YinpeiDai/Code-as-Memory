# Tasks for cross-machine validation

**Recommended: 44 distinct tasks, 2,500 held-out episodes using the best published setting for each benchmark.** This is a selected validation panel; report it separately from full-benchmark scores.

| Benchmark | Tasks | Primary setting | Held-out episodes | Archived panel successes |
|---|---:|---|---:|---:|
| MIKASA-Robo | 20 | `demo_interaction_50` | 1,000 972/1,000 |
| RoboMME | 6 | `demo_only_25` | 300 295/300 |
| RoboMemArena | 12 | `demo_interaction_50` | 600 571/600 |
| RMBench | 6 | `demo_only_50` | 600 600/600 |

Use the archived panel totals above when comparing this selected task set.

Selection was recorded on 2026-11-10 against code commit `c07c81d`. All score columns below are archived **full-evaluation** successes/episodes. The completed five-episode sanity check is separate.

## MIKASA-Robo: initial ten plus ten random imported tasks

The initial ten come from batch `20261108T040959_138834Z`. Keep all ten, including the weak `ShellGameTouch-VLA-v0` case.

| Initial task | Published result |
|---|---:|
| `BatteriesCheckerEasy-3-VLA-v0` | 49/50 |
| `BlinkCountButtonPressEasy-VLA-v0` | 50/50 |
| `BunchOfColors3-VLA-v0` | 50/50 |
| `ChainOfColors3-VLA-v0` | 50/50 |
| `FindImposterColor3-VLA-v0` | 50/50 |
| `RememberColor3-VLA-v0` | 50/50 |
| `SeqOfColors3-VLA-v0` | 50/50 |
| `ShellGameTouch-VLA-v0` | 25/50 |
| `TakeItBack-VLA-v0` | 50/50 |
| `TimedTransferEasy-VLA-v0` | 50/50 |

The other ten are a uniform sample without replacement from the other 80 tasks, using Python 3.11 `random.Random(20261110)` on the sorted task list. The sample was retained without reselection based on scores.

| Randomly selected imported task | Published result |
|---|---:|
| `FindImposterShape9-VLA-v0` | 49/50 |
| `FindImposterShapeAndColor3x2-VLA-v0` | 50/50 |
| `GatherAndRecall7-VLA-v0` | 50/50 |
| `RememberColor5-Long-VLA-v0` | 50/50 |
| `RememberColor9-Long-VLA-v0` | 50/50 |
| `RememberColor9-VLA-v0` | 50/50 |
| `RememberShape9-VLA-v0` | 49/50 |
| `SeqOfColors5-VLA-v0` | 50/50 |
| `TimedTransferHard-VLA-v0` | 50/50 |
| `TimedTransferMedium-Long-VLA-v0` | 50/50 |

The imported 80 have frozen policies and result JSONs, but lack complete historical development artifacts. Fresh development uses the current concise prompt and creates new evidence; do not claim exact reproduction of missing historical snapshots.

Reconstruct the sample from the committed catalog:

```python
import json, random
catalog = json.load(open("configs/solutions/mikasa_robo/demo_interaction_50.json"))
initial = sorted(t for t, r in catalog["tasks"].items()
                 if r.get("source_run_tag") == "20261108T040959_138834Z")
pool = sorted(set(catalog["tasks"]) - set(initial))
assert len(initial) == 10 and len(pool) == 80
print(sorted(random.Random(20261110).sample(pool, 10)))
```

## RoboMME: both non-perfect tasks plus four perfect controls

| Task | Published result | Selection |
|---|---:|---|
| `StopCube` | 47/50 | Non-perfect published result |
| `VideoPlaceOrder` | 48/50 | Non-perfect published result |
| `MoveCube` | 50/50 | Perfect published control |
| `PatternLock` | 50/50 | Perfect published control |
| `PickXtimes` | 50/50 | Perfect published control |
| `VideoUnmaskSwap` | 50/50 | Perfect published control |

`VideoPlaceOrder` also scored 4/5 in the new sanity check. The primary setting remains the existing 25-demo assignment with no training interaction.

## RoboMemArena: eight weak 25-demo cases plus four strong controls

Include every task whose 25-demo result is at most 80%, plus four controls with at least 96%. T04 covers the actual 32-demo shortfall; T18 retains residual failures even in the best setting.

| Task | 25 demos | Up to 50 demos | Up to 50 + interaction | Selection |
|---|---:|---:|---:|---|
| `T04_ButterNonEmptyDrawer` | 50/50 | 50/50 | 50/50 | Strong 25-demo control |
| `T05_ButterEmptyDrawer` | 0/50 | 50/50 | 50/50 | Weak at 25 demos |
| `T07_PourSauceFrypanTwice` | 50/50 | 50/50 | 50/50 | Strong 25-demo control |
| `T11_CookiesTopButterMiddle` | 28/50 | 36/50 | 50/50 | Weak at 25 demos |
| `T13_CookiesButterMiddleDrawer` | 29/50 | 41/50 | 50/50 | Weak at 25 demos |
| `T17_ButterChocolateMiddleDrawer` | 32/50 | 37/50 | 50/50 | Weak at 25 demos |
| `T18_ChocolateButterCabinet` | 48/50 | 48/50 | 48/50 | Strong 25-demo control |
| `T19_SauceMilkJuiceCabinet` | 12/50 | 20/50 | 25/50 | Weak at 25 demos |
| `T20_CookiesChocolateMicrowave` | 50/50 | 50/50 | 50/50 | Strong 25-demo control |
| `T23_CreamPopcornMicrowave` | 5/50 | 41/50 | 50/50 | Weak at 25 demos |
| `T25_ButterCreamPlate` | 33/50 | 42/50 | 48/50 | Weak at 25 demos |
| `T26_ChocolateCreamPlate` | 39/50 | 33/50 | 50/50 | Weak at 25 demos |

Start with `demo_interaction_50`. For a setting comparison, repeat this same task panel with `demo_only_25` and `demo_only_50`; each extra arm adds 600 held-out episodes. These settings also differ in prompt revisions, so they are not a pure demo-count ablation.

## RMBench: weak cases, requested regression targets, and controls

| Task | 25 demos | Up to 50 demos | Selection |
|---|---:|---:|---|
| `cover_blocks` | 28/100 | 100/100 | Main weak 25-demo case |
| `observe_and_pickup` | 99/100 | 100/100 | Non-perfect 25-demo case; 48-demo larger pool |
| `swap_blocks` | 100/100 | 100/100 | Requested regression target; earlier policy was 0/100, final fresh rerun is 100/100 |
| `swap_T` | 100/100 | 100/100 | Requested regression target; current final published scores are perfect |
| `blocks_ranking_try` | 100/100 | 100/100 | Perfect control; larger pool has 49 demos |
| `battery_try` | 100/100 | 100/100 | Perfect control; larger pool has 49 demos |

The exact task ID is `swap_blocks` (plural). Start with `demo_only_50`; optionally repeat the same panel with `demo_only_25` for another 600 held-out episodes. Current evaluations use observation-derived MotionError and must stay separate from historical-interface scores.

## Copyable task selections

Run these Bash definitions from the repository root:

```bash
MIKASA=(
  BatteriesCheckerEasy-3-VLA-v0
  BlinkCountButtonPressEasy-VLA-v0
  BunchOfColors3-VLA-v0
  ChainOfColors3-VLA-v0
  FindImposterColor3-VLA-v0
  RememberColor3-VLA-v0
  SeqOfColors3-VLA-v0
  ShellGameTouch-VLA-v0
  TakeItBack-VLA-v0
  TimedTransferEasy-VLA-v0
  FindImposterShape9-VLA-v0
  FindImposterShapeAndColor3x2-VLA-v0
  GatherAndRecall7-VLA-v0
  RememberColor5-Long-VLA-v0
  RememberColor9-Long-VLA-v0
  RememberColor9-VLA-v0
  RememberShape9-VLA-v0
  SeqOfColors5-VLA-v0
  TimedTransferHard-VLA-v0
  TimedTransferMedium-Long-VLA-v0
)
ROBOMME=(
  StopCube
  VideoPlaceOrder
  MoveCube
  PatternLock
  PickXtimes
  VideoUnmaskSwap
)
ROBOMEMARENA=(
  T04_ButterNonEmptyDrawer
  T05_ButterEmptyDrawer
  T07_PourSauceFrypanTwice
  T11_CookiesTopButterMiddle
  T13_CookiesButterMiddleDrawer
  T17_ButterChocolateMiddleDrawer
  T18_ChocolateButterCabinet
  T19_SauceMilkJuiceCabinet
  T20_CookiesChocolateMicrowave
  T23_CreamPopcornMicrowave
  T25_ButterCreamPlate
  T26_ChocolateCreamPlate
)
RMBENCH=(
  cover_blocks
  observe_and_pickup
  swap_blocks
  swap_T
  blocks_ranking_try
  battery_try
)
```

## Run on the other machine

Pull `main`, initialize submodules, install the shared harness and the selected simulator environment following [installation](getting_started.md). The dataset uploads are still in progress; download after that benchmark has a verified `configs/<benchmark>/datasets/release.json` pin.

Choose one row and set `TASKS` to its array:

| `BENCHMARK` | `SETTING` | Bash assignment |
|---|---|---|
| `mikasa_robo` | `demo_interaction_50` | `TASKS=("${MIKASA[@]}")` |
| `robomme` | `demo_only_25` | `TASKS=("${ROBOMME[@]}")` |
| `robomemarena` | `demo_interaction_50` | `TASKS=("${ROBOMEMARENA[@]}")` |
| `rmbench` | `demo_only_50` | `TASKS=("${RMBENCH[@]}")` |

### Re-evaluate the unchanged published policies

No demonstrations, coding client, or model credentials are needed. This runs all held-out episodes for the selected tasks and records videos and logs.

```bash
BENCHMARK=mikasa_robo
SETTING=demo_interaction_50
TASKS=("${MIKASA[@]}")
LOGS="$PWD/validate_runs/cross_machine/$BENCHMARK/$SETTING"
python scripts/evaluate_solutions.py "solutions/$BENCHMARK/$SETTING" \
  --tasks "${TASKS[@]}" --output "$LOGS" --gpus 0 1 --workers-per-gpu 5
```

### Reproduce development, then evaluate

This starts new coding sessions with the configured client/model. Use blank policies, the current benchmark prompt, and the assigned data only. The primary MIKASA and RoboMemArena settings consume one interaction per assigned demo identity (50 each; T04 has 32).

```bash
DATA="$PWD/public_runs/demos/$BENCHMARK"
cam download-demos --benchmark "$BENCHMARK" --output "$DATA" --tasks "${TASKS[@]}"
python scripts/develop_batch.py start "configs/$BENCHMARK/experiments/$SETTING.json" \
  --dataset "$DATA" --output "$LOGS" --tasks "${TASKS[@]}" \
  --workers 8 --gpus 0 1 --evaluate-on-freeze --evaluation-workers-per-gpu 5
```

Run one benchmark batch at a time to keep eight coding jobs total. Full evaluation defaults to five episode workers per GPU. The separate quick sanity check uses five episodes per policy and ten jobs per GPU.

For a 25-demo comparison, change `SETTING` to `demo_only_25`, update `LOGS` for that setting, and reuse the larger downloaded pool; preparation materializes its exact `selected_25.json` assignment before learning. Keep this host selection report and its historical scores outside the learner workspace and prompt.

```bash
cam site --data-root validate_runs
cam serve validate_runs/website --port 8936
```

Retain the new run IDs, full development logs, frozen code hashes, prompt/interface hashes, scored/expected counts, and infrastructure-retry archives. Compare the same held-out identities. Do not retry scored task failures or use held-out feedback to repair a learner.

Sources: [solution catalogs](../configs/solutions/), archived per-episode results under `public_runs/eval/`, and [result provenance](results.md).
