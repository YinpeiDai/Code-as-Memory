# MIKASA-Robo results

Our primary MIKASA comparison uses the five task identities evaluated by prior
VLA publications. We also report the full 90-task benchmark for broad coverage.
The subset follows [SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533v2),
which reports VLA baselines from MemoryVLA++ and a per-task GMP reference.
`mikasa_robo_vla_suite` names this subset in configurations and solution paths;
it uses the same MIKASA simulator and public interface. We ran both scopes and
report their scores separately.

The [original VLA training release](https://mikasarobo.github.io/datasets.html)
provides **250 demonstrations per task** (22,500 across 90 tasks; 1,250 for the
five-task subset). CaM's comparison uses **25, 50 or up to 250 demos/task**, with
the actual assignments and interaction budgets reported below.

## Five-task publication comparison

All 20 policies started blank. Demo-only permits zero simulator attempts;
interaction uses **50 attempts per task**, one per assigned demo seed across all
revisions. The 25-demo arm receives only its selected subset. Each policy freezes
before evaluation on the same 50 held-out identities, with native scoring and no
model calls.

| Task | 25 demos only | 50 demos only | Up to 250 demos only | 50 demos + 50 interactions |
|---|---:|---:|---:|---:|
| `ShellGameTouch-VLA-v0` | 10/50 | 43/50 | 50/50 | 50/50 |
| `InterceptMedium-VLA-v0` | 44/50 | 41/50 | 30/50 | 50/50 |
| `RememberColor3-VLA-v0` | 50/50 | 45/50 | 50/50 | 50/50 |
| `RememberColor5-VLA-v0` | 48/50 | 31/50 | 49/50 | 50/50 |
| `RememberColor9-VLA-v0` | 1/50 | 15/50 | 18/50 | 50/50 |
| **Total** | **153/250 (61.2%)** | **175/250 (70.0%)** | **197/250 (78.8%)** | **250/250 (100%)** |

All **1,000 held-out episodes** across the four settings completed, with zero
policy or infrastructure errors. Table entries are successful/total episodes.
All arms use 256×256 RGB-D and GPU rendering. ShellGameTouch uses GPU physics/IK;
the other four use CPU physics/IK. Matching the published task set does not imply
identical training budgets, sensing, simulator versions or evaluation protocols.

The [`mikasa_robo_vla_suite_250` plan](../../configs/mikasa_robo_vla_suite_250/experiments/demo_only_250.json)
uses only successful RGB-D replays from the 250 released training demos per task:
**237** for ShellGameTouch, **56** for InterceptMedium, **250** each for
RememberColor3/5, and **246** for RememberColor9 (**1,039 total**). Failed replays
are excluded. This arm adds the approved
[MIKASA demo-only guidance](../../src/cam/prompts/mikasa_robo/demos.md). Its score is
**8.8 percentage points** above the 50-demo result; the data budget and prompt
both differ. InterceptMedium decreases from 41/50 to 30/50.

The up-to-250 arm's development API cost is **$20.01**, as reported by the coding
client, excluding compute and data preparation. Its frozen programs are published
unchanged below. Development logs, code history, intermediate analysis and rollouts
are available in the
[development archive](https://huggingface.co/datasets/Yinpei/CaM_develop_logs/blob/main/archives/mikasa_robo_vla_suite_demo_only_250.tar.zst),
which extracts to `public_runs/dev_logs/mikasa_robo_vla_suite/demo_only_250`.
Download the [RGB-D demos](https://huggingface.co/datasets/Yinpei/mikasa_robo-vla-suite-250-demos)
separately to `public_runs/demos/mikasa_robo_vla_suite_250_demo`;
see the [download command](../benchmarks/mikasa_robo.md#up-to-250-demos-only).

| Setting | Code | Schemas, checksums and provenance |
|---|---|---|
| 25 demos only | [Solutions](../../solutions/mikasa_robo_vla_suite/demo_only_25/README.md) | [Catalog](../../configs/solutions/mikasa_robo_vla_suite/demo_only_25.json) |
| 50 demos only | [Solutions](../../solutions/mikasa_robo_vla_suite/demo_only_50/README.md) | [Catalog](../../configs/solutions/mikasa_robo_vla_suite/demo_only_50.json) |
| Up to 250 demos only | [Solutions](../../solutions/mikasa_robo_vla_suite/demo_only_250/README.md) | [Catalog](../../configs/solutions/mikasa_robo_vla_suite/demo_only_250.json) |
| 50 demos + interaction | [Solutions](../../solutions/mikasa_robo_vla_suite/demo_interaction_50/README.md) | [Catalog](../../configs/solutions/mikasa_robo_vla_suite/demo_interaction_50.json) |

[Download development logs](../getting_started.md#development-logs) for all 20 policies.
The archives retain prompts, revisions and generated files, with demonstration
data and external simulator assets excluded. The catalogs
record source checkpoints, code/prompt hashes, assigned budgets, physics backends
and evaluation coverage.

## Full 90-task benchmark

| Setting | Successful/total | Success rate |
|---|---:|---:|
| 50 demos + 50 interactions | 4,469/4,500 | 99.31% |

Scores use the ShellGameTouch evaluation with GPU physics and the archived
results for the other 89 tasks.
[All 90 published policies](../../solutions/mikasa_robo/demo_interaction_50/README.md) ·
[Policy catalog and archived provenance](../../configs/solutions/mikasa_robo/demo_interaction_50.json).

### Per-task results

Successful/total held-out episodes.

| Task | 50 demos + 50 interactions |
|---|---:|
| `BatteriesCheckerEasy-3-VLA-v0` | 49/50 |
| `BatteriesCheckerEasy-6-VLA-v0` | 49/50 |
| `BatteriesCheckerHard-3-VLA-v0` | 50/50 |
| `BatteriesCheckerHard-6-VLA-v0` | 49/50 |
| `BlinkCountButtonPressEasy-Long-VLA-v0` | 48/50 |
| `BlinkCountButtonPressEasy-VLA-v0` | 50/50 |
| `BlinkCountButtonPressHard-Long-VLA-v0` | 50/50 |
| `BlinkCountButtonPressHard-VLA-v0` | 50/50 |
| `BlinkCountButtonPressMedium-Long-VLA-v0` | 50/50 |
| `BlinkCountButtonPressMedium-VLA-v0` | 46/50 |
| `BunchOfColors3-Long-VLA-v0` | 50/50 |
| `BunchOfColors3-VLA-v0` | 50/50 |
| `BunchOfColors5-Long-VLA-v0` | 50/50 |
| `BunchOfColors5-VLA-v0` | 50/50 |
| `BunchOfColors7-Long-VLA-v0` | 50/50 |
| `BunchOfColors7-VLA-v0` | 50/50 |
| `ChainOfColors3-Long-VLA-v0` | 50/50 |
| `ChainOfColors3-VLA-v0` | 50/50 |
| `ChainOfColors5-Long-VLA-v0` | 50/50 |
| `ChainOfColors5-VLA-v0` | 50/50 |
| `ChainOfColors7-Long-VLA-v0` | 47/50 |
| `ChainOfColors7-VLA-v0` | 50/50 |
| `FindImposterColor3-VLA-v0` | 50/50 |
| `FindImposterColor5-VLA-v0` | 50/50 |
| `FindImposterColor9-VLA-v0` | 50/50 |
| `FindImposterShape3-VLA-v0` | 50/50 |
| `FindImposterShape5-VLA-v0` | 50/50 |
| `FindImposterShape9-VLA-v0` | 49/50 |
| `FindImposterShapeAndColor3x2-VLA-v0` | 50/50 |
| `FindImposterShapeAndColor3x3-VLA-v0` | 48/50 |
| `FindImposterShapeAndColor5x3-VLA-v0` | 50/50 |
| `GatherAndRecall1-VLA-v0` | 50/50 |
| `GatherAndRecall3-VLA-v0` | 50/50 |
| `GatherAndRecall5-VLA-v0` | 50/50 |
| `GatherAndRecall7-VLA-v0` | 50/50 |
| `GatherAndRecall9-VLA-v0` | 50/50 |
| `InterceptFast-VLA-v0` | 45/50 |
| `InterceptGrabFast-VLA-v0` | 50/50 |
| `InterceptGrabMedium-VLA-v0` | 50/50 |
| `InterceptGrabSlow-VLA-v0` | 50/50 |
| `InterceptMedium-VLA-v0` | 50/50 |
| `InterceptSlow-VLA-v0` | 49/50 |
| `RememberColor3-Long-VLA-v0` | 50/50 |
| `RememberColor3-VLA-v0` | 50/50 |
| `RememberColor5-Long-VLA-v0` | 50/50 |
| `RememberColor5-VLA-v0` | 50/50 |
| `RememberColor9-Long-VLA-v0` | 50/50 |
| `RememberColor9-VLA-v0` | 50/50 |
| `RememberShape3-Long-VLA-v0` | 50/50 |
| `RememberShape3-VLA-v0` | 50/50 |
| `RememberShape5-Long-VLA-v0` | 49/50 |
| `RememberShape5-VLA-v0` | 49/50 |
| `RememberShape9-Long-VLA-v0` | 50/50 |
| `RememberShape9-VLA-v0` | 49/50 |
| `RememberShapeAndColor3x2-Long-VLA-v0` | 50/50 |
| `RememberShapeAndColor3x2-VLA-v0` | 50/50 |
| `RememberShapeAndColor3x3-Long-VLA-v0` | 49/50 |
| `RememberShapeAndColor3x3-VLA-v0` | 50/50 |
| `RememberShapeAndColor5x3-Long-VLA-v0` | 50/50 |
| `RememberShapeAndColor5x3-VLA-v0` | 50/50 |
| `RotateLenientPos-VLA-v0` | 47/50 |
| `RotateLenientPosNeg-VLA-v0` | 50/50 |
| `RotateStrictPos-VLA-v0` | 50/50 |
| `RotateStrictPosNeg-VLA-v0` | 50/50 |
| `SeqOfColors3-Long-VLA-v0` | 50/50 |
| `SeqOfColors3-VLA-v0` | 50/50 |
| `SeqOfColors5-Long-VLA-v0` | 50/50 |
| `SeqOfColors5-VLA-v0` | 50/50 |
| `SeqOfColors7-Long-VLA-v0` | 47/50 |
| `SeqOfColors7-VLA-v0` | 50/50 |
| `ShellGameColorLampTouch-VLA-v0` | 50/50 |
| `ShellGamePush-VLA-v0` | 50/50 |
| `ShellGameShuffleColorLampTouch-Long-VLA-v0` | 50/50 |
| `ShellGameShuffleColorLampTouch-VLA-v0` | 50/50 |
| `ShellGameShuffleTouch-Long-VLA-v0` | 50/50 |
| `ShellGameShuffleTouch-VLA-v0` | 50/50 |
| `ShellGameTouch-VLA-v0` | 50/50 |
| `TakeItBack-VLA-v0` | 50/50 |
| `TimedTransferEasy-Long-VLA-v0` | 50/50 |
| `TimedTransferEasy-VLA-v0` | 50/50 |
| `TimedTransferHard-Long-VLA-v0` | 50/50 |
| `TimedTransferHard-VLA-v0` | 50/50 |
| `TimedTransferMedium-Long-VLA-v0` | 50/50 |
| `TimedTransferMedium-VLA-v0` | 50/50 |
| `TraceShapeEasy-VLA-v0` | 50/50 |
| `TraceShapeHard-VLA-v0` | 50/50 |
| `TraceShapeMedium-VLA-v0` | 50/50 |
| `TraceShapeSeqEasy-VLA-v0` | 50/50 |
| `TraceShapeSeqHard-VLA-v0` | 50/50 |
| `TraceShapeSeqMedium-VLA-v0` | 50/50 |
| **Total** | **4,469/4,500 (99.31%)** |

Original records remain under `public_runs/eval/mikasa_robo/demo_interaction_50/`.
The five-task comparison's 1,000 results are kept separate.

[Run or develop MIKASA](../benchmarks/mikasa_robo.md) · [All benchmark results](../results.md)
