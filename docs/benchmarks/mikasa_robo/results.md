# MIKASA-Robo results

We evaluate MIKASA at two scopes: the full 90-task benchmark for broad coverage,
and the five task identities used by prior VLA publications for comparison.
The subset follows [SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533v2),
which reports VLA baselines from MemoryVLA++ and a per-task GMP reference.
`mikasa_robo_vla_suite` names this subset in configurations and solution paths;
it uses the same MIKASA simulator and public interface. We ran both scopes and
report their scores separately.

## Five-task publication comparison

All 15 policies were developed fresh with Claude Code / Opus 5.5. Demo-only
settings use zero simulator training attempts; interaction uses one attempt on
each of the 50 assigned demo identities. The 25-demo arm receives only its exact
materialized subset. Every arm evaluates the same 50 held-out identities per task,
with native scoring and limits, after freezing and without model calls.

| Task | 25 demos only | 50 demos only | 50 demos + 50 interactions |
|---|---:|---:|---:|
| `ShellGameTouch-VLA-v0` | 10/50 | 43/50 | 50/50 |
| `InterceptMedium-VLA-v0` | 44/50 | 41/50 | 50/50 |
| `RememberColor3-VLA-v0` | 50/50 | 45/50 | 50/50 |
| `RememberColor5-VLA-v0` | 48/50 | 31/50 | 50/50 |
| `RememberColor9-VLA-v0` | 1/50 | 15/50 | 50/50 |
| **Total** | **153/250 (61.2%)** | **175/250 (70.0%)** | **250/250 (100%)** |

Coverage is **750/750 scored episodes**, with zero policy or infrastructure errors.
All arms use 256×256 RGB-D and GPU rendering. ShellGameTouch uses GPU physics/IK;
the other four use CPU physics/IK. Matching the published task set does not imply
identical training budgets, sensing, simulator versions or evaluation protocols.
This fresh comparison is distinct from the older five-task slice of the full
benchmark (225/250, including ShellGameTouch at 25/50 on CPU).

The byte-identical frozen solutions and immutable catalogs are published here:

| Setting | Code | Schemas, checksums and provenance |
|---|---|---|
| 25 demos only | [Solutions](../../../solutions/mikasa_robo_vla_suite/demo_only_25/README.md) | [Catalog](../../../configs/solutions/mikasa_robo_vla_suite/demo_only_25.json) |
| 50 demos only | [Solutions](../../../solutions/mikasa_robo_vla_suite/demo_only_50/README.md) | [Catalog](../../../configs/solutions/mikasa_robo_vla_suite/demo_only_50.json) |
| 50 demos + interaction | [Solutions](../../../solutions/mikasa_robo_vla_suite/demo_interaction_50/README.md) | [Catalog](../../../configs/solutions/mikasa_robo_vla_suite/demo_interaction_50.json) |

Development logs, immutable checkpoints and all rollouts remain under
`runs/mikasa_robo_vla_suite/20261110T190922_491466Z/` in this workspace.
Each catalog records its source run, checkpoint, harness/prompt hashes, assigned
demos, training attempts, physics backend and per-task scored/expected counts.

## Full 90-task benchmark

| Result version | Successful / scored / expected | Success rate |
|---|---:|---:|
| Original published archive | 4,444 / 4,500 / 4,500 | 98.76% |
| Aggregate with updated GPU ShellGameTouch | 4,469 / 4,500 / 4,500 | 99.31% |

The updated aggregate retains 89 historical task results and replaces only
ShellGameTouch's historical 25/50 with the current frozen policy's GPU 50/50.
That policy is from an earlier development run, distinct from the fresh five-task
comparison above. This is a mixed-provenance aggregate, not a new 90-task rerun.
Its catalog, selected result inventory and exact copied policies are retained in
`runs/mikasa_robo_full90/20261110T191044_604595Z_shellgame_gpu/`.

The [original 90 published policies](../../../solutions/mikasa_robo/demo_interaction_50/README.md)
and [catalog](../../../configs/solutions/mikasa_robo/demo_interaction_50.json)
remain unchanged. They used 50 demos and 50 training interactions per task with
CPU physics. Full development/video evidence covers the initial ten tasks;
80 imported tasks contribute 3,970/4,000 results without full historical source,
reset-seed, development or video evidence. Current compatibility targets are not
verified historical provenance.

### Original per-task results

Successful / scored episodes; every denominator also equals the expected count.
Only ShellGameTouch changes to 50/50 in the updated aggregate described above.

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
| `ShellGameTouch-VLA-v0` | 25/50 |
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
| **Total** | **4,444/4,500 (98.76%)** |

Original records remain under `public_runs/eval/mikasa_robo/demo_interaction_50/`
(locally `/data/daiyp/CaM_public`). The five-task comparison's 750 results are
never added to either 4,500-episode full-benchmark total.

[Run or develop MIKASA](README.md) · [All benchmark results](../../results.md)
