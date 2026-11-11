# MIKASA-Robo

Our primary comparison is the **five-task VLA subset** used in
[SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533v2): ShellGameTouch,
InterceptMedium and RememberColor3/5/9 (standard `-VLA-v0` IDs).
We also publish solutions for **all 90 tasks**. Both scopes use the same simulator,
public interface and demo pool; `mikasa_robo_vla_suite` keeps the five-task policies
and results separate. Matching task identities does not make training budgets,
sensing or evaluation protocols identical to prior work.

Install the [MIKASA environment](../../envs/mikasa_robo/README.md) and see
[per-task results for both scopes](../results/mikasa_robo.md).
Training and evaluation use GPU physics/IK for ShellGameTouch and CPU by default
for other tasks. All use 256×256 RGB-D with GPU rendering.

## Five-task VLA comparison subset

| Setting | Demos/task | Training attempts/task | Held-out episodes/task |
|---|---:|---:|---:|
| [demo_only_25](../../solutions/mikasa_robo_vla_suite/demo_only_25/) | 25 | 0 | 50 |
| [demo_only_50](../../solutions/mikasa_robo_vla_suite/demo_only_50/) | 50 | 0 | 50 |
| [demo_interaction_50](../../solutions/mikasa_robo_vla_suite/demo_interaction_50/) | 50 | 50 | 50 |

Interaction permits **50 total attempts using the seeds of the 50 assigned
training demos, each once**, across all policy revisions. Failed attempts count;
there are no new training seeds or retries. Held-out seeds are separate. The
25-demo arm receives only its selected subset.
[Shared budget protocol](../reproducing.md#demonstration-and-interaction-budgets).

Evaluate a published setting without demos or model calls:

```bash
python scripts/evaluate_solutions.py solutions/mikasa_robo_vla_suite/demo_interaction_50 \
  --output runs/mikasa_vla_eval --gpus 0 1 --workers-per-gpu 5
```

Replace the setting with `demo_only_25` or `demo_only_50` as needed.
To develop all three settings from blank policies:

```bash
python scripts/develop_suite.py start configs/mikasa_robo_vla_suite/suite.json \
  --dataset /path/to/downloaded/demos/mikasa_robo --output runs --gpus 0 1
```

The settings share eight coding workers and GPU leases. Full evaluation queues
as each policy freezes, with five episode workers per GPU. Check progress with
`python scripts/develop_suite.py status CAMPAIGN`.
[Individual setting plans](../../configs/mikasa_robo_vla_suite/README.md).

## Full 90-task benchmark

[All 90 published policies](../../solutions/mikasa_robo/demo_interaction_50/)
use 50 demos + interaction. Evaluate the whole setting:

```bash
python scripts/evaluate_solutions.py solutions/mikasa_robo/demo_interaction_50 \
  --output runs/mikasa_full_eval --gpus 0 1 --workers-per-gpu 5
```

For fresh development, use `configs/mikasa_robo/experiments/demo_interaction_50.json`
with the [batch launcher](../reproducing.md#concurrent-development).
Add `--tasks SeqOfColors3-VLA-v0` to select one task. The
[full-90 results](../results/mikasa_robo.md#full-90-task-benchmark) explain the
original and GPU-updated aggregates separately from the VLA suite.

## Prompt and interface

Development combines the shared core/mode, MIKASA public interface and
[`both.md`](../../src/cam/prompts/mikasa_robo/both.md) for interaction.
SeqOfColors3 also has its documented
[cue-order clarification](../../src/cam/prompts/mikasa_robo/seq_of_colors3.md).
Each preparation pins its exact prompt.

[Public interface](../../src/cam/interface/mikasa_robo/guide.md) ·
[Interface adaptations](../interface.md) · [Reproduction](../reproducing.md)
