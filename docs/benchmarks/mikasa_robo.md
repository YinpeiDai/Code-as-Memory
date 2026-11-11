# MIKASA-Robo

Our primary comparison is the **five-task VLA subset** used in
[SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533v2): ShellGameTouch,
InterceptMedium and RememberColor3/5/9 (standard `-VLA-v0` IDs).
We also publish solutions for **all 90 tasks**. Both scopes use the same simulator
and public interface; `mikasa_robo_vla_suite` keeps the five-task policies
and results separate. Matching task identities does not make training budgets,
sensing or evaluation protocols identical to prior work.

The [original VLA training release](https://mikasarobo.github.io/datasets.html)
provides **250 demonstrations per task**: **22,500 across 90 tasks**, including
**1,250 for the five-task subset**. CaM's assigned budgets are listed below;
the up-to-250 arm keeps only successful RGB-D replays.

Install the [MIKASA environment](../../envs/mikasa_robo/README.md) and see
[per-task results for both scopes](../results/mikasa_robo.md).
Training and evaluation use GPU physics/IK for ShellGameTouch and CPU by default
for other tasks. All use 256×256 RGB-D with GPU rendering.

## Five-task VLA comparison subset

| Setting | Demos/task | Rollout attempts/task | Held-out episodes/task |
|---|---:|---:|---:|
| [demo_only_25](../../solutions/mikasa_robo_vla_suite/demo_only_25/) | 25 | 0 | 50 |
| [demo_only_50](../../solutions/mikasa_robo_vla_suite/demo_only_50/) | 50 | 0 | 50 |
| [demo_only_250](../../solutions/mikasa_robo_vla_suite/demo_only_250/) | Up to 250 | 0 | 50 |
| [demo_interaction_50](../../solutions/mikasa_robo_vla_suite/demo_interaction_50/) | 50 | 50 | 50 |

The up-to-250 arm uses 237/56/250/250/246 successful RGB-D demos for ShellGameTouch,
InterceptMedium and RememberColor3/5/9, respectively, and scores **197/250 (78.8%)**.
See [results and prompt provenance](../results/mikasa_robo.md#five-task-publication-comparison).

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

Replace the setting with `demo_only_25`, `demo_only_50` or `demo_only_250` as needed.
To develop the three original settings from blank policies:

```bash
python scripts/develop_suite.py start configs/mikasa_robo_vla_suite/suite.json \
  --dataset /path/to/downloaded/demos/mikasa_robo --output runs --gpus 0 1
```

The settings share eight coding workers and GPU leases. Full evaluation queues
as each policy freezes, with five episode workers per GPU. Check progress with
`python scripts/develop_suite.py status CAMPAIGN`.
[Individual setting plans](../../configs/mikasa_robo_vla_suite/README.md).

### Up to 250 demos only

Download the [five-task RGB-D package](https://huggingface.co/datasets/Yinpei/mikasa_robo-vla-suite-250-demos)
before starting fresh development. It contains **1,039 successful demos** and their
videos, separate from the 50-demo package. The
[release record](../../configs/mikasa_robo_vla_suite_250/datasets/release.json)
pins the dataset revision and manifest checksum.

```bash
DATA="$PWD/public_runs/demos/mikasa_robo_vla_suite_250_demo"
hf download Yinpei/mikasa_robo-vla-suite-250-demos --repo-type dataset \
  --revision 186b955291af1de526b7f4d9825f2c3a8a006305 --local-dir "$DATA"
(cd "$DATA" && sha256sum -c SHA256SUMS)
```

Develop five fresh policies with zero training interactions:

```bash
python scripts/develop_batch.py start \
  configs/mikasa_robo_vla_suite_250/experiments/demo_only_250.json \
  --dataset "$DATA" \
  --output runs/mikasa_robo_vla_suite_250 --workers 8 --gpus 0 1 \
  --evaluate-on-freeze --evaluation-workers-per-gpu 5
```

The [published code](../../solutions/mikasa_robo_vla_suite/demo_only_250/) evaluates
without downloading demos. Download the
[development archive](https://huggingface.co/datasets/Yinpei/CaM_develop_logs/blob/main/archives/mikasa_robo_vla_suite_demo_only_250.tar.zst)
using the [log-download instructions](../getting_started.md#development-logs).
It extracts to `public_runs/dev_logs/mikasa_robo_vla_suite/demo_only_250`, retaining
prompts, code history, intermediate analysis and generated held-out rollouts.
Demonstration data and external simulator assets are excluded.

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
[full-90 results](../results/mikasa_robo.md#full-90-task-benchmark) are reported
separately from the five-task VLA comparison.

## Prompt and interface

Development combines the shared core/mode and MIKASA public interface with
[`demos.md`](../../src/cam/prompts/mikasa_robo/demos.md) for current demo-only runs
or [`both.md`](../../src/cam/prompts/mikasa_robo/both.md) for interaction.
SeqOfColors3 also has its documented
[cue-order clarification](../../src/cam/prompts/mikasa_robo/seq_of_colors3.md).
Each preparation pins its exact prompt.

[Public interface](../../src/cam/interface/mikasa_robo/guide.md) ·
[Interface adaptations](../interface.md) · [Reproduction](../reproducing.md)
