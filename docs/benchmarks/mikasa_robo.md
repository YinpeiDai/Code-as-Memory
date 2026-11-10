# MIKASA-Robo

We publish code solutions for all **90 MIKASA-Robo tasks** and also develop fresh
policies for a **five-task VLA comparison subset**. The subset follows the task selection in
[SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533v2): ShellGameTouch,
InterceptMedium and RememberColor3/5/9, using their standard `-VLA-v0` IDs.
The full benchmark measures broad coverage; the subset supports comparison on
the same task identities as prior work. It is named `mikasa_robo_vla_suite` in
code, and shares MIKASA's simulator, environment, demo package and public interface.
Training budgets, sensing and simulator protocols still differ from prior methods.
The subset is part of MIKASA-Robo, not an additional benchmark.

| Scope | Published code solutions |
|---|---|
| Full benchmark: all 90 tasks | [50 demos + interaction](../../solutions/mikasa_robo/demo_interaction_50/) |
| VLA subset: five tasks, three settings (15 policies) | [25 demos only](../../solutions/mikasa_robo_vla_suite/demo_only_25/), [50 demos only](../../solutions/mikasa_robo_vla_suite/demo_only_50/), [50 demos + interaction](../../solutions/mikasa_robo_vla_suite/demo_interaction_50/) |

[Results for both scopes, including every task](../results/mikasa_robo.md).

New preparations use GPU physics for `ShellGameTouch-VLA-v0`; other tasks default
to CPU. Training and evaluation share this routing. Original scores and snapshots
remain separate from evaluations under the current interface.

## Full 90-task benchmark

The original published setting has 90 policies and 4,500 result records, developed
with 50 demos and 50 interactions per task under CPU physics. Full historical
development/video evidence covers ten tasks; the other 80 are imported results
with incomplete provenance. The original and GPU-updated aggregates are explained
in [results](../results/mikasa_robo.md#full-90-task-benchmark).

### Run a published policy

Install the [dedicated environment](../../envs/mikasa_robo/README.md), then:

```bash
cam prepare runs --solution solutions/mikasa_robo/demo_interaction_50 --tasks SeqOfColors3-VLA-v0
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

For every policy in this setting and all of its held-out episodes:

```bash
python scripts/evaluate_solutions.py solutions/mikasa_robo/demo_interaction_50 \
  --output runs/mikasa_robo_full --gpus 0 1 --workers-per-gpu 5
```

### Develop from demonstrations

```bash
python scripts/develop_batch.py start configs/mikasa_robo/experiments/demo_interaction_50.json \
  --dataset /path/to/downloaded/demos/mikasa_robo --output runs/mikasa_robo \
  --workers 8 --gpus 0 1 --evaluate-on-freeze --evaluation-workers-per-gpu 5
```

Omit `--tasks` for the full benchmark, or append `--tasks SeqOfColors3-VLA-v0` for a single task.
Full held-out evaluation starts as each policy freezes.

## Five-task VLA comparison subset

| Setting | Demos/task | Training attempts/task | Held-out episodes/task |
|---|---:|---:|---:|
| `demo_only_25` | 25 | 0 | 50 |
| `demo_only_50` | 50 | 0 | 50 |
| `demo_interaction_50` | 50 | 50 total, one per training-demo seed | 50 |

The interaction budget covers the entire development process for each task:
**50 simulator attempts using the seeds of the 50 assigned training demos**,
each seed once. Failed attempts consume the same budget, and editing the policy
does not replenish it. Held-out evaluation seeds are separate.
See the shared [experiment budget protocol](../reproducing.md#demonstration-and-interaction-budgets).

The completed settings scored **153/250**, **175/250**, and **250/250**,
respectively. All 15 policies started blank with Claude Code / Opus 5.5 and the
shared prompts plus MIKASA public guidance. The 25-demo arm sees only its exact
materialized assignment. Each arm uses the same held-out identities and native
scoring. All use 256×256 RGB-D, with GPU physics/IK for ShellGameTouch and CPU
physics/IK for the other four tasks.

Evaluate any published setting without demos or model calls:

```bash
python scripts/evaluate_solutions.py solutions/mikasa_robo_vla_suite/demo_interaction_50 \
  --output runs/mikasa_robo_vla_suite_eval --gpus 0 1 --workers-per-gpu 5
```

Replace the final setting with `demo_only_25` or `demo_only_50` as needed.
For one policy, use `cam prepare runs --solution SOLUTION --tasks TASK` followed
by `cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5`.

To develop all three settings from fresh blank policies:

```bash
python scripts/develop_suite.py start configs/mikasa_robo_vla_suite/suite.json \
  --dataset /path/to/downloaded/demos/mikasa_robo --output runs --gpus 0 1
```

Three settings share eight coding workers (3/3/2), training GPU leases and a
serialized evaluation queue with five episode workers per GPU. Each policy's
full evaluation queues immediately after freeze while development continues.
The campaign saves progress, development histories, full rollouts, and exports
exact frozen code/catalogs when each setting completes. Check progress with
`python scripts/develop_suite.py status CAMPAIGN`.

[Individual setting plans](../../configs/mikasa_robo_vla_suite/README.md) ·
[Published solutions](../../solutions/mikasa_robo_vla_suite/)

## Prompt and interface

Shared core + the selected learning mode + MIKASA public interface. Interaction
adds `prompts/mikasa_robo/both.md`. Full-90's SeqOfColors3 also gets its authorized
cue-order clarification. Historical full-90 policies used earlier prompt revisions;
each new preparation pins its exact prompt, and does not replace historical evidence.

[Public interface](../../src/cam/interface/mikasa_robo/guide.md) ·
[Solutions](../../solutions/mikasa_robo/) · [Result provenance](../results/mikasa_robo.md) ·
[Reproduction rules](../reproducing.md)
