# RoboMemArena

The [original benchmark dataset](https://arxiv.org/html/2605.10921v1#S3.SS3)
provides **100 successful full-task demonstrations per task**, or **2,600 across
26 tasks**. Its subtask segments belong to those trajectories and are not extra
independent demonstrations.

CaM's larger assignment has **50 demos/task**, except **T04, which has 32**.
Demo + interaction permits **50 simulator training attempts total per task**,
one reset using each of the 50 training-demo seeds. T04 instead gets 32 attempts,
one per assigned demo seed. The budget spans all code revisions; failed attempts
count and consumed seeds are not reused. Held-out evaluation seeds are separate.
See the shared [experiment budget protocol](../reproducing.md#demonstration-and-interaction-budgets).

The demo-only alternatives use 25 or up to 50 demos. All settings evaluate 50
held-out episodes per policy. [Results](../results/robomemarena.md) also document
prompt differences between the published settings.

## Run a published policy

Install the [dedicated environment](../../envs/robomemarena/README.md), then:

```bash
cam prepare runs --solution solutions/robomemarena/demo_interaction_50 --tasks T01_CookiesTomatoBasket
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

For every policy in this setting and all of its held-out episodes:

```bash
python scripts/evaluate_solutions.py solutions/robomemarena/demo_interaction_50 \
  --output runs/robomemarena_full --gpus 0 1 --workers-per-gpu 5
```

## Develop from demonstrations

```bash
python scripts/develop_batch.py start configs/robomemarena/experiments/demo_interaction_50.json \
  --dataset /path/to/downloaded/demos/robomemarena --output runs/robomemarena \
  --workers 8 --gpus 0 1 --evaluate-on-freeze --evaluation-workers-per-gpu 5
```

Omit `--tasks` for the full benchmark, or append `--tasks T01_CookiesTomatoBasket` for a single task.
Full evaluation starts as each policy freezes. For the 25-demo setting, use
`demo_only_25.json` with the same downloaded pool; preparation extracts its exact `selected_25.json` assignment.

## Prompt and interface

Shared core + learning-mode instructions + the RoboMemArena public interface.
Interaction adds [`both.md`](../../src/cam/prompts/robomemarena/both.md).

[Public interface](../../src/cam/interface/robomemarena/guide.md) ·
[Solutions](../../solutions/robomemarena/) · [Per-task results and provenance](../results/robomemarena.md) ·
[Reproduction rules](../reproducing.md)
