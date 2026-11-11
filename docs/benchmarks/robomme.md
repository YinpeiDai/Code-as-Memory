# RoboMME

The [original training release](https://github.com/RoboMME/robomme_benchmark)
provides **100 demonstrations per task**, or **1,600 across 16 tasks**.
CaM uses **25 demos/task**, no training interaction and 50 held-out episodes per
policy. Use the prepared CaM demo package without replay.

## Run a published policy

Install the [dedicated environment](../../envs/robomme/README.md), then:

```bash
cam prepare runs --solution solutions/robomme/demo_only_25 --tasks BinFill
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

For every policy in this setting and all of its held-out episodes:

```bash
python scripts/evaluate_solutions.py solutions/robomme/demo_only_25 \
  --output runs/robomme_full --gpus 0 1 --workers-per-gpu 5
```

## Develop from demonstrations

```bash
python scripts/develop_batch.py start configs/robomme/experiments/demo_only_25.json \
  --dataset /path/to/downloaded/demos/robomme --output runs/robomme \
  --workers 8 --gpus 0 1 --evaluate-on-freeze --evaluation-workers-per-gpu 5
```

Omit `--tasks` for the full benchmark, or append `--tasks BinFill` for a single task.
Full evaluation starts as each policy freezes.

## Prompt and interface

Shared core + demo-only instructions + the RoboMME public interface. Each
preparation pins its exact prompt.

[Public interface](../../src/cam/interface/robomme/guide.md) ·
[Solutions](../../solutions/robomme/) · [Per-task results and provenance](../results/robomme.md) ·
[Reproduction rules](../reproducing.md)
