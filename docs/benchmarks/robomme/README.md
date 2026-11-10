# RoboMME

The published 25-demo experiment used its original runner and prompt. A current-harness rerun is separate evidence. Development accepts the released 25-demo package through a metadata-checked compatibility reader, preserving every H5 byte.

## Run a published policy

Install the [dedicated environment](../../../envs/robomme/README.md), then:

```bash
cam prepare validate_runs --solution solutions/robomme/demo_only_25 --tasks BinFill
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

For every policy in this setting and all of its held-out episodes:

```bash
python scripts/evaluate_solutions.py solutions/robomme/demo_only_25 \
  --output validate_runs/robomme_full --gpus 0 1 --workers-per-gpu 5
```

## Develop from demonstrations

```bash
python scripts/develop_batch.py start configs/robomme/experiments/demo_only_25.json \
  --dataset /path/to/downloaded/demos/robomme --output validate_runs/robomme \
  --workers 8 --gpus 0 1
```

Omit `--tasks` for the full benchmark, or append `--tasks BinFill` for a single task.
Add `--evaluate-on-freeze --evaluation-workers-per-gpu 5` for automatic testing
of frozen policies.

## Prompt and interface

Shared core + shared demo-only instructions + the RoboMME public interface. Historical prompt bytes remain with the source development logs.

[Public interface](../../../src/cam/interface/robomme/guide.md) ·
[Solutions](../../../solutions/robomme/) · [Result provenance](../../results.md) ·
[Reproduction rules](../../reproducing.md)
