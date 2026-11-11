# RMBench

Ten tasks, with 25 or up to 50 demonstrations and no training interaction.
The larger pool has 49 demos for `battery_try` and `blocks_ranking_try`, 48 for
`observe_and_pickup`, and 50 for the other seven tasks. Evaluation uses 100
held-out episodes per policy.

## Run a published policy

Install the [dedicated environment](../../envs/rmbench/README.md), then:

```bash
cam prepare runs --solution solutions/rmbench/demo_only_50 --tasks swap_blocks
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

For every policy in this setting and all of its held-out episodes:

```bash
python scripts/evaluate_solutions.py solutions/rmbench/demo_only_50 \
  --output runs/rmbench_full --gpus 0 1 --workers-per-gpu 5
```

## Develop from demonstrations

```bash
python scripts/develop_batch.py start configs/rmbench/experiments/demo_only_50.json \
  --dataset /path/to/downloaded/demos/rmbench --output runs/rmbench \
  --workers 8 --gpus 0 1 --evaluate-on-freeze --evaluation-workers-per-gpu 5
```

Omit `--tasks` for the full benchmark, or append `--tasks swap_blocks` for a single task.
Full evaluation starts as each policy freezes. For the 25-demo setting, use
`demo_only_25.json` with the same downloaded pool; preparation extracts its exact `selected_25.json` assignment.

## Prompt and interface

Shared core + demo-only instructions + the RMBench public interface. Offline
robot helpers support development. `MotionError` uses public post-action poses;
see [interface adaptations](../interface.md).

[Public interface](../../src/cam/interface/rmbench/guide.md) ·
[Solutions](../../solutions/rmbench/) · [Per-task results and provenance](../results/rmbench.md) ·
[Reproduction rules](../reproducing.md)
