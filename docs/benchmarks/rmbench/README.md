# RMBench

The larger assignment has 49 demos for battery_try, 49 for blocks_ranking_try, 48 for observe_and_pickup, and 50 for the other seven tasks. The selected 50-demo swap_blocks policy is the fresh rerun. Earlier outcomes remain archived. Current evaluation uses observation-derived MotionError; historical scores used their pinned interfaces.

## Run a published policy

Install the [dedicated environment](../../../envs/rmbench/README.md), then:

```bash
cam prepare validate_runs --solution solutions/rmbench/demo_only_50 --tasks swap_blocks
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

For every policy in this setting and all of its held-out episodes:

```bash
python scripts/evaluate_solutions.py solutions/rmbench/demo_only_50 \
  --output validate_runs/rmbench_full --gpus 0 1 --workers-per-gpu 5
```

## Develop from demonstrations

```bash
python scripts/develop_batch.py start configs/rmbench/experiments/demo_only_50.json \
  --dataset /path/to/downloaded/demos/rmbench --output validate_runs/rmbench \
  --workers 8 --gpus 0 1
```

Omit `--tasks` for the full benchmark, or append `--tasks swap_blocks` for a single task.
Add `--evaluate-on-freeze --evaluation-workers-per-gpu 5` for automatic testing
of frozen policies. For the 25-demo setting, use `demo_only_25.json` with the
same downloaded pool; preparation extracts its exact `selected_25.json` assignment.

## Prompt and interface

Shared core + shared demo-only instructions + the RMBench public interface. No RMBench-specific motion supplement is selected. Offline robot helpers are available in the interface. Historical prompts and the retired supplement remain archived.

[Public interface](../../../src/cam/interface/rmbench/guide.md) ·
[Solutions](../../../solutions/rmbench/) · [Result provenance](../../results.md) ·
[Reproduction rules](../../reproducing.md)
