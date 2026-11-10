# RoboMemArena

The larger assignment has 50 demos per task except T04, which has 32. Interaction uses those same identities exactly once (50/32 attempts). The 25-demo arm uses the revised offline-review prompt; the interaction arm adds manipulation/recovery guidance. These comparisons change prompts as well as data or interaction.

## Run a published policy

Install the [dedicated environment](../../../envs/robomemarena/README.md), then:

```bash
cam prepare validate_runs --solution solutions/robomemarena/demo_interaction_50 --tasks T01_CookiesTomatoBasket
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

For every policy in this setting and all of its held-out episodes:

```bash
python scripts/evaluate_solutions.py solutions/robomemarena/demo_interaction_50 \
  --output validate_runs/robomemarena_full --gpus 0 1 --workers-per-gpu 5
```

## Develop from demonstrations

```bash
python scripts/develop_batch.py start configs/robomemarena/experiments/demo_interaction_50.json \
  --dataset /path/to/downloaded/demos/robomemarena --output validate_runs/robomemarena \
  --workers 8 --gpus 0 1
```

Omit `--tasks` for the full benchmark, or append `--tasks T01_CookiesTomatoBasket` for a single task.
Add `--evaluate-on-freeze --evaluation-workers-per-gpu 5` for automatic testing
of frozen policies. For the 25-demo setting, use `demo_only_25.json` with the
same downloaded pool; preparation extracts its exact `selected_25.json` assignment.

## Prompt and interface

Shared core + shared learning-mode instructions + the RoboMemArena public interface. Interaction also uses `prompts/robomemarena/both.md`. New demo-only preparations include the final offline review.

[Public interface](../../../src/cam/interface/robomemarena/guide.md) ·
[Solutions](../../../solutions/robomemarena/) · [Per-task results and provenance](results.md) ·
[Reproduction rules](../../reproducing.md)
