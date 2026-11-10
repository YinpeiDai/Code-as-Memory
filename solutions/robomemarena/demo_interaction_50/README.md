# RoboMemArena · demo_interaction_50

**1,271/1,300 successful held-out episodes (97.77%)**, with complete scored coverage.
This setting contains 26 final policies. Their executable Python bytes are
preserved; schemas and provenance are in the [solution catalog](../../../configs/solutions/robomemarena/demo_interaction_50.json).

The larger assignment has 50 demos per task except T04, which has 32. Interaction uses those same identities exactly once (50/32 attempts). The 25-demo arm uses the revised offline-review prompt; the interaction arm adds manipulation/recovery guidance. These comparisons change prompts as well as data or interaction.

## Evaluate

```bash
cam prepare validate_runs --solution solutions/robomemarena/demo_interaction_50 --tasks T01_CookiesTomatoBasket
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

Install the [RoboMemArena environment](../../../envs/robomemarena/README.md) first.
Omit `--tasks` to prepare all policies. No demonstration download or model call
is needed to evaluate frozen code. A new run never replaces the published score.

## Develop this setting

Shared core + shared learning-mode instructions + the RoboMemArena public interface. Interaction also uses `prompts/robomemarena/both.md`. New demo-only preparations include the final offline review.

Use `configs/robomemarena/experiments/demo_interaction_50.json`, a downloaded demonstration package,
and the [development commands](../../../docs/benchmarks/robomemarena.md).
Each new preparation saves its exact prompt. Historical prompt/configuration
provenance remains attached to the source run; current wording is not a claim
of exact historical reproduction.
