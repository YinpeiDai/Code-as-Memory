# RoboMME · demo_only_25

**795/800 successful held-out episodes (99.38%)**, with complete scored coverage.
This setting contains 16 final policies. Their executable Python bytes are
preserved; schemas and provenance are in the [solution catalog](../../../configs/solutions/robomme/demo_only_25.json).

The published 25-demo experiment used its original runner and prompt. A current-harness rerun is separate evidence. Development accepts the released 25-demo package through a metadata-checked compatibility reader, preserving every H5 byte.

## Evaluate

```bash
cam prepare validate_runs --solution solutions/robomme/demo_only_25 --tasks BinFill
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

Install the [RoboMME environment](../../../envs/robomme/README.md) first.
Omit `--tasks` to prepare all policies. No demonstration download or model call
is needed to evaluate frozen code. A new run never replaces the published score.

## Develop this setting

Shared core + shared demo-only instructions + the RoboMME public interface. Historical prompt bytes remain with the source development logs.

Use `configs/robomme/experiments/demo_only_25.json`, a downloaded demonstration package,
and the [development commands](../../../docs/benchmarks/robomme.md).
Each new preparation saves its exact prompt. Historical prompt/configuration
provenance remains attached to the source run; current wording is not a claim
of exact historical reproduction.
