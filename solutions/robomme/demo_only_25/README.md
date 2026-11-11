# RoboMME · demo_only_25

**795/800 successful held-out episodes (99.38%)**, with complete scored coverage.
This setting contains 16 final policies. Their executable Python bytes are
preserved; schemas and provenance are in the [solution catalog](../../../configs/solutions/robomme/demo_only_25.json).

The published experiment used its original runner and prompt. Current-harness
reruns are separate evidence; new development uses the released 25-demo package.

## Evaluate

```bash
cam prepare runs --solution solutions/robomme/demo_only_25 --tasks BinFill
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

Install the [RoboMME environment](../../../envs/robomme/README.md) first.
Omit `--tasks` to prepare all policies. No demonstration download or model call
is needed to evaluate frozen code. A new run never replaces the published score.

## Develop this setting

Shared core + demo-only instructions + the RoboMME public interface.

Use `configs/robomme/experiments/demo_only_25.json`, a downloaded demonstration package,
and the [development commands](../../../docs/benchmarks/robomme.md).
Each preparation pins its exact prompt; original runs retain their own provenance.
