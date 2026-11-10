# RMBench · demo_only_50

**1,000/1,000 successful held-out episodes (100.00%)**, with complete scored coverage.
This setting contains 10 final policies. Their executable Python bytes are
preserved; schemas and provenance are in the [solution catalog](../../../configs/solutions/rmbench/demo_only_50.json).

The larger assignment has 49 demos for battery_try, 49 for blocks_ranking_try, 48 for observe_and_pickup, and 50 for the other seven tasks. The selected 50-demo swap_blocks policy is the fresh rerun. Earlier outcomes remain archived. Current evaluation uses observation-derived MotionError; historical scores used their pinned interfaces.

## Evaluate

```bash
cam prepare validate_runs --solution solutions/rmbench/demo_only_50 --tasks swap_blocks
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

Install the [RMBench environment](../../../envs/rmbench/README.md) first.
Omit `--tasks` to prepare all policies. No demonstration download or model call
is needed to evaluate frozen code. A new run never replaces the published score.

## Develop this setting

Shared core + shared demo-only instructions + the RMBench public interface. No RMBench-specific motion supplement is selected. Offline robot helpers are available in the interface. Historical prompts and the retired supplement remain archived.

Use `configs/rmbench/experiments/demo_only_50.json`, a downloaded demonstration package,
and the [development commands](../../../docs/benchmarks/rmbench.md).
Each new preparation saves its exact prompt. Historical prompt/configuration
provenance remains attached to the source run; current wording is not a claim
of exact historical reproduction.
