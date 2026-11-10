# MIKASA-Robo five-task comparison · demo_interaction_50

**250/250 successful held-out episodes (100.0%)**, all 250 expected episodes scored.
Five fresh policies use 50 demonstrations and 50 training attempts per task.
This subset follows the task selection in prior VLA publications; it remains part
of MIKASA and is reported separately from the full 90-task benchmark.

| Task | Successful / scored / expected |
|---|---:|
| `ShellGameTouch-VLA-v0` | 50 / 50 / 50 |
| `InterceptMedium-VLA-v0` | 50 / 50 / 50 |
| `RememberColor3-VLA-v0` | 50 / 50 / 50 |
| `RememberColor5-VLA-v0` | 50 / 50 / 50 |
| `RememberColor9-VLA-v0` | 50 / 50 / 50 |

## Evaluate

Install the [MIKASA environment](../../../envs/mikasa_robo/README.md), then run all five policies:

```bash
python scripts/evaluate_solutions.py solutions/mikasa_robo_vla_suite/demo_interaction_50 \
  --output runs/mikasa_robo_vla_suite_eval/demo_interaction_50 --gpus 0 1 --workers-per-gpu 5
```

To prepare one task, use `cam prepare runs --solution solutions/mikasa_robo_vla_suite/demo_interaction_50 --tasks ShellGameTouch-VLA-v0`,
then `cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5` with the printed run path.
Evaluation imports unchanged code, freezes R000, and uses no demos, learning or model calls.
ShellGameTouch uses GPU physics/IK; the other four tasks use CPU physics/IK.
All policies use 256×256 RGB-D and GPU rendering.

## Development provenance

Policies started blank with Claude Code / Opus 5.5, shared core and learning-mode
prompts, and MIKASA public guidance. The interaction setting adds the documented
MIKASA demo-first supplement and uses each assigned identity once. The demo-only
settings receive no simulator training feedback. The 25-demo assignment contains
only its selected identities. Held-out diagnostics never enter learner feedback.

Executable bytes and the [catalog](../../../configs/solutions/mikasa_robo_vla_suite/demo_interaction_50.json)
are copied unchanged from campaign `20261110T190922_491466Z`. The catalog pins
memory schemas, code checksums, source runs/checkpoints, prompt/harness hashes,
physics backends, assigned budgets and original results.

[MIKASA results and comparison scope](../../../docs/benchmarks/mikasa_robo/results.md) ·
[Develop both MIKASA scopes](../../../docs/benchmarks/mikasa_robo/README.md)
