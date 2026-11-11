# MIKASA five-task publication comparison

These plans select ShellGameTouch, InterceptMedium and RememberColor3/5/9
(`-VLA-v0` IDs), following the task set in
[SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533v2). The full benchmark's
90-task plan remains in `configs/mikasa_robo/experiments/`. We run both scopes;
the subset supports comparison with prior publications without changing the
full-benchmark denominator.

| Plan | Run configuration | Demos/task | Training attempts/task |
|---|---|---:|---:|
| [25 demos](experiments/demo_only_25.json) | [demos.json](demos.json) | 25 | 0 |
| [50 demos](experiments/demo_only_50.json) | [demos.json](demos.json) | 50 | 0 |
| [50 demos + interaction](experiments/demo_interaction_50.json) | [both.json](both.json) | 50 | 50 |

The separate [`mikasa_robo_vla_suite_250` plan](../mikasa_robo_vla_suite_250/experiments/demo_only_250.json)
uses up to 250 successful RGB-D demos with zero interaction: 237 for ShellGameTouch,
56 for InterceptMedium, 250 each for RememberColor3/5, and 246 for RememberColor9.
It starts fresh policies and records the actual counts in its own run namespace.

The three original settings use `benchmark: "mikasa_robo"` and
`suite: "mikasa_robo_vla_suite"`. The demo count belongs to the experiment plan;
the run configuration defines the learning mode, interaction budget and model.
The 25-demo plan materializes the exact selected subset before learner admission.
Interaction permits 50 training attempts total per task across all code revisions,
using the reset seed of each of the 50 assigned training demos once. Failed
attempts count toward that budget. Policies start blank with
Claude Code / Opus 5.5; held-out evaluation makes no model calls.

The three original settings use the MIKASA environment and its existing
`mikasa_robo-50-demos` package. The up-to-250 plan requires its separately prepared
RGB-D dataset. Configurations retain
the CPU default; the public interface resolves ShellGameTouch to GPU physics/IK
and the other four tasks to CPU, recording the actual backend per task.

```bash
python scripts/develop_suite.py start configs/mikasa_robo_vla_suite/suite.json \
  --dataset /path/to/mikasa_robo/demos --output runs --gpus 0 1
```

[suite.json](suite.json) shares eight coding workers (3/3/2) and GPU leases across
the settings. Full evaluation queues as each policy freezes, with five episode
workers per GPU. Each setting evaluates 5 × 50 episodes; all three total 750.
Use `python scripts/develop_suite.py status CAMPAIGN` for progress.

For a single setting, pass its experiment plan to `scripts/develop_batch.py start`
with `--dataset DATA --output runs/SETTING --workers 8 --gpus 0 1
--evaluate-on-freeze --evaluation-workers-per-gpu 5`. Use the suite controller
for concurrent settings to preserve the shared worker limits.

The published policies are in `solutions/mikasa_robo_vla_suite/<setting>/`.
Their evaluation catalogs in `configs/solutions/mikasa_robo_vla_suite/` import
and freeze unchanged code at R000 with zero new training, retaining the original
development budget, source hashes and results. They are evaluation catalogs,
not fresh-development configurations.

[Reproduction commands](../../docs/reproducing.md#mikasa-five-task-publication-comparison) ·
[Results for both MIKASA scopes](../../docs/results/mikasa_robo.md)
