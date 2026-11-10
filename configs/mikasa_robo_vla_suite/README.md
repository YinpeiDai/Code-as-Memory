# MIKASA five-task publication comparison

These plans select ShellGameTouch, InterceptMedium and RememberColor3/5/9
(`-VLA-v0` IDs), following the task set in
[SimpleMemVLA, Table 3](https://arxiv.org/pdf/2609.05533v2). The full benchmark's
90-task plan remains in `configs/mikasa_robo/experiments/`. We run both scopes;
the subset supports comparison with prior publications without changing the
full-benchmark denominator.

| Plan | Client configuration | Demos/task | Training attempts/task |
|---|---|---:|---:|
| [25 demos](experiments/demo_only_25.json) | [demos_claude_code.json](demos_claude_code.json) | 25 | 0 |
| [50 demos](experiments/demo_only_50.json) | [demos_claude_code.json](demos_claude_code.json) | 50 | 0 |
| [50 demos + interaction](experiments/demo_interaction_50.json) | [both_claude_code.json](both_claude_code.json) | 50 | 50 |

Every configuration uses `benchmark: "mikasa_robo"` and
`suite: "mikasa_robo_vla_suite"`. The demo count belongs to the experiment plan;
the client configuration defines the learning mode, interaction budget and model.
The 25-demo plan materializes the exact selected subset before learner admission.
Interaction uses each assigned demo identity once. Policies start blank with
Claude Code / Opus 5.5; held-out evaluation makes no model calls.

Use the MIKASA environment and its existing `mikasa_robo-50-demos` package.
No separate simulator or dataset download is required. Configurations retain
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
[Results for both MIKASA scopes](../../docs/benchmarks/mikasa_robo/results.md)
