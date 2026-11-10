# MIKASA-Robo

Each task uses 50 demos and one interaction on each of those exact identities, with CPU physics. All 90 policies and 4,500 result records are available. Full development/video evidence covers the initial ten tasks; the other 80 are imported results. Their missing historical source snapshots and reset-seed evidence are not independently verified.

New preparations use GPU physics for `ShellGameTouch-VLA-v0`; other tasks default
to CPU. Training and evaluation share this routing. Original scores and snapshots
remain separate from evaluations under the current interface.

## Run a published policy

Install the [dedicated environment](../../../envs/mikasa_robo/README.md), then:

```bash
cam prepare validate_runs --solution solutions/mikasa_robo/demo_interaction_50 --tasks SeqOfColors3-VLA-v0
# RUN is the printed task run path.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

For every policy in this setting and all of its held-out episodes:

```bash
python scripts/evaluate_solutions.py solutions/mikasa_robo/demo_interaction_50 \
  --output validate_runs/mikasa_robo_full --gpus 0 1 --workers-per-gpu 5
```

## Develop from demonstrations

```bash
python scripts/develop_batch.py start configs/mikasa_robo/experiments/demo_interaction_50.json \
  --dataset /path/to/downloaded/demos/mikasa_robo --output validate_runs/mikasa_robo \
  --workers 8 --gpus 0 1
```

Omit `--tasks` for the full benchmark, or append `--tasks SeqOfColors3-VLA-v0` for a single task.
Add `--evaluate-on-freeze --evaluation-workers-per-gpu 5` for automatic testing
of frozen policies.

## Prompt and interface

Shared core + demo-first interaction mode + MIKASA public interface + `prompts/mikasa_robo/both.md`. SeqOfColors3 also gets its explicitly documented cue-order clarification. Initial-ten and imported policies used different historical prompt revisions; do not substitute the current prompt hash for either.

[Public interface](../../../src/cam/interface/mikasa_robo/guide.md) ·
[Solutions](../../../solutions/mikasa_robo/) · [Result provenance](../../results.md) ·
[Reproduction rules](../../reproducing.md)
