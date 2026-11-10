# Run published solutions

Every setting in `solutions/<benchmark>/<setting>/` contains final Python code,
plus a README with its results and prompt provenance. Required memory schemas,
file hashes, interface identities, evaluation configuration, and source-run
provenance live in `configs/solutions/<benchmark>/`.

Install the common harness and matching [simulator environment](../envs/README.md), then:

```bash
cam prepare validate_runs --solution solutions/rmbench/demo_only_50 --tasks swap_blocks
# RUN is the new task run printed by prepare.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

Omit `--tasks` to prepare every published policy in a setting. Preparation verifies
code hashes, test identities, simulator source, and the public interface, imports
unchanged code, and freezes R000. No demos, learning, or model calls are involved.
Each preparation is a new evaluation run; source-run provenance remains explicit.

RMBench automatically selects the `*_observed_motion.json` compatibility catalog
for the current interface. Historical catalogs remain intact. The current
`MotionError` compares requested targets with public post-action EEF poses;
it exposes no native planner flags. This is a new-interface evaluation, not a
relabelled historical score.

For compact recordings, add `--recording results` to `cam prepare --solution`.
For a five-episode check of every solution:

```bash
cam check-solutions --output validate_runs/eval_sanity_check \
  --episodes 5 --gpus 0 1 --workers-per-gpu 12
```

Use `--benchmarks rmbench` and/or `--tasks swap_blocks` to select a subset.
The batch saves `status.json`, `results.json`, `PROGRESS.md`, pinned preparations,
per-episode logs, and results. It writes no video, observation H5, array dump, or
syscall trace. `--resume BATCH` preserves completed episodes and continues an
interrupted check. A `STOP.json` file in the batch stops new admission after active
episodes finish. A passing runtime check means the policy executed without a
policy/infrastructure error; task success is reported separately.

## All tasks in a setting

```bash
python scripts/evaluate_solutions.py solutions/robomme/demo_only_25 \
  --output validate_runs/robomme_full --gpus 0 1 --workers-per-gpu 5
```

Replace the solution path for another benchmark or setting. The launcher prepares
every policy and evaluates all held-out identities, with one shared concurrency
limit. It records its batch under the output directory's `batches/`; use
`--resume BATCH` after an interruption. Archive infrastructure failures with
`cam recover-test` before retrying them. Scored task failures are retained.
