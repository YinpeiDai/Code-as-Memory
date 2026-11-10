# Experiment website

Point the website at a run directory. It discovers published archives and fresh
CaM task runs without a manually maintained task list:

```bash
cam site --data-root public_runs
cam serve public_runs/website --host 127.0.0.1 --port 8935

cam site --data-root validate_runs
cam serve validate_runs/website --host 127.0.0.1 --port 8936
```

Use `--output PATH` to place the generated website elsewhere. Existing
`cam site --runs ... --datasets ... --output ...` exports remain supported.
Serve through `cam serve`, which provides lazy previews and video byte ranges.

- **Overview:** live development state, frozen policies, evaluation coverage,
  and benchmark summaries.
- **Results:** complete and partial evaluations, setting comparisons, per-task
  details, and CSV export. Sanity checks are labelled and kept separate.
- **Demonstrations:** episode gallery with camera previews and links to whole H5s.
- **Development:** task gallery; each detail page opens the agent log directly,
  alongside demo, training-rollout, and evaluation samples and links to all.
- **Evaluation:** all recorded outcomes, including failures and records without
  media. Open a card in a new tab; episode numbers and previous/next stay there.

The selected root owns:

```text
dashboard/index.json      Portable index with relative artifact paths
dashboard/results.json   Consolidated group and task counts
dashboard/results.csv    Exportable per-task results
dashboard/PROGRESS.md     Human-readable development/evaluation progress
website/                 Frontend, server allowlists, lazy preview cache
```

The server refreshes metadata as new runs and results appear. It does not interrupt
an open player or agent log; use **Refresh log** to update a transcript. Indexing
reads manifests and JSON records only. Gallery previews read a frame on demand;
missing demo videos are encoded only when opened, with at most two CPU encoders.
Results-only sanity checks have no synthetic thumbnails or videos.

Keep `run.json`, prompts, agent logs, frozen code, results, and referenced media
inside the selected run root. Public archives use
`{demos,dev_logs,eval}/<benchmark>/<setting>/`; fresh runs retain their generated
layout. The index understands both. Copying the whole root and rebuilding the
site regenerates absolute server paths from the portable index. Missing imported
media/logs are shown as unavailable, not inferred from another experiment.

The viewer is a host tool. It is never mounted into a learning sandbox. It serves
only explicitly indexed artifacts and generated website files; hidden server
registries cannot be requested through the static-file route.

The default view uses the best published setting for each benchmark and the latest
run per task/setting. Use **Show all settings**, the setting selector, or **Show
history** to inspect other experiments. Galleries start with one card per task and
setting; selecting a task or run shows its individual episodes. Queued tasks appear
before preparation, so incomplete work is visible without inventing results.
