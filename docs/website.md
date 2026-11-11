# Experiment website

Build and serve a gallery from your run directory:

```bash
cam site --data-root runs
cam serve runs/website --host 127.0.0.1 --port 8936
```

Open `http://localhost:8936`. Use `public_runs` as the root to browse downloaded
archives. `cam serve` refreshes progress and provides on-demand previews and video
seeking; no simulator or model is needed.

The website includes benchmark results, development logs, demonstration galleries
and evaluation episodes. Cards open a detail page; episode navigation stays on
that page. Failures and episodes without recordings remain visible. Sanity checks
are labelled separately from full evaluations.

By default, the viewer selects the best published setting and latest run per task.
Use **Show all settings** or **Show history** for other experiments, and
**Refresh log** to update an open transcript.

Development cards and task pages show the coding agent/model, API cost and work
time. Running timers update each second; usage refreshes automatically. Work time
counts coding sessions only, with concurrent jobs counted separately in the total.
Live costs use logged tokens and [verified standard list prices](https://assets.claude.com/5633dc848caa431115468f31c7085d695bbfb9cc.pdf)
where available, then use the client's reported total. Estimates are labelled;
missing costs remain unavailable, and simulator compute is excluded.

Generated files stay inside the selected root:

| Path | Contents |
|---|---|
| `dashboard/PROGRESS.md` | Development and evaluation progress |
| `dashboard/index.json` | Artifact index with relative paths |
| `dashboard/results.json` and `results.csv` | Group and per-task results |
| `website/` | Frontend and preview cache |

Keep run metadata, prompts, logs, frozen code, results and referenced media together.
The index supports both generated task runs and public `{demos,dev_logs,eval}/`
archives. After moving a root, rebuild the site to refresh serving paths.
Missing media stays marked unavailable. The viewer serves indexed artifacts only
and is never exposed to the learner.
