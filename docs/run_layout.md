# Data, code history, and recovery

New experiments use:

```text
runs/<benchmark>/<task>/<agent-model>/<setting>/<timetag>/
```

`cam prepare` prints one leaf per task; pass those leaves to subsequent commands.
Settings and access rules are in [reproducing.md](reproducing.md) and each run's
README. Data, runs, local files, and generated sites are ignored by Git.
Demo-plus-interaction runs use `demo_interaction_v1` or `demo_interaction_v2` as
their setting; the exact version also appears in the config, prompt, and reports.
See [version selection](demo_interaction.md). Historical unversioned runs keep
their original `demo_interaction` paths and pinned evidence.

## Saved artifacts

```text
data/<extraction>/
  manifest.json                     selection, provenance, hashes
  audit/<Task>.json                  raw identity and completion checks
  h5/record_dataset_<Task>.h5        assigned sanitized evidence
  rollouts/<Task>/episode_N.mp4      demonstration previews
  interrupted/                      incomplete outputs preserved on resume
RUN/
  README.md                         settings, commands, access boundary
  run.json, prepared.json            configuration and provenance
  runner.py                         pinned harness entry point
  inputs/harness/cam/                harness snapshot
  inputs/benchmark/<package>/        private official simulator snapshot
  inputs/metadata/{train,test}/      private episode identities
  inputs/environment.json, uv.lock   dependency provenance
  inputs/simulator_environment.json dedicated simulator interpreter/packages, if used
  inputs/simulator_build/            available simulator locks and install recipes
  runtime/cam/                      public policy API
  preflight.json                    declared sensor transport check
  queues/{learning,evaluation}/     worker logs and queue status
  tasks/<Task>/
    prompt.md                       composed assignment
    workspace/                      current policy, analysis, scratch, API reference
    program/                        synchronized source, never executed on host
    demonstrations/                 assigned H5; absent in interaction-only mode
    state/config.json, ledger.sqlite private settings and accounting
    logs/session_NNN/               launch, prompt, events.jsonl, stderr, usage, completion
    logs/usage.json                  usage across retained sessions
    audit/call_*                     tool arguments/results, stdout/stderr, strace
    audit/objects/, file_manifests/  intermediate workspace contents
    history/{objects,revisions}/    source snapshots, metadata, diffs
    checkpoints/RNNN/               immutable code and freeze metadata
    policy/                         final frozen solution
    feedback/RNNN/                  permitted training feedback
    rollouts/train/RNNN/            counted training evidence
    rollouts/test/RNNN/epNNN/        held-out episode evidence
    rollouts/infrastructure_retries/ archived infrastructure failures
  reports/summary.json, episodes.csv aggregate and episode results
sites/<site>/
  index.html, app.js, style.css      frontend
  data.json, development/*.json     display records
  figures/                          SVG, PNG, PDF exports
  .artifacts.json, .sources.json    private server registries; not served
```

Each rollout saves `result.json` and, when produced, `observations.h5`,
`episode.mp4`, actions, decisions, policy stderr, and private diagnostics.
Initialization failures still save a result. Simulator internals and private
tracebacks are excluded from learner feedback. Sandbox `/workspace/scratch`
maps to `RUN/tasks/<Task>/workspace/scratch` on the host.

## Versions versus checkpoints

`history/revisions/NNNNNN.json` records time, reason, attempt count, and source
digest; the matching `.diff` shows changes. Full policy files live in
`history/objects/<digest>/`. Identical content shares an object. Commands,
successful validation, and explicit snapshots save revisions automatically.
`audit/objects/` and `file_manifests/` also retain scratch files and incomplete
edits before/after tool calls; the full transcript is `logs/session_NNN/events.jsonl`.

Checkpoints are for evaluation. In demo-only mode, `checkpoints/R000/` and
`policy/` appear at final freeze, so their absence during development is normal.
Interaction checkpoint R_N includes changes after N attempts and before N+1;
the final checkpoint carries forward when a larger evaluation budget is requested.
Every simulator initialization, crash, or early stop spends an attempt. Offline
analysis and static validation spend none.

## Resume and recovery

Prepared inputs remain pinned even when the checkout changes; commands dispatch
through the run's `runner.py`. Changed inputs require a fresh timetag.
When shortening an existing folder name, retain a compatibility symlink for
recorded absolute paths; do not rewrite pinned metadata or historical logs.

| Command | Behavior |
|---|---|
| `cam learn RUN --resume` | Append a session for unfinished tasks, retaining code, costs, evidence, seed order, and remaining budget; skip frozen tasks |
| `cam evaluate RUN --resume` | Skip verified scored episodes with matching code hashes, including scored failures |
| `cam recover-learning RUN TASK --reason '...'` | Close interrupted attempts without refunding budget; learning/service locks must be free |
| `cam recover-test RUN TASK EPISODE --budget N --reason '...'` | Archive an incomplete/infrastructure episode before identical reevaluation; queue must be stopped; scored results cannot be replaced |

Infrastructure errors stop new evaluation dispatch; running jobs finish and keep
their evidence. Archive interrupted outputs through recovery instead of deleting
them. Workers follow their parent process lifetime to prevent orphaned attempts.

`cam serve` refreshes linked records in the background at most every 20 seconds (a page load never waits for the rebuild) and serves
only registered artifacts.

The website opens with a visual rollout library. Task cards and attempt links
open independent replay tabs; the dashboard and other videos stay in place.

- **Rollout replay** (click a task card or an attempt). Each
  camera is drawn as its own tile from `episode.mp4`. The tiles carry camera
  labels, projected TCP/end-effector markers with a short trail (only cameras
  with recorded `<camera>_K`/`<camera>_extrinsic`), the active decision, and
  the outcome on the last frame. Alongside them are a decision timeline,
  labeled action stages, decision cards (arguments, memory changes, outcome),
  and the frame-aligned command and robot state. Phase/stage/note labels use
  recorded policy memory; they do not assert intermediate simulator success.
  Click a stage to seek, or expand the player to fill the window.
  - Frame 0 is the reset observation; frame i ≥ 1 is the observation after command i−1.
  - The overlays are built by `reporting/trace.py`, only from the rollout's
    `observations.h5`, `actions.jsonl`, `decisions.jsonl` and `result.json`
    (`dry_runs.jsonl` lists `robot.dry_run` queries where a benchmark supports them)
    (no simulator state).
  - Traces and camera previews are cached under `<site>/traces/` and
    `<site>/previews/` by source modification time.
  - `#rollout=<key>` links reopen the same rollout.
- **Development log** (`reporting/devlog.py`). A chronological transcript of
  each coding session, in order: agent messages, thinking (when the client
  records it), every CaM tool call with arguments, output and duration,
  images passed to `read_image` (from `audit/objects`), code revisions with
  inline diffs, checkpoint markers, rollout cards that open the explorer, and
  each session's cost/turn summary. Tabs show the exact prompt, training rollouts,
  and code versions: latest revision, final frozen policy, and the checkpoint
  actually executed in each rollout (verified against its program hash).
  **Refresh log** reloads an open task. Use `cam site --batch BATCH --output SITE`
  to include newly prepared runs automatically. Sites reference media in the original run/dataset
paths; they are not self-contained exports. Preserve those paths or regenerate
the site after moving the artifacts.
