# MIKASA-Robo v2 on another machine

The current follow-up is **demo + interaction v2 with CPU physics**: 50 replayed
256x256 RGB-D demos and 50 rollouts per task, Claude Code / Opus 5.5, ten coding
jobs. Rollouts use exactly the assigned demo identities/seeds, once each.
The host selects the next identity and rejects overrides and attempts beyond 50.
The prompt permits unlimited offline demo review and requires every remaining
rollout for exploration or verification, even after success.

Use [the remaining-80 plan](../configs/experiments/mikasa_v2_cpu_50_demoseeds_remaining80.json).
It excludes the ten tasks already assigned to the original machine. See the
[task inventory](mikasa_v2_progress.md) and [reproduction rules](reproducing.md).
New preparations use the concise MIKASA supplement; the running ten-task batch
retains its earlier, longer prompt. Record that prompt difference in comparisons.

## Exact replay-data location

```text
/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/data/mikasa_robo_v2_50/
  manifest.json
  h5/record_dataset_<Task-ID>.h5
```

This is the completed **90-task / 4,500-episode** pool, with 50 RGB-D demos per
task. For example, the remaining-80 selection starts with:

```text
/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/data/mikasa_robo_v2_50/h5/record_dataset_BatteriesCheckerEasy-6-VLA-v0.h5
```

The plan already points at this NFS directory. The new machine must mount it;
otherwise update `datasets` in a local copy of the plan. You need the manifest
and referenced H5 files, not raw RLDS. A local copy must preserve relative paths,
file contents and manifests. Preparation validates hashes and identities before
learning, which can take time over NFS. Each learner receives only its task's
50-demo H5. Data, runs, logs and credentials are excluded from Git.

## Experiment storage on NFS

All new experiment artifacts for this machine live under:

```text
/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/mikasa_robo/
  runs/mikasa_robo_v2/<task>/<agent-model>/demo_interaction_v2/<timetag>/
  outputs/development_batches/<timetag>/
  data/development_batches/<timetag>/
  sites/mikasa_cpu_both50/
```

The checkout's `runs/`, `outputs/`, `data/`, and `sites/` are directory symlinks
to the matching NFS directories. This is configured in
`/home/daiyp/Code-as-Memory`. Run directories contain the learner workspaces,
source revisions, frozen policies, logs, audits, training/test rollouts and
reports. Batch snapshots and supervisor logs live under `outputs/`; generated
data and downloaded assets under `data/`; dashboard files and video caches under
`sites/`. Source code and Python environments stay in the local checkout. The
original replay pool remains at the separate NFS path above, referenced directly
by the remaining-80 plan.

For another fresh checkout, configure storage **before** downloading assets or
launching experiments. These commands require all four checkout paths to be
absent; preserve existing directories and historical run paths instead of
replacing them:

```bash
mikasa_store=/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/mikasa_robo
for name in runs outputs data sites; do
  mkdir -p "$mikasa_store/$name"
  ln -sT "$mikasa_store/$name" "$name"
done
```

The launcher and `cam prepare` resolve these links before recording paths, so
batch manifests and generated run READMEs use the absolute NFS locations. Keep
the standard relative commands below; their artifacts go to NFS. The learner
still sees only its assigned sandbox mounts. Record each machine's task
assignment and batch tag in the inventory before launching additional batches.

## Install code and environments

For an existing checkout:

```bash
git pull --ff-only
git submodule update --init --recursive third_party/mikasa_robo
git submodule update --init third_party/robomme_benchmark
uv sync --locked --extra dev
UV_PROJECT_ENVIRONMENT=$PWD/envs/mikasa_robo/.venv \
  uv sync --frozen --no-dev --project third_party/mikasa_robo --python 3.11
MS_ASSET_DIR=$PWD/data/raw/mikasa_robo/maniskill_data \
  envs/mikasa_robo/.venv/bin/python -I scripts/download_mikasa_robo_assets.py
```

For a fresh checkout, first clone
`https://github.com/YinpeiDai/Code-as-Memory.git`. On Linux, install `uv`,
Bubblewrap (`bwrap`), `strace`, `ffmpeg`, and `xz`, plus working GPU/Vulkan drivers.
CPU physics still needs a GPU for rendering. Keep CaM's and the simulator's
Python environments separate; see [environment details](../envs/mikasa_robo/README.md).
The root lock includes RoboMME as an optional local dependency, so `uv sync
--locked` also needs its submodule metadata even when only MIKASA is run.
Custom installations may set `CAM_MIKASA_ROBO_V2_PYTHON` and
`CAM_MIKASA_ROBO_ASSETS`.

Install and authenticate Claude Code. The supplied
[CPU config](../configs/mikasa_robo_v2/cpu/both_v2_claude_code.json) uses Bedrock
profile `sagemaker` in `us-east-1`. If your account differs, copy the config and
plan into `local/`, edit `coding.aws_profile` / `coding.aws_region`, and point the
plan's `benchmarks.mikasa_robo_v2.config` at that local config. Keep the model,
`physics_backend: "cpu"`, `budget: 50`, `training_seed_count: 50`,
`training_seed_policy: "demo_cycle"`, and RGB-D settings unchanged. Credentials
belong in client authentication, never in the plan or learner workspace.

New Claude sessions first probe the exact pinned CaM MCP server with
initialization, discovery of all six tools and a read-only `usage` call, before
paid inference. Evidence is in `logs/session_*/mcp_probe/`; this host check
consumes no training attempts. Client initialization must also report the six
tools and a connected CaM server. The process-lifetime watcher supports Python
builds without `os.pidfd_open`. MCP startup, synthetic transport preflight and
isolated policy-interface validation allow up to 120 seconds for NFS startup;
this does not change simulator budgets or policy execution limits. Authentication
and a model-only READY response do not verify the tool connection. Prepare fresh
runs after these harness changes, preserving failed runs and their logs.

## Launch and inspect

```bash
.venv/bin/python scripts/develop_batch.py start \
  configs/experiments/mikasa_v2_cpu_50_demoseeds_remaining80.json \
  --workers 10 --gpus 0 1 --evaluate-on-freeze --evaluation-workers-per-gpu 2
```

Use available rendering GPU IDs on the new machine. Ten coding sessions run
concurrently, with one live simulator per selected GPU. Each task gets a fresh
run, blank policy, exact prompt, access-boundary check, command audit, revisions
and rollout videos. The supervisor continues through all 80 tasks and freezes
completed policies. With `--evaluate-on-freeze`, a separate host queue checks
successful learning completion, all 50 training attempts, and frozen-policy /
R050 checkpoint integrity, then evaluates all 50 held-out episodes for that task.
One task is evaluated at a time with four episode workers, two per GPU, alongside
the training simulators. This is 4,000 expected test episodes across 80 tasks.
Test results stay outside learner feedback. Without the opt-in flag the launcher
only develops policies. Record the new batch tag and machine in the inventory
to avoid duplicate assignments.

On `sled-aspen`, `local/mikasa_cpu_80_plan.json` uses one dataset directory per
task under `data/mikasa_v2_task_partitions_<timetag>/`. These are complete 50-demo
assignments: H5 files and preview videos are internal hardlinks to the unchanged
NFS originals, with the original H5 hashes and episode records retained. Each
new manifest records the source manifest hash and partition provenance.
`cam prepare` validates every assigned H5's bytes, schema and identities. Use
the local plan and add `--prepare-workers 4` to validate four task datasets
concurrently and admit each learner as its preparation completes, avoiding
repeated scans of unrelated tasks. No source manifest or recording is edited.

Replace `BATCH` with the printed batch path:

`/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/mikasa_robo/outputs/development_batches/<timetag>`.

```bash
.venv/bin/python scripts/develop_batch.py status BATCH
.venv/bin/cam site --batch BATCH --output sites/mikasa_cpu_both50
.venv/bin/cam serve sites/mikasa_cpu_both50 --host 127.0.0.1 --port 8916
```

Forward port 8916 over SSH for the development logs, reference demos, code and
rollout videos. Batch `status.json` records `evaluation_status` and final
scored/expected counts separately from learning. During evaluation, per-episode
progress is in `RUN/queues/evaluation/status.json`; failures retain evidence and
partial counts for diagnosis and `cam recover-test` when appropriate.
For a frozen run that was launched without automatic evaluation, use:

```bash
.venv/bin/python RUN/runner.py evaluate RUN --budgets 50 --gpus 0 1 --workers-per-gpu 2
```

The pinned run config selects CPU physics for both development and testing;
each result records its actual backend. GPU rendering stays 256x256 RGB-D.
Native scoring is unchanged. Keep CPU results separate from the default GPU
protocol: backend changes also change IK and may change scenes for the same seed.
The offline `safety_check` remains an approximation; live `robot.dry_run` uses
the active native controller.

## Earlier protocols

The demo-only plans `mikasa_v2_demos50_10tasks.json` and
`mikasa_v2_demos50_remaining80.json` retain zero training attempts and checkpoint
R000. The GPU/new-seed plan is `mikasa_v2_50_newseeds_10tasks.json`.
To prepare a separate CPU evaluation of an unchanged frozen demo-only policy, use
`scripts/prepare_mikasa_cpu_evaluation.py SOURCE_RUN`, then evaluate its printed
new run with `--budgets 0`. Preserve original policies, prompts and scored results.
