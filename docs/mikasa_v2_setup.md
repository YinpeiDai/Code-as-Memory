# MIKASA-Robo v2 on another machine

Read [the reproduction rules](reproducing.md) first. The current experiment uses
the same ten selected tasks, 50 replay-derived 256x256 RGB-D demos per task,
Claude Code / Opus 5.5, eight coding jobs and zero training rollouts.
`mikasa_robo_v2` is the CaM adapter name; the official simulator and scoring
remain unchanged.

## Code and environments

Clone or update the repository, including the pinned MIKASA submodule:

```bash
git clone --recurse-submodules https://github.com/YinpeiDai/Code-as-Memory.git
cd Code-as-Memory
# Existing checkout: git pull --ff-only, then update submodules below.
```

On Linux, install `uv`, Bubblewrap (`bwrap`), `strace`, `ffmpeg`, and `xz`.
Install and authenticate Claude Code separately. From the checkout:

```bash
git submodule update --init --recursive
uv sync --locked --extra dev
UV_PROJECT_ENVIRONMENT=$PWD/envs/mikasa_robo/.venv \
  uv sync --frozen --no-dev --project third_party/mikasa_robo --python 3.11
MS_ASSET_DIR=$PWD/data/raw/mikasa_robo/maniskill_data \
  envs/mikasa_robo/.venv/bin/python -I scripts/download_mikasa_robo_assets.py
```

Both MIKASA adapters share this simulator environment. For a custom installation,
set `CAM_MIKASA_ROBO_V2_PYTHON` to its Python executable and
`CAM_MIKASA_ROBO_ASSETS` to the asset directory. Keep simulator dependencies
separate from the root CaM environment. GPU/Vulkan is needed for native replay
and evaluation; demo-only coding performs no native simulator rollouts.
See [the environment guide](../envs/mikasa_robo/README.md) for pinned versions.

## Demonstrations and account settings

The completed replay dataset is already on shared NFS:

```text
/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/data/mikasa_robo_v2_50
```

Point the new machine's plan directly at that directory if it mounts the same
NFS. It contains 90 tasks with 50 demos each (4,500 episodes); the plan exposes
only the selected task's 50 demos to each learner. Keep new runs and outputs
under the new checkout. Do not reuse old prepared run directories or copy virtual
environments. Data, credentials, logs and runs stay outside Git.
The prepared data already contain rendered depth; raw RLDS is needed only to
regenerate demonstrations. Preparation verifies hashes and training identities.

With a different NFS mount, use its local absolute path. For a local copy,
preserve `manifest.json`, referenced H5 files, relative paths and bytes. Copy
actual files if the original directory is a storage symlink.

For a smaller transfer, use the existing ten per-task dataset directories under
`/data/daiyp/CaM/datasets/mikasa_v2_demos50_local_20261108/` on the original machine.
Each has its own manifest and exactly 50 assigned demos. List their new locations
in the local plan's `datasets` array; this also avoids validating the entire
90-task pool for a ten-task experiment. Preserve the original manifests.

```bash
mkdir -p local
cp configs/mikasa_robo_v2/demos_claude_code.json local/mikasa_demos50.json
cp configs/experiments/mikasa_v2_demos50_10tasks.json local/mikasa_demos50_plan.json
```

In the local plan, set `benchmarks.mikasa_robo_v2.config` to
`local/mikasa_demos50.json` and set `datasets` to the shared NFS path above (or
your local dataset paths). Reading and hashing the full 90-task pool over NFS
can make preparation slow; the per-task copies avoid that full-pool scan.
In the local config, set `coding.aws_profile` and `coding.aws_region` for your
Bedrock account. Keep the requested client/model, `mode: "demos"`, `budget: 0`,
`depth: true` and `builtin_demo: false`. Credentials belong in the client's
authentication setup, never in the plan or learner workspace.

## Launch and inspect

```bash
.venv/bin/python scripts/develop_batch.py start local/mikasa_demos50_plan.json \
  --workers 8 --gpus 0 1
```

Use the GPU IDs present on the new machine. The command prints a new batch path
and launches development in the background. Each task is prepared under a fresh
timetag and gets its own README, isolated workspace, logs and code revisions.
Policies freeze to R000 on completion. The learner may use the pure offline
`safety_check` API; live training rollouts are disabled. Evaluation is a separate
host action after policy freeze and is not launched by this command.

Replace `BATCH` with the printed batch path:

```bash
.venv/bin/python scripts/develop_batch.py status BATCH
.venv/bin/cam site --batch BATCH --output sites/mikasa_demos50
.venv/bin/cam serve sites/mikasa_demos50 --host 127.0.0.1 --port 8914
```

Forward port 8914 over SSH and open `http://127.0.0.1:8914/#development` for
agent logs, source revisions and reference demo videos.
