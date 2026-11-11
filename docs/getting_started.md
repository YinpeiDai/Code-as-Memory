# Installation and data

The root Python 3.11 environment contains the shared harness, policy runtime,
analysis libraries, and website. Each simulator uses its own environment:

| Benchmark | Install from the repository root |
|---|---|
| RoboMME | `uv sync --locked --project envs/robomme --python 3.11` |
| MIKASA-Robo | `bash envs/mikasa_robo/install.sh` |
| RMBench | `bash envs/rmbench/install.sh` |
| RoboMemArena | `uv sync --locked --project envs/robomemarena --python 3.10` |

```bash
git clone https://github.com/YinpeiDai/Code-as-Memory.git
cd Code-as-Memory
git submodule update --init
uv sync --locked --inexact --extra dev
source .venv/bin/activate
```

For an existing checkout, run `git submodule update --init` first. CaM needs only
the four top-level simulator submodules. Nested baseline repositories, including
SimpleMemVLA, are unused; recursive initialization is unnecessary.
Install system `bubblewrap`, `strace`, and `ffmpeg`.
NVIDIA/Vulkan or EGL support is required by the chosen simulator. Each
[environment README](../envs/README.md) describes assets and interpreter overrides.
`cam doctor --simulator` checks the four dedicated environments without resetting
simulators or contacting a model.

## Demonstration packages

All four packages are published and transfer-verified. `cam download-demos`
uses the immutable revision and metadata hashes recorded in
`configs/<benchmark>/datasets/release.json`.

We replay the released MIKASA-Robo and RoboMemArena expert trajectories in their
simulators to capture aligned **256×256 RGB-D demonstrations** through the same
public observation interface used at rollout time. This gives CaM consistent RGB,
depth, camera calibration and robot-state inputs. RoboMME retains its released
RGB-D recordings; RMBench regenerates official-expert trajectories at the released
training seeds and captures RGB-D at its native **320×240** resolution.
The downloadable packages already contain these prepared demonstrations; users
do not need to replay trajectories during installation or experiment preparation.

Set `DATA` to a downloaded, extracted **prepared** package:

```text
DATA/
  manifest.json
  h5/record_dataset_<Task>.h5
  rollouts/                         optional preview videos
```

| Benchmark | Public dataset | Released demonstrations |
|---|---|---:|
| RoboMME | [robomme-25-demos](https://huggingface.co/datasets/Yinpei/robomme-25-demos) | 400 (25/task) |
| MIKASA-Robo | [mikasa_robo-50-demos](https://huggingface.co/datasets/Yinpei/mikasa_robo-50-demos) | 4,500 (50/task) |
| RMBench | [rmbench-50-demos](https://huggingface.co/datasets/Yinpei/rmbench-50-demos) | 496 (48–50/task) |
| RoboMemArena | [robomemarena-50-demos](https://huggingface.co/datasets/Yinpei/robomemarena-50-demos) | 1,282 (T04: 32; others: 50) |

MIKASA's full 90-task experiment and its five-task publication comparison
(`mikasa_robo_vla_suite`) share the MIKASA environment and demonstration package.
The comparison plans select only ShellGameTouch, InterceptMedium and
RememberColor3/5/9. See [both MIKASA scopes](benchmarks/mikasa_robo.md).

For **demo + interaction**, each task gets 50 demonstrations and a total budget
of **50 simulator training attempts** for the entire development process: one
attempt using the reset seed of each assigned training demonstration. Failed
attempts count toward the same budget; seeds are not reused. RoboMemArena's T04
uses its actual 32 demos and 32 attempts. Demo-only settings allow zero simulator
training attempts. Held-out evaluation uses separate seeds after freezing.

```bash
uv sync --locked --inexact --extra data
cam download-demos --benchmark robomme --output public_runs/demos/robomme --videos
```

Use `--tasks TASK_NAME` to download selected task H5s. `--videos` includes existing
preview videos (RoboMME and MIKASA); all packages contain frames in H5. The command
pins a verified release revision and downloads its manifest and `selected_25.json`.
An interrupted download can be resumed with the same command.

For a 25-demo batch plan, the launcher creates standalone H5s containing the exact
selected identities under the experiment's `datasets/` directory before learning.
Only that assignment enters the learner sandbox. The downloaded pool stays intact.
For a manual preparation from a larger pool, run:

```bash
cam subset --dataset "$DATA" --output /path/to/experiment/assigned25 \
  --selection "$DATA/selected_25.json" --count 25 --manifest-only
```

Then pass `assigned25` to `cam prepare`. RoboMME already contains its exact 25/task
assignment. Normal preparation uses the released manifest without replaying demos.
Use the [benchmark interface guides](benchmarks/) for observation formats.

## Development logs

[Yinpei/CaM_develop_logs](https://huggingface.co/datasets/Yinpei/CaM_develop_logs)
contains the recorded agent logs, prompts, code revisions, results and available
videos for RoboMME, RMBench, RoboMemArena and the five-task MIKASA VLA suite.
The four archives total **5.54 GB compressed** (about 10.60 GB of file contents).
H5 data and the separate historical `mikasa_robo/` collection are excluded.

Install `zstd` and the Hugging Face CLI, then run from the repository root:

```bash
hf download Yinpei/CaM_develop_logs --repo-type dataset \
  --local-dir public_runs/releases/development_logs
(cd public_runs/releases/development_logs && sha256sum -c SHA256SUMS)
mkdir -p public_runs/dev_logs
for archive in public_runs/releases/development_logs/archives/*.tar.zst; do
  tar --zstd -xf "$archive" -C public_runs/dev_logs
done
```

Original log and code bytes are preserved. The archive manifests list omitted H5
files and external asset/evaluation links; historical absolute paths remain in
the logs. Download demonstrations separately and use a fresh preparation to
reproduce development. These archives preserve evidence from the recorded runs.

## One task

```bash
cam prepare validate_runs --dataset "$DATA" \
  --config configs/robomme/demos.json --tasks BinFill
# RUN is the leaf path printed by prepare.
cam learn "$RUN" --gpus 0 1
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
cam site --data-root validate_runs
cam serve validate_runs/website --port 8936
```

For a different benchmark, use the matching configuration and task. Configure
and authenticate the coding client before `learn`; it starts paid sessions.
Published solutions need neither demonstrations nor coding credentials:
[run a solution](solutions.md). For concurrent all-task development, use the
batch commands in the [reproduction guide](reproducing.md#concurrent-development).
