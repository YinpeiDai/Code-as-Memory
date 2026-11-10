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
git clone --recurse-submodules https://github.com/YinpeiDai/Code-as-Memory.git
cd Code-as-Memory
uv sync --locked --inexact --extra dev
source .venv/bin/activate
```

For an existing checkout, run `git submodule update --init --recursive` first.
Install system `bubblewrap`, `strace`, and `ffmpeg`.
NVIDIA/Vulkan or EGL support is required by the chosen simulator. Each
[environment README](../envs/README.md) describes assets and interpreter overrides.
`cam doctor --simulator` checks the four dedicated environments without resetting
simulators or contacting a model.

## Demonstration packages

RoboMME and RMBench are published and verified. MIKASA-Robo and RoboMemArena
uploads are still in progress. `cam download-demos` becomes available for each
benchmark after its verified revision is recorded in
`configs/<benchmark>/datasets/release.json`.

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

## One task

```bash
cam prepare validate_runs --dataset "$DATA" \
  --config configs/robomme/demos_claude_code.json --tasks BinFill
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
