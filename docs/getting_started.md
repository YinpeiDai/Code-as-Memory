# First run on RoboMME

Start with one task, 25 RGB-D demonstrations, and demonstration-only learning.
Read the [reproduction rules](reproducing.md) before launching an experiment.

## 1. Install

Use Linux, Python 3.11, an NVIDIA GPU with Vulkan, and the `uv`, `bwrap`
(Bubblewrap), `strace`, `ffmpeg`, and `xz` executables. From the checkout:

```bash
git submodule update --init --recursive
uv sync --locked --extra dev --extra robomme
source .venv/bin/activate
cam doctor --simulator
```

Follow the pinned benchmark's [installation guide](../third_party/robomme_benchmark/readme.md)
for assets and drivers. `doctor` checks dependencies and isolation without model
calls or simulator resets; it does not verify provider access or GPU rendering.
Install and authenticate Claude Code or Codex separately.

## 2. Supply data and configure the client

With the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
download and unpack one task from the
[regular training dataset](https://huggingface.co/datasets/Yinpei/robomme_data_h5):

```bash
mkdir -p data/raw_archives data/raw_robomme
hf download Yinpei/robomme_data_h5 record_dataset_ButtonUnmaskSwap.h5.tar.xz \
  --repo-type dataset --local-dir data/raw_archives
tar -xJf data/raw_archives/record_dataset_ButtonUnmaskSwap.h5.tar.xz -C data/raw_robomme
mkdir -p local
cp configs/robomme/demos_claude_code.json local/first_run.json
```

Raw recordings can occupy tens of GB per task. Point `--raw-data` at the directory
containing unpacked `record_dataset_<Task>.h5` files in modern
`episode_N/timestep_N` format. Legacy recordings need an audited conversion;
see [demonstration validation](reproducing.md#demonstration-validation).

Edit `local/first_run.json` for your installation. Keep `mode: "demos"`,
`budget: 0`, and `depth: true` for this example.

| Client/provider | Template in `configs/robomme/` | Configure |
|---|---|---|
| Claude Code / Bedrock | `demos_claude_code.json` | `coding.model`, `aws_profile`, `aws_region`; shipped values are `us.anthropic.claude-opus-5-5`, `sagemaker`, `us-east-1` |
| Claude Code / Anthropic | `demos_claude_code.json` | Set `coding.provider` to `anthropic`, choose an available model, authenticate the client |
| Codex / OpenAI | `demos_codex.json` | Choose a model in the client's catalog; shipped example is `gpt-6.1-sol` |

Keep credentials in the client's authentication setup, outside experiment files.

## 3. Extract and prepare

```bash
cam extract --benchmark robomme --raw-data data/raw_robomme \
  --output data/first_25_rgbd --tasks ButtonUnmaskSwap --count 25 --workers 1
cam prepare runs --dataset data/first_25_rgbd \
  --config local/first_run.json --tasks ButtonUnmaskSwap --timetag first_run
```

Extraction selects **9 easy, 8 medium, 8 hard** training demonstrations. Selection
and hashes are in `data/first_25_rgbd/manifest.json`; previews are in `rollouts/`.
To resume interrupted extraction, repeat the same command with `--resume`.

Use the run path printed by `prepare`. With the unmodified Claude template:

```bash
run_dir=runs/robomme/ButtonUnmaskSwap/cc-opus5.5/demo_only/first_run
```

Preparation creates a blank policy, the composed prompt, public API reference,
and a run README. Source, configuration, and input hashes are pinned; changed
inputs require a new timetag. No manual prompt assembly is needed.

## 4. Develop and evaluate

```bash
cam learn "$run_dir" --gpus 0 --workers-per-gpu 1
cam evaluate "$run_dir" --budgets 0 --gpus 0 --workers-per-gpu 4
cam status "$run_dir"
cam report "$run_dir"
```

`learn` incurs model costs and allows zero simulator attempts in this setting.
After learning freezes the policy, the separate `evaluate` command runs all
**50 held-out episodes**, with zero model calls. The coding agent cannot launch
evaluation. Use `status` from another terminal to monitor progress.
See [run layout](run_layout.md) for logs, code revisions, checkpoints, and recovery.

## 5. View development and results

```bash
cam site --runs "$run_dir" --datasets data/first_25_rgbd --output sites/first_run
cam serve sites/first_run --host 127.0.0.1 --port 8910
```

Open <http://127.0.0.1:8910/#development>. Camera previews and colored attempt links
open separate replay tabs. Each page shows the video with synchronized action
stages, policy labels, frame stepping, and robot commands. **Executed code** and
**Agent log** open the matching checkpoint and chronological transcript in new
tabs. Automatic updates preserve playback. Held-out results have their own
navigation tab. On a remote machine, forward port 8910.
**Reference demos** shows the examples assigned to the selected learner, with
their own video pages. Missing previews are generated on first viewing by `cam serve`.
The frontend is `src/cam/web/`; the builder/server is `src/cam/reporting/site.py`.

## More tasks or settings

Download and unpack all 16 task archives, then omit `--tasks` from `extract`
and `prepare` to process the full benchmark. `prepare` creates a separate leaf
per task; run learning/evaluation for each printed leaf. All 16 regular tasks
give **800 test episodes per setting and evaluation budget**.

| Setting | Config for either `claude_code` or `codex` | Evaluation `--budgets` |
|---|---|---|
| `demo_only` | `demos_<client>.json` | `0` |
| `interaction_only` | `interaction_<client>.json` | `1 5 10 20` |
| `demo_interaction_v1` | `both_v1_<client>.json` | `1 5 10 20 25` |
| `demo_interaction_v2` | `both_v2_<client>.json` | `1 5 10 20 25` |

V1 allows free choice between demos and interaction; v2 instructs the agent to
work from demos first and use interaction to verify and explore uncertainties.
Use v2 for current demo-plus-interaction experiments. Both versions use the
same tools and limits. See [version selection and examples](demo_interaction.md).

Demo-plus-interaction templates allow at most 25 attempts; v2 uses the remaining
distinct training seeds for verification even when the solution seems ready.
V1 and repeating seed schedules permit justified early stopping.
Interaction-only templates allow 20. RoboMME templates cycle three fixed
demonstration seeds. MIKASA-Robo demo-plus-interaction templates use 25 distinct training seeds
outside the assigned demos and tests; the RGB-D v2 comparison uses 50.
R1/R5/R10/R20/R25 are checkpoints from one development run; see the
[protocol](reproducing.md). Each command owns its worker pool, so concurrent
commands add their GPU workloads.

If setup fails, inspect `cam doctor --simulator` and
`RUN/tasks/<Task>/logs/session_NNN/`. Resume only unchanged unfinished runs;
preserve failures using the [recovery commands](run_layout.md#resume-and-recovery).

For the current 50-demo MIKASA-Robo v2 experiment, follow the
[shared-NFS setup guide](mikasa_v2_setup.md).
