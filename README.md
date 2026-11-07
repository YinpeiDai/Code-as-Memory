# Code-as-Memory (CaM)

Learn reusable Python programs from demonstrations, interaction, or both.
A coding agent develops the program; the frozen program performs perception,
decisions, and control with **zero model calls during evaluation**.

RoboMME is the first runnable benchmark. Claude Code and Codex use the same task
prompt, six isolated CaM tools, attempt accounting, and result format.

```text
Raw training H5 → validated demonstrations → isolated coding agent
  → frozen Python policy → held-out evaluation → logs, videos, reports, website
```

## Get started

Use Linux, Python 3.11, Bubblewrap, strace, ffmpeg, and an NVIDIA/Vulkan setup.
Install and authenticate your chosen coding client separately.

```bash
git clone --recurse-submodules https://github.com/YinpeiDai/Code-as-Memory.git
cd Code-as-Memory
uv sync --locked --extra dev --extra robomme
source .venv/bin/activate
cam doctor --simulator
```

Follow the [first-run guide](docs/getting_started.md) for data download, model
configuration, and a complete one-task experiment. The core workflow is:

```bash
cam extract --benchmark robomme --raw-data /path/to/raw_h5 \
  --output data/robomme_25_rgbd --tasks ButtonUnmaskSwap --count 25
cam prepare runs --dataset data/robomme_25_rgbd \
  --config configs/robomme/demos_claude_code.json --tasks ButtonUnmaskSwap

# Set run_dir to the leaf path printed by prepare.
cam learn "$run_dir" --gpus 0 --workers-per-gpu 1
cam evaluate "$run_dir" --budgets 0 --gpus 0 --workers-per-gpu 4
cam report "$run_dir"
cam site --runs "$run_dir" --datasets data/robomme_25_rgbd --output sites/robomme
cam serve sites/robomme --host 127.0.0.1 --port 8910
```

Edit the example configuration for your account before preparing. `learn` starts
paid model sessions. Evaluation is a separate host command after policy freeze;
the coding agent cannot trigger it or access test feedback.

## Experiments and documentation

```text
runs/<benchmark>/<task>/<agent-model>/<setting>/<timetag>/
```

Short names include `cc-opus5.5` (Claude Code) and `cx-gpt6.1-sol` (Codex).
Settings are `demo_only`, `interaction_only`, and `demo_interaction`. Every run
gets a README with its settings, artifact paths, commands, and filesystem access.
Datasets, runs, logs, and generated sites stay outside version control.

| Guide | Contents |
|---|---|
| [First run](docs/getting_started.md) | Install, download, develop, evaluate, view the website |
| [Reproducing experiments](docs/reproducing.md) | Selection, settings, learner access, held-out test rules |
| [Run layout and recovery](docs/run_layout.md) | Logs, every code version, checkpoints, resume and retries |
| [Statistics](docs/statistics.md) | Complete/partial rates, efficiency, development cost |
| [Architecture](docs/architecture.md) | Components, client isolation, adding benchmarks |
| [Validation](docs/validation.md) | Tests, native checks, completed ButtonUnmaskSwap experiment |
| [Historical selection](docs/ROBOMME_25_DEMO_SEEDS.md) | The 25-demo sampling rule and original provenance |
| [Prompt comparison](docs/prompt_comparison.md) | Differences from the earlier CAP protocols |

The historical regular RoboMME result was **795/800 (99.375%)**, from the original
runner. Its [logs and rollouts](https://huggingface.co/datasets/Yinpei/robomme-CaM-log-rollout)
are reference evidence; they do not initialize new policies. The current CaM
runner has completed **50/50 ButtonUnmaskSwap test episodes** with Claude Code /
Opus 5.5; a full 800-episode reproduction remains outstanding.

RoboMME-Plus is a protocol draft; other adapters are planned. Run `cam benchmarks`
for status. Website code is in `src/cam/web/`, with its builder and server in
`src/cam/reporting/site.py`.

Apache-2.0. See [NOTICE](NOTICE) and [source provenance](docs/migration_sources.json).
