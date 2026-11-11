# Getting started

Build and evaluate robot memory as executable Python: learn what to remember
from demonstrations, retain it during an episode, and use it to guide actions.

## Environment installation

Install the shared Python 3.11 harness from the repository root:

```bash
git clone https://github.com/YinpeiDai/Code-as-Memory.git
cd Code-as-Memory
git submodule update --init
uv sync --locked --inexact --extra dev
source .venv/bin/activate
```

Then install the environment for the benchmark you want to run:

| Benchmark | Installation command |
|---|---|
| RoboMME | `uv sync --locked --project envs/robomme --python 3.11` |
| MIKASA-Robo | `bash envs/mikasa_robo/install.sh` |
| RMBench | `bash envs/rmbench/install.sh` |
| RoboMemArena | `uv sync --locked --project envs/robomemarena --python 3.10` |

Install the system packages `bubblewrap`, `strace`, and `ffmpeg`. The chosen simulator requires
NVIDIA/Vulkan or EGL support; [environment guides](../envs/README.md) describe
assets and interpreter overrides. `cam doctor --simulator` checks the four simulator
environments without resetting them or contacting a model.
Only the four top-level simulator submodules are needed.

## How CaM works

Code-as-Memory represents robot memory in executable Python. The learned program
stores reusable task knowledge: what to observe, what to remember and how to act.
Its explicit memory variables retain observations and progress within an episode,
so the robot can use information that is no longer visible.

1. **Develop:** a coding agent studies assigned demonstrations and writes a memory
   program, including its memory fields, update rules and actions. It checks the
   program offline; demo + interaction also permits counted training rollouts
   using the same training seeds.
2. **Freeze:** the harness saves the final memory program, its declared fields and
   exact source/configuration provenance.
3. **Evaluate:** the program combines new observations with remembered information,
   updates its memory and chooses robot actions. Episode memory resets for each
   new episode; the learned code stays fixed. Evaluation makes **zero model/API calls**.

The trusted harness owns resets, budgets and scoring.
See [interface adaptations](interface.md) for observations and action conventions.

## Learning settings

**Demo only:** develop memory fields, update rules and actions from the assigned
demonstrations using offline analysis and checks. Simulator
training attempts are **zero**.

**Demo + interaction:** inspect the demonstrations and build an initial memory
program, then use **50 total simulator attempts per task** to test and refine its
memory updates and actions.
Each attempt uses the reset seed of one of the **50 assigned training demos,
exactly once**. There are no new training seeds or retries. Failed attempts count;
code revisions do not reset the budget. RoboMemArena T04 uses its actual **32 demos
and 32 attempts**. Held-out seeds are separate.

| Published setting | Demo-only assignments | Demo + interaction |
|---|---|---|
| RoboMME | 25/task | — |
| MIKASA-Robo five-task VLA suite | 25, 50 or up to 250/task | 50 demos + 50 attempts/task |
| RMBench | 25 or up to 50/task | — |
| RoboMemArena | 25 or up to 50/task | 50 demos + 50 attempts/task; T04: 32 + 32 |

RMBench's larger pool has 49 demos for `battery_try` and `blocks_ranking_try`,
48 for `observe_and_pickup`, and 50 for the other tasks.
MIKASA's primary comparison uses ShellGameTouch, InterceptMedium and
RememberColor3/5/9, following prior publications. Full-90 solutions remain available
separately; both scopes share the simulator and public interface. The additional
`mikasa_robo_vla_suite_250` experiment uses up to 250 successful RGB-D demos per
task with zero interaction; see [exact counts and results](results/mikasa_robo.md#five-task-publication-comparison).

## Offline demonstrations

We prepare RGB-D observation and action trajectories from each benchmark's official
training data, replaying trajectories where needed and retaining successful ones.

| Benchmark | RGB-D preparation | Prepared package |
|---|---|---|
| RoboMME | Retain the [released RGB-D recordings](https://huggingface.co/datasets/Yinpei/robomme_data_h5). | [robomme-25-demos](https://huggingface.co/datasets/Yinpei/robomme-25-demos) |
| MIKASA-Robo | Replay [official trajectories](https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-rlds) into **256×256 RGB-D**. | [50 demos/task](https://huggingface.co/datasets/Yinpei/mikasa_robo-50-demos) · [up to 250, five tasks](https://huggingface.co/datasets/Yinpei/mikasa_robo-vla-suite-250-demos) |
| RMBench | Regenerate [official expert trajectories](https://huggingface.co/datasets/TianxingChen/RMBench) at the released training seeds, capturing **320×240 RGB-D**. | [rmbench-50-demos](https://huggingface.co/datasets/Yinpei/rmbench-50-demos) |
| RoboMemArena | Replay [expert trajectories](https://huggingface.co/datasets/RoboMemArenaBenchmark/RoboMemArena) into **256×256 RGB-D**. | [robomemarena-50-demos](https://huggingface.co/datasets/Yinpei/robomemarena-50-demos) |

MIKASA-Robo and RoboMemArena replays use the same public observation interface
as evaluation.

## Run published solutions

We have published the robot memory programs developed during our experiments.
You can evaluate them directly on each benchmark to see their performance.
Code lives in `solutions/<benchmark>/<setting>/`; schemas, checksums and provenance
live in `configs/solutions/`. Preparation verifies compatibility and freezes the
unchanged imported policy as checkpoint R000.

For one task:

```bash
cam prepare runs --solution solutions/rmbench/demo_only_50 --tasks swap_blocks
# RUN is the task run printed by prepare.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
cam report "$RUN"
```

For all tasks and held-out episodes in a setting:

```bash
python scripts/evaluate_solutions.py solutions/mikasa_robo_vla_suite/demo_interaction_50 \
  --output runs/mikasa_vla_eval --gpus 0 1 --workers-per-gpu 5
```

Choose another [published setting](../solutions/README.md) by replacing the solution
path. Use `--resume BATCH` to continue an interrupted batch. New evaluations are
reported separately from historical scores, with scored failures retained.

### View evaluation results

Each task run saves aggregate metrics in `reports/summary.json`, per-episode rows
in `reports/episodes.csv`, and original episode results under `tasks/<task>/rollouts/test/`.
Batch `plan.json` lists the task run paths; `status.json` tracks progress.
To browse results and available rollout videos:

```bash
cam site --data-root runs
cam serve runs/website --port 8936
```

Open `http://localhost:8936`. See [published results](results.md) for the reference scores.

### Sanity checks

For a five-episode execution check of every policy, with logs/results only:

```bash
cam check-solutions --output runs/eval_sanity_check \
  --episodes 5 --gpus 0 1 --workers-per-gpu 10
```

Filter with `--benchmarks mikasa_robo_vla_suite` or `--tasks TASK`.
Use `--resume BATCH` to continue; add `--retry-infrastructure` to archive and retry
unscorable crashes with identical inputs. Scored failures are never replaced.
A passing check means error-free execution; task success is reported separately.
See [recovery](reproducing.md#artifacts-and-recovery) for other evaluation runs.

## Develop your own solution

After downloading our demonstrations, you can reproduce our experiments by
developing your own code solutions. Configure and authenticate your coding client
to develop a new robot memory program.

### Download demonstrations

For example, download the RoboMME package:

```bash
uv sync --locked --inexact --extra data
DATA="$PWD/public_runs/demos/robomme"
cam download-demos --benchmark robomme --output "$DATA" --videos
```

The command verifies the pinned release in `configs/<benchmark>/datasets/release.json`.
`DATA` contains `manifest.json`, task H5s under `h5/`, and `selected_25.json`.
Use `--tasks TASK` for selected tasks; `--videos` includes available preview videos.
The MIKASA 50-demo package supplies both its five-task comparison and full benchmark.
For the up-to-250 setting, download the separate
[five-task package](https://huggingface.co/datasets/Yinpei/mikasa_robo-vla-suite-250-demos)
to `public_runs/demos/mikasa_robo_vla_suite_250_demo` using the
[MIKASA download command](benchmarks/mikasa_robo.md#up-to-250-demos-only).

A learner assigned 25 demos receives only those 25. Batch plans materialize the
selected subset automatically; for manual preparation from a larger pool, use:

```bash
cam subset --dataset "$DATA" --output runs/assigned25 \
  --selection "$DATA/selected_25.json" --count 25 --manifest-only
```

Pass `runs/assigned25` as the dataset afterward. RoboMME already contains its
exact 25/task assignment and needs no additional subset.

### Start development

Choose a mode-named configuration under `configs/<benchmark>/` and set its `coding`
object to your client, model and account settings. Authenticate that client before
starting development. [Experiment plans and batch commands](reproducing.md#concurrent-development)
handle concurrent development and evaluation after each policy freezes.

For RoboMME, use the downloaded package:

```bash
cam prepare runs --dataset "$DATA" \
  --config configs/robomme/demos.json --tasks BinFill
# RUN is the leaf run path printed by prepare.
cam learn "$RUN" --gpus 0 1
```

For demo + interaction, use the matching benchmark's `both.json` and demo package.
See [benchmark guides](benchmarks/README.md) for task names and complete commands.

### Development logs

[Yinpei/CaM_develop_logs](https://huggingface.co/datasets/Yinpei/CaM_develop_logs)
records how the robot memory programs were developed: agent logs, prompts, code
revisions, results and available videos for RoboMME, RMBench, RoboMemArena and
the five-task MIKASA VLA suite.

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

The archives preserve recorded log and code bytes. Demonstration data and external
simulator assets are excluded; download demos separately for fresh development.
The additional [up-to-250 MIKASA archive](https://huggingface.co/datasets/Yinpei/CaM_develop_logs/blob/main/archives/mikasa_robo_vla_suite_demo_only_250.tar.zst)
extracts to `public_runs/dev_logs/mikasa_robo_vla_suite/demo_only_250` and retains
code history, intermediate analysis images and generated held-out rollouts, including
observation H5s. The original four archives omit all H5s; manifests record each
archive's scope.
