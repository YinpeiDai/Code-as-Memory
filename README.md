# CaM · Robot Memory

Build robot memory from demonstrations and interaction. A coding agent develops a
Python program that remembers what it observes, makes decisions, and controls the
robot. The frozen program runs with **no model calls during evaluation**.

Four benchmarks, one development harness, and a browsable record of every experiment.

| Benchmark | Setting | Successful / evaluated | Success rate |
|---|---|---:|---:|
| RoboMME | 25 demonstrations | 795 / 800 | **99.38%** |
| MIKASA-Robo | 50 demonstrations + 50 interactions | 4,444 / 4,500 | **98.76%** |
| RMBench | 25 demonstrations | 927 / 1,000 | **92.70%** |
| RMBench | Up to 50 demonstrations | 1,000 / 1,000 | **100.00%** |
| RoboMemArena | 25 demonstrations | 1,070 / 1,300 | **82.31%** |
| RoboMemArena | Up to 50 demonstrations | 1,179 / 1,300 | **90.69%** |
| RoboMemArena | Up to 50 demonstrations + matching interactions | 1,271 / 1,300 | **97.77%** |

These are the published experiments, with complete scored coverage. Prompt and
interface revisions differ between some settings. MIKASA includes 80 imported
tasks with result records only. [Results and provenance](docs/results.md) explain
coverage, actual demonstration counts, and comparison limits.

## Start with a published solution

Use Linux, Python 3.11, an NVIDIA GPU, and `uv`. Install `bubblewrap`, `strace`, and
`ffmpeg` through your system package manager.

```bash
git clone --recurse-submodules https://github.com/YinpeiDai/Code-as-Memory.git
cd Code-as-Memory
uv sync --locked --inexact --extra dev
source .venv/bin/activate

# Install only the benchmark you want to run.
uv sync --locked --project envs/robomme --python 3.11
cam doctor --benchmark robomme

cam prepare validate_runs --solution solutions/robomme/demo_only_25 --tasks BinFill
# Set RUN to the task run path printed by prepare.
cam evaluate "$RUN" --budgets 0 --gpus 0 1 --workers-per-gpu 5
```

To evaluate **all 16 RoboMME tasks** (all 800 held-out episodes):

```bash
python scripts/evaluate_solutions.py solutions/robomme/demo_only_25 \
  --output validate_runs/robomme_full --gpus 0 1 --workers-per-gpu 5
```

Use any setting directory from the table below with the same command. It evaluates
one task at a time with five episode workers on each GPU, keeping the limit shared
across the whole batch. `--resume BATCH` continues its unchanged preparations.

Published solutions need simulator assets, but no demonstrations, coding client,
or model credentials. Each setting has its own README with results, prompt
provenance, and a runnable command.

| Benchmark | Environment | Published solutions | Development plan |
|---|---|---|---|
| [RoboMME](docs/benchmarks/robomme/README.md) | [envs/robomme](envs/robomme/README.md) | [demo_only_25](solutions/robomme/demo_only_25/README.md) | `configs/robomme/experiments/demo_only_25.json` |
| [MIKASA-Robo](docs/benchmarks/mikasa_robo/README.md) | [envs/mikasa_robo](envs/mikasa_robo/README.md) | [demo_interaction_50](solutions/mikasa_robo/demo_interaction_50/README.md) | `configs/mikasa_robo/experiments/demo_interaction_50.json` |
| [RMBench](docs/benchmarks/rmbench/README.md) | [envs/rmbench](envs/rmbench/README.md) | [25 demos](solutions/rmbench/demo_only_25/README.md) · [50 demos](solutions/rmbench/demo_only_50/README.md) | `configs/rmbench/experiments/demo_only_50.json` |
| [RoboMemArena](docs/benchmarks/robomemarena/README.md) | [envs/robomemarena](envs/robomemarena/README.md) | [25 demos](solutions/robomemarena/demo_only_25/README.md) · [50 demos](solutions/robomemarena/demo_only_50/README.md) · [interaction](solutions/robomemarena/demo_interaction_50/README.md) | `configs/robomemarena/experiments/demo_interaction_50.json` |

## Reproduce development

Download a prepared demonstration pool, choose a benchmark plan from the table,
and supply the data directory and the directory for your logs.
Authenticate Claude Code or Codex separately; the supplied plans use Claude Code
with Opus 5.5. Edit the referenced client configuration for your account before
launching paid development sessions.

```bash
uv sync --locked --inexact --extra data
DATA="$PWD/public_runs/demos/mikasa_robo"
cam download-demos --benchmark mikasa_robo --output "$DATA" --videos
LOGS="$PWD/validate_runs/mikasa_reproduction"

# Eight concurrent coding jobs; all run records stay under LOGS.
python scripts/develop_batch.py start \
  configs/mikasa_robo/experiments/demo_interaction_50.json \
  --dataset "$DATA" --output "$LOGS" --workers 8 --gpus 0 1
```

The [four datasets](docs/getting_started.md#demonstration-packages) contain the
released pools and `selected_25.json`. For a 25-demo plan, preparation extracts
those exact identities locally; download the larger pool only once. Videos are
optional. Add `--tasks TASK_NAME` to develop selected tasks. For automatic evaluation as
policies finish and freeze, use:

```bash
python scripts/develop_batch.py start \
  configs/mikasa_robo/experiments/demo_interaction_50.json \
  --dataset "$DATA" --output "$LOGS" --workers 8 --gpus 0 1 \
  --evaluate-on-freeze --evaluation-workers-per-gpu 5
```

**Defaults: eight development jobs; five evaluation episode workers per GPU.**
On GPUs 0 and 1, evaluation uses up to ten episode workers. Training simulators
use a shared GPU lease independently of the coding-job count.

Prompts share a core and learning-mode instructions, then add each benchmark’s
public interface. MIKASA and RoboMemArena also have their own demo-first
interaction guidance. RMBench demo-only uses the shared prompt and its public
interface without an additional motion supplement. Exact prompts are saved in
every run. See [prompt composition](src/cam/prompts/README.md) and the
[reproduction guide](docs/reproducing.md).

## Inspect experiments

```bash
# A complete archive, or any directory containing your new task runs.
cam site --data-root public_runs
cam serve public_runs/website --port 8935

cam site --data-root validate_runs
cam serve validate_runs/website --port 8936
```

The website provides live development and evaluation progress, consolidated
results, and galleries for demonstrations, development logs, and evaluation.
Gallery cards open a separate page; episode numbers and previous/next navigate
within that page. The selected directory holds its progress files, results,
website metadata, and preview cache. [Website guide](docs/website.md).

To check every published policy on five complete episodes, saving logs and JSON
results without videos or observation recordings:

```bash
cam check-solutions --output validate_runs/eval_sanity_check \
  --episodes 5 --gpus 0 1 --workers-per-gpu 10
```

This dedicated sanity check defaults to 10 workers per GPU and checks available
GPU memory before admission. Its results stay separate from published scores.

## Repository map

```text
src/cam/interface/<benchmark>/  Simulator adapters, public APIs, task metadata
src/cam/harness/                Preparation, isolated development, evaluation
src/cam/prompts/                Shared prompts and benchmark guidance
envs/<benchmark>/              Four independent simulator environments
configs/<benchmark>/           Client settings and reproducible experiment plans
configs/solutions/<benchmark>/  Policy schemas, checksums, provenance
solutions/<benchmark>/         Final Python programs and setting READMEs
docs/                          Setup, reproduction, results, website, architecture
public_runs/ · validate_runs/   Downloaded or generated experiment artifacts
```

[Installation and data](docs/getting_started.md) ·
[Reproduction](docs/reproducing.md) · [Architecture](docs/architecture.md) ·
[Running solutions](docs/solutions.md)

Apache-2.0. Benchmark code and assets retain their upstream licenses; see [NOTICE](NOTICE).
