# Reproducing experiments

Use [installation and data](getting_started.md) to set up the repository and the
[benchmark guides](benchmarks/) for exact settings. New development uses the
current prompt and interface. Published scores refer to their original frozen
runs; rerunning a policy creates separate evidence.

## Demonstration and interaction budgets

In the **50-demo + interaction** setting, each task receives 50 training
demonstrations and **at most 50 simulator training attempts in total**, shared
across all code revisions during development. Attempt starts use the reset seeds
of those same 50 demonstrations, once per seed. Starting an attempt consumes its
budget even if it fails; a code revision does not reset the budget. RoboMemArena's
T04 uses 32 demonstrations and 32 attempts. Demo-only settings permit zero
simulator training attempts. Held-out evaluation uses separate seeds only after
the policy freezes and makes no model calls.

## Prepare, develop, freeze, evaluate

1. Install the common harness and the benchmark’s environment under `envs/`.
2. Download the prepared demonstration package. `--dataset` points to its
   directory containing `manifest.json` and `h5/`. The batch launcher materializes the exact `selected_25.json` assignment for
   25-demo plans. Manual preparations require `cam subset` first when the pool
   is larger than the assignment; see [data setup](getting_started.md).
3. Select a configuration and run `cam prepare OUTPUT --dataset DATA --config
   CONFIG --tasks TASK`, or use the batch commands below. Each task gets a
   new timestamped run, a blank initial policy, a pinned source/configuration
   snapshot, its composed prompt, and a generated README.
4. `cam learn RUN` develops and freezes the policy. Demo-only mode permits zero
   simulator training attempts. Interaction consumes the pinned identity schedule.
5. Run `cam evaluate RUN --gpus 0 1 --workers-per-gpu 5` after freeze. Evaluation
   makes no model calls. Retain all scored outcomes and report scored/expected
   coverage; infrastructure failures are separate.

## Concurrent development

Authenticate Claude Code or Codex separately. The supplied plans use Claude Code
with Opus 5.5; configure the referenced client settings for your account before
starting paid development sessions.

| Benchmark | Development plan | Best published solution |
|---|---|---|
| RoboMME | `configs/robomme/experiments/demo_only_25.json` | `solutions/robomme/demo_only_25` |
| RMBench | `configs/rmbench/experiments/demo_only_50.json` | `solutions/rmbench/demo_only_50` |
| MIKASA-Robo | `configs/mikasa_robo/experiments/demo_interaction_50.json` | `solutions/mikasa_robo/demo_interaction_50` |
| RoboMemArena | `configs/robomemarena/experiments/demo_interaction_50.json` | `solutions/robomemarena/demo_interaction_50` |

The MIKASA row covers all 90 tasks. We also run the five task identities used
in prior VLA publications, under the `mikasa_robo_vla_suite` configuration and
solution namespace. Both scopes use the same MIKASA environment, public interface
and demonstration package. See [their purpose and results](results/mikasa_robo.md).

```bash
DATA="$PWD/public_runs/demos/mikasa_robo"
LOGS="$PWD/runs/mikasa_reproduction"
uv sync --locked --inexact --extra data
cam download-demos --benchmark mikasa_robo --output "$DATA" --videos

python scripts/develop_batch.py start \
  configs/mikasa_robo/experiments/demo_interaction_50.json \
  --dataset "$DATA" --output "$LOGS" --workers 8 --gpus 0 1 \
  --evaluate-on-freeze --evaluation-workers-per-gpu 5
```

Omit `--evaluate-on-freeze` to develop without automatic held-out evaluation.
Videos are optional. Add `--tasks TASK_NAME` to select tasks. For a 25-demo plan,
preparation extracts the exact `selected_25.json` identities locally; the full
downloaded pool stays outside the learner's sandbox.

The batch launcher defaults to eight coding jobs. `--dataset`, `--output`, and
`--tasks` override a single-benchmark plan without editing it. `--evaluate-on-freeze`
starts a separate host evaluation queue, with five workers per GPU by default.
Training simulators have a shared GPU lease; coding concurrency is independent.
Across GPUs 0 and 1, normal evaluation admits up to ten episode workers total.
See [running solutions](solutions.md) for evaluation without development and the
separate five-episode sanity check, which uses ten workers per GPU.

## MIKASA five-task publication comparison

The subset is ShellGameTouch, InterceptMedium and RememberColor3/5/9 (standard
`-VLA-v0` IDs), following the task selection in prior publications. Its three
settings develop fresh policies independently; they are not imports from the
90-task solutions. All have 50 held-out episodes per task, for 750 total.

| Setting | Plan under `configs/mikasa_robo_vla_suite/` | Assigned demos/task | Training attempts/task |
|---|---|---:|---:|
| 25 demos only | `experiments/demo_only_25.json` | 25 | 0 |
| 50 demos only | `experiments/demo_only_50.json` | 50 | 0 |
| 50 demos + interaction | `experiments/demo_interaction_50.json` | 50 | 50 total, one per demo seed |

Use the existing MIKASA demo package (`DATA` above). Start all three settings
with shared GPU admission and eight total coding workers:

```bash
python scripts/develop_suite.py start configs/mikasa_robo_vla_suite/suite.json \
  --dataset "$DATA" --output runs --gpus 0 1
# CAMPAIGN is the directory printed by start.
python scripts/develop_suite.py status "$CAMPAIGN"
```

The controller assigns 3/3/2 coding workers, materializes only the selected 25
demonstrations for that arm, and queues full evaluation immediately when each
policy freezes. Evaluation uses five workers per GPU and makes no model calls.
Every interaction arm uses each assigned demo identity once. ShellGameTouch uses
GPU physics/IK; the other four use CPU physics/IK, in both training and evaluation.

For one setting, use its plan with the batch launcher:

```bash
python scripts/develop_batch.py start configs/mikasa_robo_vla_suite/experiments/demo_only_25.json \
  --dataset "$DATA" --output runs/mikasa_vla_25 --workers 8 --gpus 0 1 \
  --evaluate-on-freeze --evaluation-workers-per-gpu 5
```

Use the suite controller when running settings concurrently so they share the
eight-worker limit and GPU leases. Completed campaign arms export exact frozen
code and catalogs inside the campaign; the published completed experiment is
already in `solutions/mikasa_robo_vla_suite/<setting>/` and
`configs/solutions/mikasa_robo_vla_suite/<setting>.json`.

To evaluate those published solutions without development or a demo download:

```bash
python scripts/evaluate_solutions.py solutions/mikasa_robo_vla_suite/demo_interaction_50 \
  --output runs/mikasa_robo_vla_suite_eval --gpus 0 1 --workers-per-gpu 5
```

Substitute `demo_only_25` or `demo_only_50` for the other settings. Imports freeze
at R000 while retaining the original development budget and provenance in the
catalog. New evaluations remain separate from published results and full-90 scores.
[Plan details](../configs/mikasa_robo_vla_suite/README.md).

## Released settings

| Benchmark | Demonstrations per task | Training interaction |
|---|---:|---|
| RoboMME | 25 | None |
| MIKASA-Robo, original full 90 tasks | 50 | 50 attempts, each assigned demo identity once; CPU physics |
| MIKASA-Robo, five publication tasks | 25 or 50 | None, or 50 attempts using each assigned identity once |
| RMBench | 25 or up to 50 | None |
| RoboMemArena | 25 or up to 50 | None, or one attempt per assigned demo identity |

New MIKASA preparations use GPU physics for `ShellGameTouch-VLA-v0`, overriding
the configured CPU default. Other tasks default to CPU and permit an explicit
GPU comparison override. The resolved backend is recorded per task in `run.json`;
historical snapshots and scores retain their original backend.

RMBench’s larger pool has 49 demos for `battery_try`, 49 for
`blocks_ranking_try`, 48 for `observe_and_pickup`, and 50 for the other seven
(496 total). RoboMemArena has 32 for T04 and 50 for the other 25 (1,282 total).
Its interaction arm uses exactly 32/50 attempts, respectively. Never pad counts,
repeat consumed identities, or borrow held-out identities. Use every assigned
interaction attempt, including verification after apparent success.

Demonstration manifests carry identities and file hashes. Normal preparation
reads the published manifest and checks assignments without replay. When a
25-demo assignment must be materialized, subset preparation copies just those
H5 groups and validates the new file; it does not re-audit the larger pool. `cam validate-data` is an optional full file audit.
The historical RoboMME package is read through an explicit compatibility layer:
its train/test metadata must match, its H5 bytes remain unchanged, and the current
simulator hash is recorded as a compatibility target, not historical provenance.

## Prompts and isolation

The [prompt directory](../src/cam/prompts/README.md) describes shared instructions
and each benchmark’s additions. Existing snapshots keep their exact original
prompts. The shared demo-only prompt now requires final offline execution checks;
some published runs predate that revision. See [results](results.md).

Prompts combine a shared core and learning mode with the benchmark's public
interface. MIKASA and RoboMemArena add their own demo-first interaction guidance.
RMBench demo-only uses the shared prompt and public interface without an extra
motion supplement. Each run saves the exact composed prompt.

Learners have six tools: `command`, `read_image`, `usage`, `snapshot`, `validate`,
and `rollout`. Their sandbox exposes `/workspace`, assigned `/demonstrations`,
counted training `/feedback`, and public `/runtime` libraries. Simulator code,
held-out metadata/results, host files, credentials, network access, other runs,
and historical solutions are unavailable. Evaluation executes frozen code with
fresh episode memory and no demonstration mount.

Hosts may repair infrastructure and public interface documentation. Do not pass
hidden state, test feedback, task-specific fixes, or historical policies to a
learner. Published-policy evaluation is an explicit evaluation-only import and
has no learner. Any harness, prompt, configuration, or data change requires a
fresh preparation. Never edit a pinned snapshot to bypass integrity checks.

## Artifacts and recovery

A task run is `OUTPUT/<benchmark>/<task>/<client-model>/<setting>/<timestamp>/`.
Batch output keeps `batches/` and `dev_logs/` beneath the requested output root.
Each run contains `run.json`, `README.md`, `runner.py`, pinned `inputs/`, and
`tasks/<task>/` with the prompt, workspace, all revisions, immutable checkpoints,
agent transcripts, training rollouts, and evaluation results.

Use the run’s `runner.py` for historical execution. `cam learn RUN --resume`
resumes unchanged unfinished learning. For an unscorable infrastructure failure,
`cam recover-test RUN TASK EPISODE --budget N --reason TEXT` archives the failed
attempt before an identical retry. Do not retry scored failures to change scores.

`recording: "results"` in a new configuration omits videos, observation H5s,
array dumps, and syscall traces while retaining action/decision logs and result
JSON. Policy observations, actions, scoring, limits, and isolation are unchanged.
Published preparations also accept `--recording results`. The five-episode
`cam check-solutions` command uses this mode; it does not claim full benchmark
coverage. Its current exception is 10 episode workers per GPU.

Generate the directory’s `dashboard/PROGRESS.md`, `index.json`, `results.json`,
and `results.csv` with `cam site --data-root OUTPUT`. The server refreshes them
as development and evaluation progress. See [website](website.md).
