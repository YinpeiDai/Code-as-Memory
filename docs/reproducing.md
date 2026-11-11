# Reproducing experiments

Start with [installation, learning settings and data](getting_started.md).
To evaluate released policies without development, see
[running published solutions](getting_started.md#run-published-solutions).
New runs use the current harness and remain separate from published scores.

## Demonstration and interaction budgets

Demo-only development permits **zero simulator training attempts**. Demo +
interaction assigns **50 demonstrations and 50 total attempts per task**:
each attempt uses one assigned demo's reset seed, exactly once across all code
revisions. Failed attempts count; there are no new seeds or retries.
RoboMemArena T04 uses its actual **32 demos and 32 attempts**. Held-out identities
are separate and used only after freezing, with no model calls.

The larger RMBench pool has 49 demos for `battery_try` and `blocks_ranking_try`,
48 for `observe_and_pickup`, and 50 for the other seven tasks. For 25-demo plans,
the launcher materializes the exact `selected_25.json` assignment; the learner
never receives the larger pool. Manual preparations must first use
[`cam subset`](getting_started.md#demonstration-packages).
Normal preparation reads released manifests without replaying demonstrations.

## Concurrent development

Authenticate your coding client and set the configuration's `coding` object.
The supplied plans select Claude Code / Opus 5.5 and start from blank policies.

| Benchmark | Plan under `configs/` | Published settings |
|---|---|---|
| RoboMME | `robomme/experiments/demo_only_25.json` | 25 demos |
| MIKASA-Robo, five-task VLA suite | `mikasa_robo_vla_suite/experiments/demo_interaction_50.json` | 25 demos, 50 demos, 50 demos + interaction |
| RMBench | `rmbench/experiments/demo_only_50.json` | 25 or up to 50 demos |
| RoboMemArena | `robomemarena/experiments/demo_interaction_50.json` | 25 demos, up to 50 demos, demos + interaction |

For example, develop and evaluate the five-task MIKASA interaction setting:

```bash
DATA="$PWD/public_runs/demos/mikasa_robo"
cam download-demos --benchmark mikasa_robo --output "$DATA" --videos
python scripts/develop_batch.py start \
  configs/mikasa_robo_vla_suite/experiments/demo_interaction_50.json \
  --dataset "$DATA" --output runs/mikasa_vla_interaction --workers 8 --gpus 0 1 \
  --evaluate-on-freeze --evaluation-workers-per-gpu 5
```

`--tasks TASK` selects individual tasks. Omit `--evaluate-on-freeze` to develop
only. Otherwise, each frozen policy enters full held-out evaluation immediately,
while development continues. Eight coding workers share training GPU leases;
evaluation uses five episode workers per GPU. Each task receives a timestamped
run with pinned source, configuration, assigned data and the exact composed prompt.
See the [benchmark guides](benchmarks/README.md) for task names and other plans.

## MIKASA five-task publication comparison

The `mikasa_robo_vla_suite` subset follows prior publications: ShellGameTouch,
InterceptMedium and RememberColor3/5/9 (`-VLA-v0` IDs). It shares MIKASA's simulator
and demonstration package. Each of its three settings develops independent
policies and evaluates 50 held-out episodes per task: **750 episodes total**.

Run all three settings with shared worker and GPU limits:

```bash
python scripts/develop_suite.py start configs/mikasa_robo_vla_suite/suite.json \
  --dataset "$DATA" --output runs --gpus 0 1
# CAMPAIGN is the directory printed by start.
python scripts/develop_suite.py status "$CAMPAIGN"
```

The suite shares eight coding workers (3/3/2) and evaluates each policy after
freeze. ShellGameTouch uses GPU physics/IK; other MIKASA tasks use CPU by default,
in both training and evaluation. The backend is selected by the interface.

The full 90-task plan remains at
`configs/mikasa_robo/experiments/demo_interaction_50.json`. Keep its results
separate from the five-task comparison.
[Plans](../configs/mikasa_robo_vla_suite/README.md) ·
[MIKASA results](results/mikasa_robo.md).

## Prompts and isolation

Prompts combine the shared core and learning mode with the benchmark's public
interface. MIKASA and RoboMemArena add demo-first interaction guidance; RMBench
uses the shared demo-only instructions. See [prompt composition](../src/cam/prompts/README.md).
Archived runs retain their original prompts; relevant differences are documented
on the [benchmark result pages](results.md).

The learner's six tools are `command`, `read_image`, `usage`, `snapshot`,
`validate` and `rollout`. Its sandbox exposes `/workspace`, assigned
`/demonstrations`, counted training `/feedback` and public `/runtime` libraries.
Simulator implementations, hidden state, held-out results, prior solutions,
host credentials and network access are unavailable.

Evaluation executes only frozen code, with fresh episode memory and no demo
mount. A harness, prompt, configuration or data change requires a fresh preparation;
never edit a prepared snapshot. Hosts may fix infrastructure and public interfaces
without giving learners task-specific fixes or held-out diagnostics.

## Artifacts and recovery

Each task run contains `run.json`, `README.md`, `runner.py`, pinned `inputs/`,
and `tasks/<task>/` with its prompt, revisions, checkpoints, transcripts and
rollouts. Batch output keeps `batches/` and `dev_logs/` under the selected root.

- Resume unfinished learning with `cam learn RUN --resume`.
- Execute a pinned run with its generated `runner.py`.
- For an unscorable infrastructure failure, use
  `cam recover-test RUN TASK EPISODE --budget N --reason TEXT` to archive the
  attempt before an identical retry. Preserve scored failures and report
  scored/expected coverage.
- Set `recording: "results"` in a new configuration, or pass `--recording results`
  when importing a solution, to retain logs/JSON without videos and observation
  arrays. Scoring and policy inputs remain unchanged.

The [five-episode sanity check](getting_started.md#sanity-checks) uses results-only
recording and ten workers per GPU. It checks execution rather than full benchmark
coverage. For progress and result galleries, see the [website guide](website.md).
