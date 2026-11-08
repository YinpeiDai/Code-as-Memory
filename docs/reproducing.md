# Reproducing CaM experiments

These rules apply to people and host agents orchestrating experiments. The learner
receives only its composed prompt and public API reference. For installation and
commands, follow [the first-run guide](getting_started.md).

## Run organization and settings

```text
runs/<benchmark>/<task>/<agent-model>/<setting>/<timetag>/
```

`cam prepare runs --dataset DATA --config CONFIG --tasks TASK` creates one leaf
per task and prints its path. Pass that leaf to `learn`, `evaluate`, `status`,
`report`, and `site --runs`. Each leaf gets a README with its actual settings,
commands, artifacts, and access boundary. Automatic timetags use UTC;
`--timetag` sets one explicitly. Existing leaves are never overwritten.
Agent abbreviations are `cc` (Claude Code) and `cx` (Codex): for example,
`cc-opus5.5` and `cx-gpt6.1-sol`. Standard provider prefixes are omitted and
model versions shortened; variant/date suffixes remain. Unsafe path characters
are percent-encoded. Full client, provider, and model IDs remain in `run.json`
and the run README. A short name never overrides the configured model.

For the MIKASA second-machine experiment, the checkout's `runs/`, `outputs/`,
`data/`, and `sites/` link to matching directories under
`/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/mikasa_robo`. This places development
batches, task workspaces, policies, logs, training/test rollouts, reports,
generated data and dashboard caches on NFS. The existing replay pool remains
at its original shared NFS location. Preparation resolves the links and records
absolute NFS run paths in manifests and generated READMEs; sandbox access stays
limited to the assigned task. See the [storage setup](mikasa_v2_setup.md#experiment-storage-on-nfs).
Configure storage before preparation; preserve historical run paths and pinned
snapshots when setting up another checkout.

| Directory setting | Config `mode` | Demonstrations | Training attempts |
|---|---|---|---|
| `demo_only` | `demos` | Assigned H5 | Zero |
| `interaction_only` | `interaction` | Hidden | Configured budget |
| `demo_interaction_v1` | `both` + `both_protocol: "v1"` | Assigned H5 | Configured budget |
| `demo_interaction_v2` | `both` + `both_protocol: "v2"` | Assigned H5 | Configured budget |

V1 lets the agent freely choose when to use demonstrations or interaction. V2
instructs it to build and check a solution from demonstrations first, then use
interaction to verify the solution and explore uncertain success conditions.
Both use the same six-tool harness; the
difference is in the prompt, not a tool-level gate. New `both` runs require an
explicit version. See [demo + interaction protocols](demo_interaction.md).

Use the requested client, model, and setting. Current demo-plus-interaction
experiments use **v2, at most 25 attempts**. With distinct training seeds, the v2
prompt instructs the agent to use all remaining attempts to verify and strengthen
the solution, even when it seems ready for testing. Successful verification needs
no code change. Repeating seed schedules still permit justified early stopping.
Both versioned templates have this ceiling; interaction-only templates retain 20.
MIKASA-Robo templates use
`training_seed_policy: "disjoint_once"` and `training_seed_count: 25`: every
attempt uses a distinct pinned training seed outside the 25 assigned demos and
held-out tests. Selection is deterministic, without outcome-based filtering.
The budget cannot exceed the number of distinct interaction identities.
RoboMME and historical runs retain `demo_cycle`, which cycles through selected
demonstration identities (three by default). In that policy, a null seed count
uses all extracted identities. Report the policy and actual coverage;
R1/R5/R10/R20/R25 are causal checkpoints from one run, not replicates.

This v2 stopping instruction was revised on 2026-11-07. Earlier prepared runs
retain their early-stopping prompts and results. Use fresh timetags for the
revised prompt and identify the prompt revision when comparing runs.

New MIKASA v2-protocol runs also receive a task-independent
[motion-verification supplement](../src/cam/prompts/benchmarks/mikasa_robo_both_v2.md):
demos first, then counted rollouts to check control response and guide revisions,
with public safety-API limits and rotation/settling checks. This applies to both
MIKASA adapters only in `mode: "both", both_protocol: "v2"`. Shared core/mode
prompts remain unchanged. Its text/source are pinned in `run.json` as
`benchmark_mode_guidance`, included in each learner prompt, and linked from the
run README. Use a fresh timetag and report this prompt revision; do not add
private diagnostic findings, scorer thresholds or tuned task-specific gains.

The original MIKASA-Robo v2 (`mikasa_robo_v2`) interaction plan is an explicit **50-demo +
50-interaction** variant: 256x256 RGB-D, the same ten MIKASA tasks, Claude Code /
Opus 5.5, eight coding jobs, demo-first v2 and `disjoint_once` seeds. Use
`configs/experiments/mikasa_v2_50_newseeds_10tasks.json` for that GPU/new-seed variant;
the current CPU/demo-seed assignment is below.
The launcher requires a final dataset manifest and exactly 50 demos per selected
task; preparation validates data and seed separation before learning. Rendered
depth is derived sensor data. Preserve the earlier 25+25 runs and use fresh timetags.
The v2 live simulator requires `depth: true` for both development and held-out
evaluation: front and wrist RGB are `uint8[256,256,3]`, with aligned
`int16[256,256]` optical-axis depth in millimetres. Calibration accompanies every
frame. The same adapter/broker path handles both splits, without image resizing
or frame subsampling. Historical `mikasa_robo` runs keep their 128x128 RGB interface.

The completed **demo-only** follow-up used the same ten tasks,
50 RGB-D demos each, Claude Code / Opus 5.5 and eight coding jobs. Launch with
`scripts/develop_batch.py start configs/experiments/mikasa_v2_demos50_10tasks.json
--workers 8 --gpus 0 1`. Training-rollout budget is zero. The shared demo-only
prompt asks learners to infer memory representations and candidate completion
conditions from recordings, documenting uncertainty rather than assuming the
exact hidden scorer is identifiable from successful examples.
For another machine, follow the [MIKASA-Robo v2 setup guide](mikasa_v2_setup.md),
including code availability, separate environments and dataset transfer.

The current development uses **50 demos + all 50 training attempts with CPU
physics** on those same ten tasks. Use
`configs/experiments/mikasa_v2_cpu_50_demoseeds_10tasks.json`; its dedicated CPU
config pins the backend for both training and later evaluation, while cameras
still use GPU rendering. The MIKASA supplement permits unlimited rereading of
assigned demos and requires using remaining attempts even when the policy seems
ready. Release/lift distances and cautious motion are hypotheses to investigate
from demos and counted rollouts, with no supplied scorer thresholds. Keep CPU
results separate from prior GPU runs and freeze before held-out evaluation.
The latest seed assignment uses the **same 50 identities/seeds as the demos**,
once each (`demo_cycle`, `training_seed_count: 50`, `budget: 50`). Preparation
verifies exact identity equality and test-seed exclusion. The argument-free
`rollout` API selects the next pinned identity; the ledger rejects a different
identity or attempt 51. No new training seeds are admitted in this follow-up.
The other machine should use
`configs/experiments/mikasa_v2_cpu_50_demoseeds_remaining80.json`, which excludes
those ten tasks and points directly at the completed NFS replay pool:
`/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/data/mikasa_robo_v2_50`.
New runs use the concise MIKASA supplement; the active ten-task batch retains
its earlier pinned wording. See [setup and launch instructions](mikasa_v2_setup.md).
The latest second-machine request uses **ten coding jobs** for the remaining 80
tasks and `--evaluate-on-freeze --evaluation-workers-per-gpu 2 --gpus 0 1`.
The batch host queues full 50-episode testing only after that task's learning
finishes, all 50 training attempts are consumed, and its frozen policy and R050
checkpoint pass integrity checks. One task is evaluated at a time with four
episode workers, two per GPU, alongside at most one training simulator per GPU.
Other tasks may still be learning; none receive held-out feedback. Track actual
scored/expected counts and infrastructure failures separately. Without the
opt-in flag, the development launcher retains its development-only behavior.

## Demonstration validation

Prepare demo pools for every task before assigning experiment subsets. MIKASA-Robo
v2 has 90 tasks with 50 replay-derived demos each. A learner receives only its
assigned task H5; use `cam subset` when assigning fewer demos. Never pad shortfalls.

For RoboMME, `cam extract --count 25` selects 9 easy, 8 medium, and 8 hard demonstrations,
evenly spaced by episode rank within each difficulty. Extraction validates raw
IDs, seeds, difficulty, completion, allowed fields, and lack of test-seed overlap.
It records selection, provenance, and output hashes in the extraction manifest.
Use depth when available, unless the benchmark explicitly prohibits it. MIKASA's
raw release contains 128x128 RGB only; the original adapter keeps that interface.
The v2 replay dataset and live adapter use 256x256 RGB-D. Rendered depth is
derived sensor data, with provenance in the manifest.

Existing extractions use `cam-demonstrations/1` for RoboMME or
`cam-demonstrations/2` for MIKASA-Robo. An old
`cap-regular-fewshot/2` or `data_1206` file needs re-extraction from verified raw
recordings or an audited conversion. Preserve originals, verify identities and
hashes, copy only permitted fields without changing values, and record provenance.
Changing a schema label alone is not a conversion. See the
[historical selection rule](ROBOMME_25_DEMO_SEEDS.md).

## RMBench RGB-D workflow

RMBench also supports the user-requested RGB-D collection at 320x240 from all
four cameras. Its release contains sampled states, but omits the dense native
trajectory files, so exact replay of the original recording is unavailable.
`scripts/prepare_rmbench_rgbd.py` regenerates the official expert on the same
500 released training identities (50 per task), records RGB and depth together,
and retains only native successful collections. It labels this as regenerated
data, saves failure/trajectory evidence privately, and creates a separate
up-to-25-demo subset. Counts and shortfalls are explicit; test seeds are excluded.
The job normally waits for idle GPUs. For the user-requested five-per-GPU collection,
use `start --gpus 0 1 --workers 10 --allow-gpu-sharing`; this explicitly permits
sharing existing workloads after checking free GPU memory, with separate five-worker
queues on each device. Capture starts as source episodes pass verification.
To resize an active collection, run `scripts/resize_rmbench_rgbd.py start JOB
--workers-per-gpu 5`. It creates a new scheduler timetag, preserves the original
acquisition snapshot and data directory, adopts active captures, and runs only
missing attempts. It checks free memory before each launch and reserves memory
for workers still starting. The original collection settings remain in the
manifest, with the scheduling transition recorded in `collection_history.json`
and the final manifest. The background monitor follows the new scheduler.
`--reuse-captures POOL` preserves completed attempts from an earlier preparation
after checking source configuration, identities and observation hashes; reuse is
recorded in each result. New experiments using these
datasets require `depth: true` and fresh preparation; historical RGB runs remain
unchanged. No state row from the released HDF5 is treated as a control command.

The current user-requested RMBench follow-up assigns **up to 50 RGB-D demos per
task, demo-only**, across all ten tasks with Claude Code / Opus 5.5. Use
`configs/experiments/rmbench_rgbd_demos50_all10.json` and eight coding workers.
The full successful pool supplies 496 demos: 49 each for `battery_try` and
`blocks_ranking_try`, 48 for `observe_and_pickup`, and 50 for the other seven
tasks. The plan declares these shortfalls through explicit `demo_counts`;
preparation and the learner boundary check require the exact declared count for
each task. Defaults remain strict, with no automatic acceptance of fewer demos.
This follow-up uses `depth: true`, the shared demo-only prompt, and zero training
rollouts. Preserve the separate 25-demo subset and all historical RGB results.

While assembly is still running, queue the follow-up with
`.venv/bin/python scripts/queue_rmbench_development.py start
configs/experiments/rmbench_rgbd_demos50_all10.json --producer
outputs/rmbench_rgbd_preparation/20261108T003437_459008Z --workers 8`.
The queue copies its experiment, config and launcher under a fresh
`outputs/rmbench_development_queue/` timetag. It waits for the producer's final
pool manifest and removal of the partial manifest, which occurs only after pool
validation. It need not wait for the separate 25-demo subset. The new development
batch then pins the current harness and simulator source, validates the data
again through `cam prepare`, creates fresh per-task runs, and checks learner
access before starting development. Queue `status.json` records errors or launch;
`batch.json` points to the development batch. Held-out testing remains separate
and follows frozen policies.

The same live dashboard can follow this queue before development starts:
`cam site --batch outputs/rmbench_development_queue/QUEUE_TAG --output
sites/rmbench_rgbd_demos50`, then
`cam serve sites/rmbench_rgbd_demos50 --port 8915`.
Open `http://localhost:8915/#development` (forward port 8915 over SSH when
needed). It displays tasks waiting for RGB-D validation, then discovers the
launched batch, prepared runs, development logs and frozen policies on refresh.

The user initially authorized full RMBench testing after **all ten** demo-only learners
finish and freeze their policies. Queue this separately with
`.venv/bin/python scripts/queue_rmbench_evaluation.py start
outputs/rmbench_development_queue/QUEUE_TAG --gpus 0 1 --workers-per-gpu 6`.
The new `outputs/rmbench_evaluation_queue/` job checks every completed learner,
zero training attempts, frozen policy hashes, and 100 assigned test episodes per
task before starting any test. It evaluates 1,000 episodes using each run's
unchanged pinned runner. One task occupies each GPU, with at most six episode
workers; strict admission waits for idle GPU compute and queued MIKASA simulation.
Only unscorable infrastructure episodes may be archived with `cam recover-test`
and retried with the same policy, seed and limits (up to three evaluation passes
per task). Scored failures remain scored. The job's `results.json` reports actual
scored/expected counts and only reports a final success rate at full coverage.
The existing dashboard's Evaluation tab follows the same prepared runs and shows
their test performance. Test outcomes never trigger new learner work.

The RMBench user subsequently requested local SSD storage and per-task startup
instead of waiting for the entire pool. Use
`.venv/bin/python scripts/develop_rmbench_progressively.py start OLD_QUEUE`.
This creates a fresh `outputs/rmbench_progressive_development/TAG` with pinned
code and unchanged acquisition data, retires only the waiting development and
evaluation schedulers and the completed-capture NFS assembler/monitor, and keeps
all original captures, partial H5s, manifests and logs. The new H5s live on the
local `/data` SSD at `JOB/datasets/TASK/h5/record_dataset_TASK.h5`; the user will
copy them to NFS later. Two assembly slots perform sequential capture reads and
local HDF5 copying. Every capture hash and every output schema/training identity
is checked. Five seeded random successful episodes per task additionally receive
first/middle/last-frame checks across all four RGB-D cameras, with an audit JSON
and contact sheet. These are offline file inspections, not simulator replays.
Each task starts through fresh `cam prepare` and the same eight-slot demo-only
development schedule when its own dataset is ready. Point the dashboard at this
new job with `cam site --batch JOB --output sites/rmbench_rgbd_demos50`; its status
has the same task/run structure as a development batch. Queue a new
`queue_rmbench_evaluation.py start JOB` follow-up; the all-ten-frozen barrier and
1,000 held-out test assignment remain unchanged. The original 25-demo subset
preparation plan is preserved and not relabeled as this full-pool assignment.

**Latest RMBench evaluation scheduling revision:** start each task's 100 held-out
episodes as soon as its own learner has finished and its policy is frozen; other
tasks may continue offline development. Launch with
`scripts/queue_rmbench_evaluation.py start DEVELOPMENT_JOB --as-ready --gpus 0 1
--workers-per-gpu 6 --supersedes OLD_WAITING_EVALUATION_JOB`.
This writes a fresh scheduler timetag and preserves the old waiting scheduler's
plan, status history and snapshots. It changes no prepared learner or policy.
Each admission still checks successful learner completion, zero training
attempts, the frozen program hash, RGB-D configuration and all 100 assigned test
identities. GPU admission, infrastructure-only retries and the prohibition on
test feedback to any learner remain unchanged. Per-task complete success rates
appear as tasks finish testing; the overall rate remains incomplete until all
1,000 tests are scored. The historical default without `--as-ready` retains the
all-task barrier for reproduction.

**Queued RMBench demo-count comparison (user-requested):** after all ten current
up-to-50-demo learners and all 1,000 baseline tests finish, run a separate
25-demo RGB-D demo-only experiment across all ten tasks. Queue it with
`.venv/bin/python scripts/queue_rmbench_comparison.py start BASELINE_DEVELOPMENT`.
The queue copies the baseline's exact harness, benchmark source, configuration,
shared prompts and launcher, and records deterministic subset identities before
launch. It waits for complete scored coverage, irrespective of success rate.
Then the baseline's `cam subset` creates self-contained local SSD task H5s with
exactly 25 demos each (250 total). Fresh `cam prepare` runs use eight coding
slots, Claude Code / Opus 5.5 and zero training rollouts. Larger pools, previous
solutions and test feedback are unavailable to learners. The separate evaluator
admits each new frozen policy using the same 100 test identities, one task per
GPU and at most six workers per GPU. Existing infrastructure-only recovery rules
apply. `outputs/rmbench_comparison_queue/TAG/` contains selection.json, plan.json,
status.json, datasets/TASK, batch.json and evaluation_followup.json. After both
arms have complete coverage, comparison.json and comparison.md report per-task
and overall results, with the baseline's actual 48/49/50 counts. All original
datasets and run evidence remain intact; no NFS transfer happens automatically.

Existing extractions use `cam-demonstrations/1` for RoboMME or
`cam-demonstrations/2` for the three additional adapters. An old
`cap-regular-fewshot/2` or `data_1206` file needs re-extraction from verified raw
recordings or an audited conversion. Preserve originals, verify identities and
hashes, copy only permitted fields without changing values, and record provenance.
Changing a schema label alone is not a conversion. See the
[historical selection rule](ROBOMME_25_DEMO_SEEDS.md).

## Learner access

The host runs the simulator and authenticates the coding client. Model-accessible
I/O is restricted to six CaM tools: `command`, `read_image`, `usage`, `snapshot`,
`validate`, and `rollout`. Built-in shell/file/browser/delegation tools are disabled.
Coding commands run in Bubblewrap's private mount, PID, and network namespaces,
without host credentials or network access.

| Sandbox path | Contents and access |
|---|---|
| `/workspace` | Read/write task policy, notes, `scratch/`, prompt, public `api_reference/` |
| `/demonstrations` | Read-only assigned sanitized H5; absent in interaction-only mode |
| `/feedback` | Read-only counted training feedback; empty in demo-only mode |
| `/runtime` | Read-only public CaM policy API |
| `/packages` | Read-only allowlisted analysis libraries |
| `/usr`, `/bin`, `/lib`, `/lib64`, `/python`, `/tools` | Read-only runtime support; `python`, `python3`, `python3.11` use Python 3.11 |
| `/tmp`, `/home`, `/proc`, `/dev` | Private temporary directories and namespaced process/device views |

**`inputs/benchmark/<package>/` is private simulator code.** It is not copied into
the coding workspace or mounted into coding commands. The public API reference
contains policy/memory, observation, action, and trajectory interfaces, without
benchmark implementations or scoring logic. Host run files, seeds, test metadata,
logs/audits, other runs, raw datasets, and historical solutions are not mounted.
`read_image` accepts allowed images only in the first three task directories.
Simulator environments and linked assets are also private. Runs record their
interpreter, dependency versions, and asset hashes; coding commands and deployed
policies continue to use the separate CaM Python 3.11 sandbox.

During policy execution, `/workspace` contains frozen program files read-only.
Development scratch, demonstrations, and feedback are absent. Only allowed
sensor observations and robot RPC cross from the simulator. A task-provided
reset video, when enabled, is an episode observation. Evaluation makes no model
calls, and declared memory resets between episodes.

MIKASA-Robo adapters also expose `robot.dry_run(..., mode="absolute" or "delta")`
inside episode action routines. It checks the requested next arm joint target
against robot limits without advancing physics or inspecting task state. This
uses the existing robot RPC, keeping the same six coding tools and counted
training rollouts. The host caps queries at `max_action_steps` per action routine.
Requests/results are saved in each rollout's `dry_runs.jsonl` and copied to
training feedback. Only the [public reference](../src/cam/benchmarks/mikasa_robo_api/api.md)
is copied into the learner's API reference; the simulator-side implementation
stays private. It does not check collisions or guarantee physical stability.
Prepare a fresh timetag to use this API; do not modify historical snapshots.

MIKASA also exposes the pure offline `cam.runtime.mikasa_robo.safety_check`
API through coding commands, including demo-only mode. The user-authorized public
files are `offline.py`, sanitized `panda_model.json` (robot geometry, limits and
controller parameters), the package initializer and `safety_check.md`. These are
copied to `/runtime` and `api_reference`; live-query implementations and benchmark
source remain private. The helper uses only caller-supplied robot state/actions,
without simulator, RPC, hidden state or GPU. It reports joint-position margins,
supplied torques versus URDF effort limits, and initial PD-demand estimates versus
simulator drive caps when velocities and targets are supplied. Missing information
is explicitly unavailable; contacts and actual future torques are not predicted.
Calls/results are retained in the normal command audit, not counted as rollouts.

## Learning and held-out evaluation

The shared demo-only prompt was revised at the user's request on 2026-11-08 to
require a final code review and offline execution checks after the last edit.
Check the first decision with every declared memory field and `last` set to
`None`, later memory transitions, types, empty detections and action arguments.
`validate` checks interfaces; it does not execute the policy. These checks use
assigned demo observations or synthetic inputs and permit no simulator calls.
All benchmarks use the same wording; old prepared prompts stay intact.

The user also authorized one additional Opus 5.5 repair session for the existing
50-demo RMBench `swap_blocks` policy. For this explicit continuation, run
`cam prepare runs --dataset SAME_DATA --config SAME_CONFIG --tasks swap_blocks
--initial-policy-run SOURCE_RUN`. This option requires one demo-only task,
identical configuration and assigned demos, an intact frozen source policy,
and the same native benchmark snapshot. It imports only policy files and their
pre-evaluation notes, pins them in `inputs/initial_policy/`, records lineage in
`run.json` and the README, and initializes a fresh code history. No scored
results, evaluation traces, learner transcript or test feedback are imported.
The prompt identifies a user-requested repair instead of a blank-policy start.
Use this option only when the user explicitly authorizes continuing an existing
solution. Preserve the original policy and scored result; any later evaluation
of the repair is a separate follow-up, not a replacement of the original
25-versus-50-demo comparison.

The subsequent user request supersedes reuse for the next `swap_blocks` run:
start from the blank policy with the same 50 RGB-D demos and revised shared
demo-only prompt, then evaluate all 100 tests. Do not pass `--initial-policy-run`.
After complete scored coverage, make the new development and rollout selection
current on the website regardless of success rate. The user's latest display
revision removes the original and repair entries from the website entirely,
including the Historical selector, while preserving their source files on disk.
The 25-demo swap_blocks entry remains part of its separate experiment.
The host-only `scripts/publish_rmbench_rerun.py` records `.run_visibility.json`
in the website directory and writes `current_results.json` beside the new job;
it does not change old logs, policies, scores or the original comparison. The
current selection explicitly records this task's revised prompt and new run ID.
The site-local `.excluded_runs.json` lists the retired run paths and takes
precedence over historical visibility metadata. Excluded runs contribute no
development logs, rollout entries, downloads or aggregate statistics to the
generated dashboard. Source indices remain reserved so other run links stay stable.

Finish learning and freeze the policy before invoking `cam evaluate` on the host.
`cam learn` does not launch evaluation. The coding agent has no evaluation tool;
`rollout` permits only assigned training episodes and is disabled in demo-only
mode. Never feed test results back to the learner or tune the reported policy
using them. Preserve scored failures and report actual scored/expected counts,
with infrastructure errors separate.

The host may inspect the repository and repair infrastructure, but must not give
the learner simulator internals, hidden state, old solutions, or test feedback.
Do not edit pinned snapshots to bypass integrity checks. Changes to the harness,
configuration, prompt, or data require a fresh preparation and timetag.

Before each paid Claude session, the host verifies the pinned MCP server with
initialization, all six tool schemas and a read-only `usage` call. This probe
is recorded separately in `logs/session_*/mcp_probe/` and consumes no training
attempts. Claude initialization must also expose exactly the six CaM tools and
report the server connected; missing tools fail the session. A model-only
connectivity check is insufficient. The process watcher falls back to checking
the parent relationship when Python lacks `os.pidfd_open`, preserving client
lifetime cleanup without depending on the spawning thread.

## Development evaluation and guide changes

Diagnose a weak frozen policy on held-out *training* identities, never on test
episodes:

```bash
.venv/bin/python scripts/dev_evaluate.py "$run_dir" --episodes 16 --gpus 0 --workers 4
```

It runs the run's frozen `policy/` through the run's own pinned harness,
benchmark snapshot, interpreters and limits on training identities that are
neither assigned demonstrations nor simulator training seeds, and writes to
`outputs/dev_eval/`; the run, its ledger and test evidence are untouched. A fix
may repair interface, observation or accounting defects and document public
sensor/action semantics. Do not turn diagnoses into task-specific success
contracts, scorer thresholds or solution hints in learner prompts. The learner
must infer those conditions through its assigned demos and counted interactions.
Never feed unassigned development-evaluation evidence back into that learner.
Any change requires a fresh preparation and timetag,
and every evaluated run's held-out result is reported, including earlier ones.

**Explicit exception (user-authorized, 2026-11-07):** new runs of MIKASA-Robo
`SeqOfColors3-VLA-v0` receive a [static clarification](../src/cam/prompts/clarifications/mikasa_robo_seq_of_colors3.md)
to touch colors in cue order, overriding the native goal's "in any order" wording.
This applies only to that task in the `mikasa_robo` and `mikasa_robo_v2` adapters,
in every learning mode; the RGB-D follow-up retains the same authorized correction.
`cam prepare` records the text and source in `run.json` (`task_clarifications`),
the generated README, and the learner prompt. Native goals and scoring remain
unchanged; historical runs retain their original prompts. This is the sole
authorized task-specific correction; shared core/mode prompts stay generic.

## Audit and recovery

For a live development dashboard, including logs, code revisions and training
rollouts with the exact policy used in each attempt:

```bash
cam site --batch outputs/development_batches/BATCH --output sites/development
cam serve sites/development --port 8913
```

Open `http://localhost:8913/#development` (forward port 8913 when using SSH).
The visual task cards and individual attempt links open **separate tabs**, leaving
the dashboard in place. Each replay page has camera videos, synchronized action
stages and policy memory labels, frame stepping, and links to the executed code
and agent log. Labels come from recorded policy decisions, not inferred success
annotations. Automatic refresh preserves playback; **Refresh log** reloads an
open transcript. The batch discovers newly prepared tasks automatically. Batch errors
and frozen-policy state are shown separately. This host viewer does not start
learning or evaluation and is not available inside the learner sandbox.

**Reference demos** opens the run's assigned training examples in separate tabs.
Existing preview videos are reused; `cam serve` encodes missing previews from
the sanitized camera frames on demand, with at most two single-thread CPU encoders.
The cache lives under the site, leaving datasets and run snapshots unchanged.
Demo playback has camera labels and frame controls; it does not invent expert
decision labels. RMBench recordings are sampled frames, not a real-time timing reference.

```bash
.venv/bin/python scripts/audit_run.py "$run_dir" --task ButtonUnmaskSwap
```

This writes `orchestration/commands.txt`, `access_audit.json`, and `ACCESS_AUDIT.md`,
listing offered/used tools and scanning completed command traces for Internet
sockets and successful absolute opens under private host prefixes. Run it after
development to cover all calls. It is a bounded scan: relative/dynamic paths and
other syscall forms may need manual review of `audit/call_*.json` and `.strace`.
Bubblewrap's host-side mount setup and the client's provider traffic are distinct
from access by sandbox commands.

[Run layout and recovery](run_layout.md) documents transcripts, every source
revision, immutable checkpoints, and retry commands. `validate` checks interfaces
and restrictions, not physical task success. Resume only unchanged unfinished
runs, preserve prior evidence, and record the relationship of any replacement run.
