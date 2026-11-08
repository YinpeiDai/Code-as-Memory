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
