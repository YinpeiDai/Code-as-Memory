# MIKASA-Robo v2 development inventory

## Current CPU demo + interaction assignment

The [MCP startup incident report](mikasa_v2_mcp_startup_incident.md) explains
the two infrastructure bugs, their impact, the repair and verification evidence.

**Replacement launched, 2026-11-08 14:22 UTC:** batch
`20261108T142219_066189Z`, supervisor PID `2356076`, has the repaired MCP startup
and required-tool checks. All 80 tasks get fresh policies and snapshots; the
50-demo/50-demo-seed assignment, ten coding jobs and four evaluation workers
(two per GPU) are unchanged. At launch, four tasks were preparing and 76 waiting;
no replacement learning or evaluation result is claimed yet. Relevant tests:
37 passed. A second real Claude smoke check, including blocking MCP startup,
passed with all six tools connected and one actual `usage` call (reported cost
$0.0042206), recorded in `live_claude_smoke_003/result.json` under the fixture
diagnostic directory below. The failed batch is superseded, not resumed.

**Failed batch snapshot, 2026-11-08 14:06 UTC:** batch
`20261108T044922_221919Z` has produced no valid learning or test results:
0 actual CaM tool calls, 0 counted training attempts, 0 frozen policies and
0/4,000 scored test episodes across the 41 run directories checked. Batch state
was 37 failed, four in the learning stage, four preparing and 35 waiting.
Recorded model cost was approximately $44.28 (CLI estimates from completed
sessions, excluding compute). Client initialization records show empty tool
lists and the CaM MCP connection pending or failed; some runs also failed
preflight or isolated validation. Transcript claims of rollouts are unsupported
and must not be reported as results.

The supervisor and active learner process trees were suspended with `SIGSTOP`
to prevent further failed launches/model charges; existing preparation children
were left running. All original evidence and snapshots remain intact. See
`outputs/diagnostics/status_20261108T1403Z.json` and the batch's
`orchestration/learning_suspension.json` under the NFS artifact root for counts,
process identities and the suspension record. After verifying the repair, those
same suspended processes were terminated after checking their PID start times;
`orchestration/learning_termination.json` records this. All logs and snapshots
remain intact. The changed harness uses the fresh replacement timetag above.

**Confirmed cause:** both the pinned service and local source crashed before
MCP initialization with `AttributeError: module 'os' has no attribute
'pidfd_open'`. This machine's CaM Python 3.11.11 build lacks that wrapper. The
client guard rejected extra tools but accepted an empty tool list, allowing
paid sessions to continue without CaM. Reproduction traces are in
`outputs/diagnostics/mcp_startup_20261108T141644Z/measurements.json`. The later
unchanged-starter errors are consequences; NFS-related readiness/validation
timeouts are separate infrastructure failures. The repair adds a compatible
parent watcher, a pre-inference MCP probe and mandatory connected-tool checks.
A real Claude/Opus 5.5 smoke test on a fresh synthetic fixture connected all six
tools and completed one actual `usage` call with no simulator activity (reported
model cost $0.0045406). Evidence is under
`outputs/diagnostics/mcp_repair_fixture_20261108T1430Z/live_claude_smoke_002/`.

The original machine is developing the same ten tasks listed below in batch
`20261108T040959_138834Z`: 50 demos and 50 rollouts on the same demo seeds,
CPU physics, Claude Code / Opus 5.5, eight coding jobs. These are new policies,
separate from the completed demo-only batch.

The [CPU remaining-80 plan](../configs/experiments/mikasa_v2_cpu_50_demoseeds_remaining80.json)
contains the other 80 official tasks. They are assigned on `sled-aspen.eecs.umich.edu`
to replacement batch **`20261108T142219_066189Z`**, launched at 14:22 UTC.
Do not launch duplicate learners. This batch uses the same demo/seed
assignment and CPU settings, with the concise MIKASA prompt; the original
ten-task batch retains its earlier longer prompt.

The second machine runs **10 coding jobs**, four concurrent task preparations,
and automatic evaluation after each task finishes and freezes at R050. One task
is tested at a time with **four episode workers, two per GPU** (GPUs 0 and 1),
alongside at most one training simulator per GPU. Expected held-out coverage is
**80 × 50 = 4,000 episodes**; actual scored counts and infrastructure failures
must be read from the run reports and batch status, not inferred from launch.

Artifacts are under
`/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/mikasa_robo/`:

- Current batch: `outputs/development_batches/20261108T142219_066189Z/`.
- Current runs: `runs/mikasa_robo_v2/<task>/cc-opus5.5/demo_interaction_v2/20261108T142219_066189Z/`.
- Preserved failed batch and runs: timetag `20261108T044922_221919Z`.
- Input partitions: `data/mikasa_v2_task_partitions_20261108T044730_506152Z/`.
- Setup evidence: `outputs/setup/20261108T044638_191819Z/`.
- Repair evidence: `outputs/diagnostics/mcp_repair_20261108T1422Z/`.

The replacement uses `local/mikasa_cpu_80_mcp_repair_plan.json` (also preserved
as `launch_plan.json` under the repair evidence directory) with
`--workers 10 --prepare-workers 4 --gpus 0 1
--evaluate-on-freeze --evaluation-workers-per-gpu 2`. Each per-task dataset keeps
the original H5 bytes, hashes, all 50 episode identities, and manifest provenance;
preparation validates each before admitting its learner. The original replay
pool is unchanged. At launch, four tasks were preparing and 76 were waiting;
no test episodes had yet been scored. Check the live `status.json` for progress.

Replay data: `/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/data/mikasa_robo_v2_50`
(`manifest.json` plus `h5/record_dataset_<Task-ID>.h5`). All 90 tasks have 50
episodes; all files referenced by the remaining-80 plan were present at this check.
Follow the [current setup and launch guide](mikasa_v2_setup.md).

## Historical demo-only snapshot

Snapshot: **2026-11-08 01:58 UTC**. Scope: canonical `runs/mikasa_robo_v2/`
on the original development machine. Historical `mikasa_robo` runs do not count
as v2 development. Update this inventory when another machine starts tasks.

| State | Tasks |
|---|---:|
| Official MIKASA-Robo v2 task set | 90 |
| Original selection: development completed, policy frozen at R000 | 10 |
| Remaining: no v2 run found in the checked storage | 80 |

All ten selected tasks have now finished, including TakeItBack. Completion means
the agent exited successfully and its frozen policy hash was verified; it does
not imply simulator task success. This batch used 50 RGB-D demos per task,
Claude Code / Opus 5.5 and zero training rollouts. No held-out results were present
at this check.

Evidence: `outputs/development_batches/20261108T003718_500842Z/status.json`;
each task's `logs/completion.json`, `policy/freeze.json`, and `checkpoints/R000/`
under `runs/mikasa_robo_v2/<task>/cc-opus5.5/demo_only/20261108T003718_500842Z/`.
Logs and policy files remain outside Git. The unfinished preparation batch
`20261108T002858_961829Z` launched no learner and adds no developed tasks.

## Original ten tasks — completed

```text
BatteriesCheckerEasy-3-VLA-v0
BlinkCountButtonPressEasy-VLA-v0
BunchOfColors3-VLA-v0
ChainOfColors3-VLA-v0
FindImposterColor3-VLA-v0
RememberColor3-VLA-v0
SeqOfColors3-VLA-v0
ShellGameTouch-VLA-v0
TakeItBack-VLA-v0
TimedTransferEasy-VLA-v0
```

## Remaining 80 tasks

The [remaining-task plan](../configs/experiments/mikasa_v2_demos50_remaining80.json)
contains exactly these IDs and excludes the original ten. All 80 have 50 demos
in the completed shared NFS replay dataset. The plan retains demo-only learning,
zero training attempts and a separate held-out evaluation step. It has **not
been launched** as part of this inventory update.

```text
BatteriesCheckerEasy-6-VLA-v0
BatteriesCheckerHard-3-VLA-v0
BatteriesCheckerHard-6-VLA-v0
BlinkCountButtonPressEasy-Long-VLA-v0
BlinkCountButtonPressHard-Long-VLA-v0
BlinkCountButtonPressHard-VLA-v0
BlinkCountButtonPressMedium-Long-VLA-v0
BlinkCountButtonPressMedium-VLA-v0
BunchOfColors3-Long-VLA-v0
BunchOfColors5-Long-VLA-v0
BunchOfColors5-VLA-v0
BunchOfColors7-Long-VLA-v0
BunchOfColors7-VLA-v0
ChainOfColors3-Long-VLA-v0
ChainOfColors5-Long-VLA-v0
ChainOfColors5-VLA-v0
ChainOfColors7-Long-VLA-v0
ChainOfColors7-VLA-v0
FindImposterColor5-VLA-v0
FindImposterColor9-VLA-v0
FindImposterShape3-VLA-v0
FindImposterShape5-VLA-v0
FindImposterShape9-VLA-v0
FindImposterShapeAndColor3x2-VLA-v0
FindImposterShapeAndColor3x3-VLA-v0
FindImposterShapeAndColor5x3-VLA-v0
GatherAndRecall1-VLA-v0
GatherAndRecall3-VLA-v0
GatherAndRecall5-VLA-v0
GatherAndRecall7-VLA-v0
GatherAndRecall9-VLA-v0
InterceptFast-VLA-v0
InterceptGrabFast-VLA-v0
InterceptGrabMedium-VLA-v0
InterceptGrabSlow-VLA-v0
InterceptMedium-VLA-v0
InterceptSlow-VLA-v0
RememberColor3-Long-VLA-v0
RememberColor5-Long-VLA-v0
RememberColor5-VLA-v0
RememberColor9-Long-VLA-v0
RememberColor9-VLA-v0
RememberShape3-Long-VLA-v0
RememberShape3-VLA-v0
RememberShape5-Long-VLA-v0
RememberShape5-VLA-v0
RememberShape9-Long-VLA-v0
RememberShape9-VLA-v0
RememberShapeAndColor3x2-Long-VLA-v0
RememberShapeAndColor3x2-VLA-v0
RememberShapeAndColor3x3-Long-VLA-v0
RememberShapeAndColor3x3-VLA-v0
RememberShapeAndColor5x3-Long-VLA-v0
RememberShapeAndColor5x3-VLA-v0
RotateLenientPos-VLA-v0
RotateLenientPosNeg-VLA-v0
RotateStrictPos-VLA-v0
RotateStrictPosNeg-VLA-v0
SeqOfColors3-Long-VLA-v0
SeqOfColors5-Long-VLA-v0
SeqOfColors5-VLA-v0
SeqOfColors7-Long-VLA-v0
SeqOfColors7-VLA-v0
ShellGameColorLampTouch-VLA-v0
ShellGamePush-VLA-v0
ShellGameShuffleColorLampTouch-Long-VLA-v0
ShellGameShuffleColorLampTouch-VLA-v0
ShellGameShuffleTouch-Long-VLA-v0
ShellGameShuffleTouch-VLA-v0
TimedTransferEasy-Long-VLA-v0
TimedTransferHard-Long-VLA-v0
TimedTransferHard-VLA-v0
TimedTransferMedium-Long-VLA-v0
TimedTransferMedium-VLA-v0
TraceShapeEasy-VLA-v0
TraceShapeHard-VLA-v0
TraceShapeMedium-VLA-v0
TraceShapeSeqEasy-VLA-v0
TraceShapeSeqHard-VLA-v0
TraceShapeSeqMedium-VLA-v0
```

## Historical demo-only launch

For the earlier zero-rollout protocol only, copy
`configs/experiments/mikasa_v2_demos50_remaining80.json` as your local plan instead
of the ten-task plan. It already points at:

```text
/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/data/mikasa_robo_v2_50
```

Set the local plan's `config` to your account-specific local config, then launch:

```bash
.venv/bin/python scripts/develop_batch.py start local/mikasa_demos50_plan.json \
  --workers 8 --gpus 0 1
```

Use the new machine's GPU IDs. Record its new batch tag here to prevent duplicate
assignments across machines; the current inventory is a dated snapshot, not a
cross-machine scheduler.
