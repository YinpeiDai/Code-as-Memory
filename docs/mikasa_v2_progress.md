# MIKASA-Robo v2 development inventory

## Current CPU demo + interaction assignment

The original machine is developing the same ten tasks listed below in batch
`20261108T040959_138834Z`: 50 demos and 50 rollouts on the same demo seeds,
CPU physics, Claude Code / Opus 5.5, eight coding jobs. These are new policies,
separate from the completed demo-only batch.

The [CPU remaining-80 plan](../configs/experiments/mikasa_v2_cpu_50_demoseeds_remaining80.json)
contains the other 80 official tasks and is ready for the second machine; it has
not been launched here. It uses the same assignment and CPU settings, with the
concise MIKASA prompt. The running ten-task batch retains its original longer
prompt. Record the second machine's batch tag before making further assignments.

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
