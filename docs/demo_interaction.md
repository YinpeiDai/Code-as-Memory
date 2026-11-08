# Demo + interaction: v1 and v2

CaM supports two explicit prompt strategies for `mode: "both"`. They share the
core prompt, benchmark guide, six tools, training seed schedule, attempt budget,
checkpoint rules, logging, and held-out evaluation. The strategy is expressed
through instructions to the coding agent; it does not change simulator access
or introduce a separate tool-level gate.

| Version | Learning strategy | Prompt | Run setting |
|---|---|---|---|
| `v1` | The agent freely chooses when to inspect demos, check code offline, or interact. No required demo-first phase or fixed alternation. | [both_v1.md](../src/cam/prompts/modes/both_v1.md) | `demo_interaction_v1` |
| `v2` | Build/check a solution from all demos first. Use rollouts to resolve uncertainties, then use remaining distinct training seeds to verify and strengthen the solution. | [both_v2.md](../src/cam/prompts/modes/both_v2.md) | `demo_interaction_v2` |

Use **v2 with at most 25 attempts per task** for the current experiments. Both
versions' templates use the same 25-attempt ceiling for controlled comparisons.
V1 permits justified early stopping. V2 uses all remaining attempts when the
assignment provides distinct training seeds, even when the solution seems ready
for testing. Verifying unchanged code on another assigned seed is sufficient
reason for a rollout; no new hypothesis or code change is required. Repeating
seed schedules still permit justified early stopping.
MIKASA-Robo templates assign 25 distinct interaction seeds disjoint
from the offline demos and tests (`training_seed_policy: "disjoint_once"`).
RoboMME retains its historical `demo_cycle` schedule. Seed policy is independent
of the v1/v2 prompting choice and must match in controlled comparisons.
The MIKASA-Robo v2 RGB-D follow-up explicitly uses 50 offline demos and up to
50 distinct interactions per task; this changes the budget, not the v2 strategy.
Every simulator attempt counts, including
diagnostics, initialization failures and early stops. Both preserve intermediate
code, notes, offline checks, API usage, and rollout feedback. R1/R5/R10/R20/R25 are
causal checkpoints of the same development run, not independent repeats.

New prompts contain the native goal and public interface documentation, without
task-specific contracts or thresholds derived from scorer code. The learner
infers success conditions and unobserved task variables from assigned demos,
allowed sensor history, action outcomes, and terminal training feedback. Hidden
state, private stage scores, benchmark source, and test feedback stay inaccessible.

### MIKASA motion-verification supplement

New `mikasa_robo` and `mikasa_robo_v2` runs with `both_protocol: "v2"`
automatically add the [MIKASA-specific prompt](../src/cam/prompts/benchmarks/mikasa_robo_both_v2.md).
It keeps demo-first development, then uses counted training rollouts to verify
control response and revise the policy. It covers offline `safety_check`, live
`robot.dry_run`, rotation-command oscillation, and measured settling. Joint-limit
or estimated torque checks cannot establish dynamic stability or task success.
The guidance is task-independent and supplies no scorer thresholds or prior
rollout findings. Shared core/mode prompts and other learning settings are unchanged.

Preparation pins its exact text and source in `run.json` under
`benchmark_mode_guidance`, includes it in `prompt.md` / `workspace/PROTOCOL.md`,
and links it from the generated README. This is a user-requested prompt revision
within v2, with unchanged tools and budgets; report this supplement when comparing
protocols. Use a fresh timetag. Historical runs retain their original prompts.

The current ten-task MIKASA run uses
`configs/experiments/mikasa_v2_cpu_50_demoseeds_10tasks.json`: 50 RGB-D demos and
50 attempts on those same 50 demo identities with CPU physics for development and later
evaluation. GPU rendering remains enabled. The learner may reread and analyze
assigned demos without an access-count limit; only simulator attempts consume
the 50-attempt budget. Use every remaining attempt even if the code seems ready.
The supplement asks the learner to explore release/lift and timing hypotheses,
verify intermediate motion and joint margins, and avoid unchecked large commands.
Any inferred completion parameter must come from its own allowed evidence.
`demo_cycle` with count 50 and budget 50 gives one pass through the demo seeds,
without repeats or new training seeds. The host enforces this schedule: rollout
arguments cannot override seeds, and the ledger rejects unassigned/out-of-order
identities and attempts after the cap. Test seeds stay excluded.

## Choose the version

Use a versioned configuration in `configs/<benchmark>/`:

- `both_v1_claude_code.json` or `both_v1_codex.json`
- `both_v2_claude_code.json` or `both_v2_codex.json`

These templates are provided for RoboMME and both MIKASA-Robo adapters.
Match the template's sensor settings to the extracted data, and configure the
chosen client's model and authentication as described in the first-run guide.
The selector in the run configuration is:

```json
{
  "mode": "both",
  "both_protocol": "v2",
  "budget": 25
}
```

Use `"v1"` only when comparing the free-choice strategy. This is not a complete config;
the supplied templates include the benchmark, model, and execution limits.
`both_protocol` is required for mode `both` and is invalid for other modes.
There is no default or third simplified prompt.

For example, from the CaM checkout with a prepared RoboMME dataset:

```bash
cam prepare runs --dataset data/robomme_25_rgbd \
  --config configs/robomme/both_v1_claude_code.json --tasks BinFill
cam prepare runs --dataset data/robomme_25_rgbd \
  --config configs/robomme/both_v2_claude_code.json --tasks BinFill
```

Each command prints a new task leaf. Pass that leaf to `cam learn`, then
`cam evaluate --budgets 1 5 10 20 25` after development finishes. Omit `--budgets`
to use the milestone checkpoints plus the configured final ceiling. An early
stop carries its frozen final policy forward to later budgets. Use the Codex
templates for Codex runs. For a controlled comparison, keep the demonstrations,
training identities, model, budget, sensors, limits, and benchmark snapshot the
same between versions.

## Recorded evidence and historical runs

To start the original ten-task MIKASA v2 development comparison:

```bash
.venv/bin/python scripts/develop_batch.py start \
  configs/experiments/mikasa_25_newseeds_10tasks.json --workers 8 --gpus 0 1
.venv/bin/python scripts/develop_batch.py status outputs/development_batches/TIMETAG
```

Each start creates fresh task runs with blank policies. The background supervisor
pins its code and task selection, prepares missing 25-demo datasets, checks demo,
interaction and test identity separation, and probes each coding sandbox before
queuing it. Eight coding sessions can run concurrently; host GPU leases serialize
learning simulators on each GPU. Replay data preparation reserves one whole GPU.
Logs, per-task status and run paths are under `outputs/development_batches/TIMETAG/`.
Data/preparation failures remain explicit; tasks are never silently substituted.

With the completed 50-demo RGB-D pool, start the current CPU-physics follow-up
on the same ten MIKASA tasks (use a local copy of this plan for different dataset paths):

```bash
.venv/bin/python scripts/develop_batch.py start \
  configs/experiments/mikasa_v2_cpu_50_demoseeds_10tasks.json --workers 8 --gpus 1
```

This plan requires exactly 50 prepared demos per task and uses those same 50
identities for training interactions, once each. GPU IDs select rendering devices;
choose available devices on the host. It fails before launching if the dataset is incomplete
or has a shortfall; it never starts another rerender or pads the demos. Each task
gets a new timetag, the dry-run API reference, and the same learner access boundary.
This command launches paid learning and freezes completed policies, without
starting held-out evaluation. Existing sessions must be stopped before replacing
an active batch; interruption preserves evidence and requires explicit recovery.
The earlier `mikasa_v2_50_newseeds_10tasks.json` plan remains available separately
for the GPU/new-seed comparison and is not the current assignment.

New runs use `demo_interaction_v1/` or `demo_interaction_v2/` in their directory
hierarchy. The version is saved in `run.json`, the task's `state/config.json`
and ledger configuration, the rendered `prompt.md`, and the generated README.
Reports, figure labels, and website views distinguish both settings. Prompt and
configuration hashes are pinned during preparation.

The old simplified `prompts/modes/both.md` and unversioned `both_<client>.json`
templates have been removed. Preparing a new `both` run without an explicit
version fails with a configuration error. Already prepared runs retain their
original snapshots and logs; historical unversioned results remain labeled
`demo_interaction` and must not be relabeled as v1 or v2. Prepare a fresh run to
change protocol.

The [historical Plus v1](legacy_prompts/plus_both_v1.md) and
[historical Plus v2](legacy_prompts/plus_both_v2.md) remain reference documents.
CaM's versions implement the two learning strategies above on one shared harness.
They do not reproduce the historical v2 candidate-registration, mandatory review,
candidate-selection safeguards, or reserved confirmation cycle. In particular,
CaM v2's demo-first requirement is a prompt instruction, not an enforced gate.

On 2026-11-07, the shared v2 prompt was simplified to require using remaining
distinct training seeds for verification, replacing agent-chosen early stopping
for those assignments. This is a prompt revision within v2, with the same tools,
seed schedules, budgets and checkpoint rules. Earlier prepared v2 runs retain
their original prompts and outcomes. Use new timetags and distinguish the prompt
revisions when comparing results; do not resume old runs with the new wording.
