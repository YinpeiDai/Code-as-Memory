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
| `demo_interaction` | `both` | Assigned H5 | Configured budget |

Use the requested client, model, and setting. Example interaction configs allow
20 attempts across three fixed training seeds, one per difficulty, cycling
7/7/6 times by R20. Demo count and interaction seed count are separate.
`training_seed_count: null` permits all extracted identities; report actual
coverage. R1/R5/R10/R20 are causal checkpoints from one run, not replicates.

## Demonstration validation

`cam extract --count 25` selects 9 easy, 8 medium, and 8 hard demonstrations,
evenly spaced by episode rank within each difficulty. Extraction validates raw
IDs, seeds, difficulty, completion, allowed fields, and lack of test-seed overlap.
It records selection, provenance, and output hashes in the extraction manifest.
RGB-D is the default; `--rgb-only` is an explicit sensor ablation.

Existing extractions must use `cam-demonstrations/1`. An old
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

**`inputs/benchmark/robomme/` is private simulator code.** It is not copied into
the coding workspace or mounted into coding commands. The public API reference
contains policy/memory, observation, action, and trajectory interfaces, without
benchmark implementations or scoring logic. Host run files, seeds, test metadata,
logs/audits, other runs, raw datasets, and historical solutions are not mounted.
`read_image` accepts allowed images only in the first three task directories.

During policy execution, `/workspace` contains frozen program files read-only.
Development scratch, demonstrations, and feedback are absent. Only allowed
sensor observations and robot RPC cross from the simulator. A task-provided
reset video, when enabled, is an episode observation. Evaluation makes no model
calls, and declared memory resets between episodes.

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

## Audit and recovery

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
