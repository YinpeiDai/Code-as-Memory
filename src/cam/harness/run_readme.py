"""Human-readable run organization and the actual agent visibility boundary."""

import shlex


HISTORY = """## Development history versus checkpoints

Within `tasks/<Task>/`, `history/revisions/NNNNNN.json` records time, reason,
attempt count, and digest; the matching `.diff` shows changes. Full policy files
live in `history/objects/<digest>/`; identical content shares an object. Commands,
successful validation, and explicit snapshots save revisions automatically.

`audit/call_NNNNNN.json` records tool arguments/results, with command stdout/stderr.
`audit/file_manifests/` and `audit/objects/` retain workspace files before/after
calls, including scratch and incomplete edits. The full client transcript is
`logs/session_NNN/events.jsonl`.

In demo-only learning, `checkpoints/R000/` and `policy/` appear at final freeze;
their absence during development is normal. Interaction runs checkpoint before
the next counted attempt and at final freeze. Revisions are development history;
checkpoints are immutable code for evaluation.

"""

ACCESS = """## What the coding agent can access

The host client authenticates to the provider. Model-accessible I/O uses only
`command`, `read_image`, `usage`, `snapshot`, `validate`, and `rollout`; built-in
shell/file/browser/delegation tools are disabled. Commands run in Bubblewrap with
private mount, PID, and network namespaces, without host credentials or network.

| Agent-visible path | Host source relative to this run | Access during development |
|---|---|---|
| `/workspace` | `tasks/<Task>/workspace/` | Read/write: `task.json`, `policy.md`, `tools/*.py`, analysis notes, `scratch/`, `PROTOCOL.md`, and `api_reference/` |
| `/demonstrations` | `tasks/<Task>/demonstrations/` | Read-only assigned extracted H5; present only in `demo_only` and `demo_interaction` |
| `/feedback` | `tasks/<Task>/feedback/` | Read-only feedback from this task's counted training attempts; empty in `demo_only` |
| `/runtime` | `runtime/` | Read-only public CaM Python API |
| `/packages` | Allowlisted analysis packages from the host environment | Read-only NumPy, SciPy, Pillow, OpenCV, h5py, imageio, and their required libraries |
| `/usr`, `/bin`, `/lib`, `/lib64`, `/python` | System tools, libraries, and Python installation | Read-only runtime support |
| `/tools` | Sandbox-local interpreter aliases | `python`, `python3`, and `python3.11` select Python 3.11 |
| `/tmp`, `/home`, `/proc`, `/dev` | Private temporary directories and namespaced process/device views | No host home directory, credentials, or host process view |

`read_image` accepts allowed images in `/workspace`, `/feedback`, or assigned
`/demonstrations`. `rollout` uses only the next allowed training episode and is
denied at zero budget. Held-out evaluation requires a separate host `cam evaluate`
after freeze; neither `cam learn` nor a coding-agent tool can launch it.

**The RoboMME simulator source is not copied into the coding workspace.**
`inputs/benchmark/robomme/` serves the trusted simulator. `api_reference/` contains
only public policy/memory, sensor, robot RPC, and enabled demo-reader interfaces,
without task implementations, scoring logic, or hidden state.

Other host run paths are not mounted: `inputs/`, run manifests, `runner.py`,
`state/`, `program/`, `history/`, `checkpoints/`, `policy/`, `logs/`, `audit/`,
`reports/`, and `rollouts/`. This excludes seeds, test evidence, other runs, raw
datasets, and historical solutions. Only permitted training feedback is copied
to `/feedback`; the host records logs and snapshots.

## During policy execution

The policy gets frozen program files in read-only `/workspace`, public runtime,
and runtime support. Development scratch, `/demonstrations`, and `/feedback`
are absent. Only allowed observations and robot RPC cross from the simulator.
When `builtin_demo` is enabled, the task's reset video is an episode observation.
Execution makes zero model calls; declared memory resets each episode.

## Inspect actual command and network access

From the checkout, run `.venv/bin/python scripts/audit_run.py RUN --task TASK`.
It writes commands and an access summary under `orchestration/`, distinguishing
Bubblewrap host setup from sandbox activity. Inspect original `audit/call_*.json`
and `.strace` files for details. The scan covers Internet sockets and successful
absolute opens under checked private prefixes; relative paths and other syscall
forms may need manual review. Regenerate after development for full call coverage.
"""


def write_runs_readme(base):
    path = base / "README.md"
    if not path.exists():
        path.write_text("""# CaM runs

Each new experiment has its own task-specific leaf:

```text
runs/<benchmark>/<task>/<agent-model>/<setting>/<timetag>/
```

Settings are `demo_only`, `interaction_only`, and `demo_interaction`, corresponding
to modes `demos`, `interaction`, and `both`. Short folders use `cc` for Claude Code
and `cx` for Codex, e.g. `cc-opus5.5` or `cx-gpt6.1-sol`. Provider prefixes are
omitted from standard model names; variant suffixes are retained and unsafe
characters percent-encoded. Exact client/provider/model IDs remain in `run.json`.
Automatic timetags use UTC. `cam prepare runs --dataset DATA --config CONFIG`
creates one leaf per selected task and prints those paths. `--tasks` selects a
subset; `--timetag` sets an explicit tag. Pass each printed leaf to `cam learn`,
`evaluate`, `status`, and `report`. Existing leaves are never overwritten.

Each leaf's `README.md` documents its settings, commands, saved artifacts, and
the coding agent's filesystem permissions. Private simulator code under `inputs/`
is not mounted into coding tools. Existing legacy run paths remain usable.
""")


def write_run_readme(root, manifest):
    cfg = manifest["config"]
    coding = cfg["coding"]
    settings = {
        "demos": "demo_only",
        "interaction": "interaction_only",
        "both": "demo_interaction",
    }
    run = shlex.quote(str(root))
    budgets = "0" if cfg["mode"] == "demos" else str(cfg["budget"])
    tasks = ", ".join(manifest["tasks"])
    (root / "README.md").write_text(
        f"""# {manifest["config"]["benchmark"]} / {tasks}

| Setting | Value |
|---|---|
| Coding agent | `{coding["agent"]}` |
| Model | `{coding["model"]}` |
| Provider | `{coding["provider"]}` |
| Reasoning effort | `{coding["reasoning_effort"]}` |
| Learning setting | `{settings[cfg["mode"]]}` |
| Demonstrations per task | `{manifest["demonstrations_per_task"]}` |
| Maximum training simulator attempts | `{cfg["budget"]}` |
| Depth / task-provided reset video | `{cfg["depth"]}` / `{cfg["builtin_demo"]}` |
| Expected test episodes | `{sum(len(v) for v in manifest["test_episodes"].values())}` |

## Commands

```bash
cam learn {run} --gpus 0 --workers-per-gpu 1
cam evaluate {run} --budgets {budgets} --gpus 0 --workers-per-gpu 1
cam status {run}
cam report {run}
```

`learn` starts paid model sessions. Use `--resume` to continue an unfinished
session/queue with its retained evidence and budget. Change inputs by preparing
a new timetag; prepared source and metadata snapshots are pinned.

## Saved files

```text
README.md                         this run's settings and visibility boundaries
run.json, prepared.json           configuration and provenance
inputs/                           private pinned harness, benchmark, and metadata
runtime/cam/                      public policy API
queues/                           learning/evaluation queue state and process logs
tasks/<Task>/prompt.md             exact initial coding instructions
tasks/<Task>/workspace/            agent's current code, analysis, and scratch files
tasks/<Task>/logs/session_NNN/     transcript, prompt, errors, usage, completion
tasks/<Task>/audit/                tool calls, command output, archived file states
tasks/<Task>/history/              program revisions and diffs
tasks/<Task>/checkpoints/RNNN/     immutable checkpoints
tasks/<Task>/policy/               final frozen code solution and freeze metadata
tasks/<Task>/rollouts/train/       counted training attempts (if enabled)
tasks/<Task>/rollouts/test/        held-out H5, video, actions, decisions, and results
reports/                          aggregate JSON and CSV results
```

"""
        + HISTORY
        + ACCESS
    )
