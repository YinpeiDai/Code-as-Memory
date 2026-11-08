"""Human-readable run organization and the actual agent visibility boundary."""

import shlex

from cam.config import BOTH_PROTOCOLS, learning_setting


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

MIKASA_DRY_RUN = """## MIKASA robot motion preview

Inside an episode, `robot.dry_run(..., mode="absolute" or "delta")` checks the
next requested arm joint target against robot joint limits. It does not step
physics, expose task state, or check collisions. Absolute inputs become one
bounded command toward a world XYZ/RPY pose; delta inputs use `robot.step` units.
This is robot RPC inside an action routine, not an additional coding-agent tool.
Queries are capped at `max_action_steps` per action routine by the host.

The public reference is `api_reference/runtime/mikasa_robo/dry_run.md` in the
coding workspace. The live-query implementation stays private. Each rollout
records requests/results in `dry_runs.jsonl`;
only counted training attempts copy this log to `/feedback`. Queries consume
no native steps or extra attempts; their containing training rollout is counted.
Earlier prepared runs retain their original API; changing it requires a new timetag.

For simulator-free development, `from cam.runtime.mikasa_robo import safety_check`
provides a pure offline API on supplied robot state and optional actions. It
reports position margins, supplied torque versus URDF effort limits, and an
initial PD-demand estimate versus drive caps when velocities and a target are
supplied. Missing inputs produce unavailable torque feedback. The public module
`offline.py`, sanitized `panda_model.json`, and `safety_check.md` are mounted in
`runtime/cam/runtime/mikasa_robo/` and mirrored in the coding API reference.
They contain no task, object, scorer or simulator implementation. This API is
allowed in demo-only mode, uses no simulator or attempt, and its command output
is preserved in the normal audit. `robot.safety_check(...)` exposes the same pure
calculation to generated action routines.

"""

MIKASA_RGBD = """## MIKASA v2 observation interface

Development and held-out evaluation both render native 256x256 RGB-D from the
front (scene) and wrist cameras on every control step. `front_rgb`/`wrist_rgb`
are `uint8[256,256,3]`; aligned `front_depth`/`wrist_depth` are `int16[256,256]`
optical-axis millimetres (0 unknown). Camera intrinsics and world-to-camera
extrinsics accompany every frame. `depth: true` is required; CaM does not resize
or subsample these observations. The same interface is used by the assigned demos.

"""

ACCESS = """## What the coding agent can access

The host client authenticates to the provider. Model-accessible I/O uses only
`command`, `read_image`, `usage`, `snapshot`, `validate`, and `rollout`; built-in
shell/file/browser/delegation tools are disabled. Commands run in Bubblewrap with
private mount, PID, and network namespaces, without host credentials or network.

Before paid Claude inference, the host probes the pinned MCP server with
initialization, all six tool schemas and a read-only `usage` call. Evidence is
in `tasks/<Task>/logs/session_NNN/mcp_probe/`; this host check consumes no
training attempt. Client initialization must also expose exactly the six tools
and report CaM connected, or the session fails. Startup and isolated interface
validation allow up to 120 seconds for NFS imports without changing simulator
budgets or policy execution limits.

| Agent-visible path | Host source relative to this run | Access during development |
|---|---|---|
| `/workspace` | `tasks/<Task>/workspace/` | Read/write: `task.json`, `policy.md`, `tools/*.py`, analysis notes, `scratch/`, `PROTOCOL.md`, and `api_reference/` |
| `/demonstrations` | `tasks/<Task>/demonstrations/` | Read-only assigned extracted H5; present in demo-only and both demo-plus-interaction versions |
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

**The BENCHMARK simulator source is not copied into the coding workspace.**
`inputs/benchmark/PACKAGE/` serves the trusted simulator, run by the pinned
interpreter in `run.json` (`simulator_python`). `api_reference/` contains only
public policy/memory, sensor, robot RPC, and enabled demo-reader interfaces,
without task implementations, scoring logic, or hidden state.
Prompts include public sensor/action semantics, without scorer-derived task
contracts or success thresholds. The learner infers task conditions from its
assigned evidence and counted training feedback. Any user-authorized task
instruction correction is listed above and recorded in `run.json` and the prompt.
Large simulator assets may be linked under that private snapshot; their hashes
are recorded in `run.json`. Coding commands and policies use the pinned CaM
Python 3.11 sandbox even when the simulator needs its own Python environment.

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

Settings are `demo_only`, `interaction_only`, `demo_interaction_v1`, and
`demo_interaction_v2`. Mode `both` requires `both_protocol: "v1"` (free evidence
choice) or `"v2"` (demos first, interaction for verification). Both use the same
six-tool harness. Short folders use `cc` for Claude Code
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
    setting = learning_setting(cfg["mode"], cfg.get("both_protocol"))
    protocol = "Not applicable"
    if cfg["mode"] == "both":
        version = cfg.get("both_protocol")
        protocol = (
            f"{version}: {BOTH_PROTOCOLS[version]} (prompt instructions; shared harness)"
            if version else "Historical unversioned prompt"
        )
    run = shlex.quote(str(root))
    budgets = "0" if cfg["mode"] == "demos" else str(cfg["budget"])
    tasks = ", ".join(manifest["tasks"])
    if (
        cfg["mode"] == "both"
        and cfg.get("both_protocol") == "v2"
        and (cfg.get("training_seed_policy") == "disjoint_once" or manifest.get("benchmark_mode_guidance"))
    ):
        attempt_guidance = (
            "The v2 prompt requires using all remaining attempts on the assigned "
            "training seeds to verify and strengthen the solution, even when it seems ready "
            "for testing. Successful verification does not require a code change."
        )
    elif cfg["mode"] == "demos":
        attempt_guidance = "No training simulator attempts are allowed in demo-only learning."
    else:
        attempt_guidance = "The attempt limit is a ceiling. The agent may stop early."
    clarifications = "".join(
        f"{value['text'].rstrip()}\n\n"
        f"User-authorized correction for `{task}`; pinned source: "
        f"`inputs/harness/{value['source']}`. The exact text is also recorded "
        "in `run.json` under `task_clarifications` and in the learner prompt.\n\n"
        for task, value in manifest.get("task_clarifications", {}).items()
    )
    guidance = manifest.get("benchmark_mode_guidance")
    guidance_note = (
        "## Benchmark-specific development guidance\n\n"
        f"Pinned supplement: [{guidance['source']}](inputs/harness/{guidance['source']}). "
        "The exact text and source are recorded in `run.json` under "
        "`benchmark_mode_guidance` and included in each task's `prompt.md` and "
        "`workspace/PROTOCOL.md`. It supplements shared v2 with demo-first "
        "motion verification and the public robot safety APIs. It does not change "
        "the tools, attempt accounting or native scoring.\n\n"
        if guidance else ""
    )
    initial = manifest.get("initial_policy")
    initial_note = (
        "## User-requested code repair\n\n"
        f"Initialized from the same task's frozen policy in `{initial['source_run']}` "
        f"(program hash `{initial['program_hash']}`). The configuration, assigned demos "
        "and native benchmark snapshot match that source run. This is a separate repair "
        "follow-up, not a fresh-policy comparison or replacement of the original scored run. "
        "Only policy files and pre-evaluation policy notes were imported; no test feedback, "
        "rollouts, evaluator logs or other task policies were supplied. The imported files "
        "are pinned under `inputs/initial_policy/` and retained in the initial code revision. "
        "The learner prompt explicitly requests a code review and repair using offline evidence.\n\n"
        if initial else ""
    )
    (root / "README.md").write_text(
        f"""# {manifest["config"]["benchmark"]} / {tasks}

| Setting | Value |
|---|---|
| Coding agent | `{coding["agent"]}` |
| Model | `{coding["model"]}` |
| Provider | `{coding["provider"]}` |
| Reasoning effort | `{coding["reasoning_effort"]}` |
| Learning setting | `{setting}` |
| Demo + interaction protocol | {protocol} |
| Demonstrations per task | `{manifest["demonstrations_per_task"]}` |
| Maximum training simulator attempts | `{cfg["budget"]}` |
| Interaction seed schedule | `{cfg.get("training_seed_policy", "demo_cycle")}`; `{cfg.get("training_seed_count")}` identities |
| Depth / task-provided reset video | `{cfg["depth"]}` / `{cfg["builtin_demo"]}` |
| Physics backend | `{cfg.get("physics_backend") or "benchmark default"}` (independent of GPU rendering) |
| Expected test episodes | `{sum(len(v) for v in manifest["test_episodes"].values())}` |

{attempt_guidance} Logs and freeze metadata record the number actually used.
Ending learning freezes the final
validated policy without automatically starting held-out evaluation.
`disjoint_once` assigns distinct training seeds outside the demonstrations and
held-out tests, used once each in the pinned order. `demo_cycle` is the historical
schedule that cycles through selected demonstration identities; a budget equal
to the identity count makes one pass without repeats. The host enforces the next
identity and budget; `rollout` accepts no seed or episode override. With a host GPU pool,
rollouts wait for a free simulator slot before consuming an attempt; coding and
offline analysis continue independently in other sessions.

{clarifications}{guidance_note}{initial_note}## Commands

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
        + (MIKASA_RGBD + (
            "CPU physics is an explicit comparison variant, not the official GPU evaluation. "
            "The controller uses native CPU IK; rendering still uses the GPU. "
            "Keep CPU and GPU scores separate.\n\n"
            if cfg.get("physics_backend") == "cpu" else ""
        ) if cfg["benchmark"] == "mikasa_robo_v2" else "")
        + (MIKASA_DRY_RUN if cfg["benchmark"] in ("mikasa_robo", "mikasa_robo_v2") else "")
        + ACCESS.replace("BENCHMARK", manifest["config"]["benchmark"]).replace(
            "PACKAGE", manifest["benchmark_package"]
        )
    )
