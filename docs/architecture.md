# Architecture

The generated Python program is persistent task memory. Declared memory fields
track progress within an episode and reset between episodes. Deployed perception,
decisions, and control use no model calls.

All paths below are relative to `src/cam/`.

| Layer | Responsibility | Entry points |
|---|---|---|
| Core | Policy interface, bounded execution, attempt ledger, revisions, checkpoints | `core/` |
| Coding client | Client launch, tool restrictions, transcript and usage parsing | `providers/` |
| Trusted harness | Pinned preparation, tool server, simulator broker, queues | `harness/` |
| Benchmark | Data validation, native simulator, observations/actions/scoring | `benchmarks/` |
| Public runtime | Sensor objects, robot RPC, demonstration reader | `runtime/` |
| Reporting | Episode statistics, website, figures | `reporting/`, `web/` |

## Prompts and isolation

The task prompt combines `prompts/core.md`, a learning-mode file, and a benchmark
guide. Counts, IDs, limits, and sensors come from validated manifests. Both
clients receive the same task prompt; their system framing and tool schemas are
client-specific and recorded. Historical prompts are documentation only.

Mode `both` requires `both_protocol: "v1"` or `"v2"`, selecting
`prompts/modes/both_v1.md` or `both_v2.md`. V1 permits free evidence choice; v2
instructs demo-first development, resolves uncertainty through rollouts, and uses
remaining distinct training seeds to verify and strengthen the solution. Both share
the same six tools and checkpoint behavior; v2 has no extra candidate system or
enforced phase gate. See [protocol details](demo_interaction.md).

The learner receives `workspace/api_reference/{core,runtime}/`, containing only
public policy, memory, observation, action, and trajectory interfaces. Six CaM
tools provide model-accessible I/O. Commands run inside Bubblewrap; generated
policies use a separate read-only sandbox without demonstrations or feedback.
See the [mount table and evaluation boundary](reproducing.md#learner-access).

The trusted simulator owns seeds, action limits, native scoring, and the attempt
ledger. Allowed observations cross a bounded JSON/array transport; reset videos
stream frame by frame. Validation never imports generated code into the host.
Each adapter declares sensor names, shapes, units, and action bounds; those
declarations also generate the prompt and validate captured frames. Dedicated
simulator environments may use Python 3.10, while coding commands and policy
workers retain CaM's Python 3.11 sandbox. See [benchmark setup](benchmarks.md).

Claude Code uses bare mode, disabled built-ins, and one explicit MCP configuration.
Codex uses an empty client directory, ignores user rules/config, disables optional
tools and skills, and uses a sanitized model catalog. Before paid Codex learning,
the actual CLI calls CaM's MCP `usage` through a local fake model endpoint. The
gate rejects unexpected tools without calling OpenAI. It was checked with
Codex 0.166.0; compatibility remains version-sensitive. An unknown model requires
an explicit valid `coding.model_catalog`, not a fallback to another model.

## Adding a benchmark

1. Pin its official implementation and register its status.
2. Implement raw identity/split validation and extraction, including allowed
   fields, units, missing values, and action alignment.
3. Implement the `benchmarks/base.py` protocol. `interface(task, depth)` declares
   every allowlisted sensor (dtype, shape, meaning), the action vector (names,
   bounds, ignored components), video tiles and the starter step; the broker
   rejects any frame or action outside it and the prompt's interface table is
   generated from it. Implement native counters and terminal scoring, and keep
   the adapter importable without its simulator (import it only in
   `make_environment`, from the pinned copy). Register a dedicated
   `envs/<name>/.venv` when its stack conflicts with CaM's; policies always run
   in CaM's Python 3.11 sandbox. Declare bulky trees in `pin_ignore` and link
   assets via `asset_links` (hashed, not copied).
4. Add a short benchmark guide for the public observation/action interface,
   without scorer-derived task contracts, and pinned train/test metadata. Write
   demonstrations as `cam-demonstrations/2` via `cam.data.demonstrations`.
   Route every training reset through the shared ledger and keep test evidence
   outside coding tools.
5. Test extraction, isolation, timeouts, terminal semantics, and a native smoke
   episode before marking the adapter runnable.
6. Emit `cam-episode/1` or a documented successor for shared reporting.

Model-based runtime perception is not implemented. A future adapter must record
its calls, model, latency, and costs separately from zero-API Python policies.
