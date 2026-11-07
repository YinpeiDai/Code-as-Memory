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

The learner receives `workspace/api_reference/{core,runtime}/`, containing only
public policy, memory, observation, action, and trajectory interfaces. Six CaM
tools provide model-accessible I/O. Commands run inside Bubblewrap; generated
policies use a separate read-only sandbox without demonstrations or feedback.
See the [mount table and evaluation boundary](reproducing.md#learner-access).

The trusted simulator owns seeds, action limits, native scoring, and the attempt
ledger. Allowed observations cross a bounded JSON/array transport; reset videos
stream frame by frame. Validation never imports generated code into the host.

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
3. Implement `benchmarks/base.py`: sensor translation, actions, native counters,
   and terminal scoring. Different sensors need an explicit schema version.
4. Add a benchmark guide and provenance. Route every training reset through the
   shared ledger and keep test evidence outside coding tools.
5. Test extraction, isolation, timeouts, terminal semantics, and a native smoke
   episode before marking the adapter runnable.
6. Emit `cam-episode/1` or a documented successor for shared reporting.

Model-based runtime perception is not implemented. A future adapter must record
its calls, model, latency, and costs separately from zero-API Python policies.
