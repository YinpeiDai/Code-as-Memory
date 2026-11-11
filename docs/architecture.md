# Architecture

A coding agent builds a robot memory program. A trusted host owns the simulator,
identity schedule, budget, freeze, and evaluation. The program receives public
observations and sends actions through a narrow interface.

```text
Prepared demonstrations → isolated coding tools → Python program + declared memory
                                                   ↓ freeze
Benchmark environment ↔ trusted simulator interface ↔ isolated policy runtime
                                                   ↓
                                      results, logs, directory-backed website
```

| Component | Responsibility |
|---|---|
| `interface/<benchmark>/` | Native simulator adapter, public observation/action contract, task metadata, demo format |
| `harness/` | Immutable preparation, tool access, process isolation, scheduling, recording, recovery |
| `core/` | Policy validation, memory declarations, history, checkpoints, attempt ledger |
| `runtime/` | Observation transport, robot RPC, trajectory access, Python policy worker |
| `providers/` | Coding-client integration and development usage accounting |
| `prompts/` | Shared core/modes and explicit benchmark guidance |
| `reporting/` · `web/` | Reports, portable directory index, live dashboard, galleries |

The four interface packages are `robomme`, `mikasa_robo`, `rmbench`, and
`robomemarena`. `interface/base.py` defines the common boundary;
`interface/registry.py` locates adapters and simulator interpreters. Importing an
adapter must not import or create its simulator. Native imports belong inside
simulator methods, in the benchmark’s dedicated environment.
See [simulator interface adaptations](interface.md) for what each wrapper
preserves and changes relative to the original benchmark.

The root environment runs the host and isolated policies. `envs/<benchmark>/.venv`
runs only that benchmark’s trusted simulator broker. Preparation records both
interpreters and package sets, copies the code and build recipes, and pins asset
hashes. Public robot math is selectively copied to the policy runtime; private
simulator packages never enter the coding sandbox.

Configurations remain necessary: client settings and experiment plans describe
reproducible runs; solution catalogs contain required memory schemas, code hashes,
interface hashes, and provenance. Executable solutions contain Python only, with
one README per setting.

To add an interface, implement the protocol in `interface/base.py`, declare exact
public sensors/actions, supply a data protocol and dedicated environment, register
it, and add transport/isolation tests. No scorer-derived task hints belong in
learner prompts. See [reproducing](reproducing.md) for experiment boundaries.
