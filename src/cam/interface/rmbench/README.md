# RMBench interface

This package connects the trusted harness to RMBench. It declares the public
sensors and actions, selects pinned episode identities, and translates simulator
steps into policy observations. Native scoring remains in the official simulator.

- `adapter.py`: environment lifecycle, sensor/action interface, scoring boundary.
- `data.py`: demonstration assignment and format handling.
- `guide.md`: public task-independent guidance composed into learner prompts.
- `api/`: sanitized offline robot helpers and public API reference.

Install [envs/rmbench](../../../../envs/rmbench/README.md), then follow the
[RMBench commands](../../../../docs/benchmarks/rmbench/README.md).
The [shared boundary](../base.py) validates every public frame and action.
Simulator imports must stay inside native environment methods. Only explicitly
published helpers may be copied into the isolated policy runtime.
