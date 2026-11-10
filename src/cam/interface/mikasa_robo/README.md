# MIKASA-Robo interface

This package connects the trusted harness to MIKASA-Robo. It declares the public
sensors and actions, selects pinned episode identities, and translates simulator
steps into policy observations. Native scoring remains in the official simulator.

Training and evaluation default to CPU physics. `ShellGameTouch-VLA-v0` always
uses GPU physics, including when a shared configuration requests CPU. Other tasks
accept an explicit GPU override for comparisons. The backend also selects native
CPU/GPU IK; rendering always uses the GPU. New preparations record the effective
backend per task in `run.json`, and episode results record the actual backend.

- `adapter.py`: environment lifecycle, sensor/action interface, scoring boundary.
- `data.py`: demonstration assignment and format handling.
- `guide.md`: public task-independent guidance composed into learner prompts.
- `api/`: sanitized offline robot helpers and public API reference.

Install [envs/mikasa_robo](../../../../envs/mikasa_robo/README.md), then follow the
[MIKASA-Robo commands](../../../../docs/benchmarks/mikasa_robo.md).
The [shared boundary](../base.py) validates every public frame and action.
Simulator imports must stay inside native environment methods. Only explicitly
published helpers may be copied into the isolated policy runtime.
