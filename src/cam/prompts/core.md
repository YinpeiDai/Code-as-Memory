# Code-as-Memory

Learn a general Python program for the assigned task. The program is persistent
task knowledge; its declared memory stores observations and progress within one
episode. At deployment, Python performs perception, reasoning and control with
no model calls. Each episode starts with empty memory.

Use only the assigned demonstrations and/or counted training interactions in
the assignment. Work through the CaM tools. Keep analysis in `task_analysis.md`,
checks in `scratch/`, and changes with their evidence in `CHANGELOG.md`. Save a
snapshot after each complete iteration. Never use test feedback, other policies,
simulator source, hidden object state or unassigned data.

Infer reusable rules from current observations and instructions. Do not replay
demonstrations or encode seed-to-solution lookups. Separate observations from
assumptions; check perception, goal parsing and control with the evidence available.
Record what was checked and what remains uncertain. Offline checks do not prove
physical success. Finish with a validated program and a concise evidence summary.

Write `task.json`, `policy.md`, and `tools/*.py`. Declare memory fields in
`task.json` with `runtime="python"`. Define
`perceive_decide(obs, goal, memory, last)` returning exactly
`{"memory": {"every_declared_field": value}, "act": {"tool": "act_name", "args": {...}}}`.
An `act_*` function takes `robot` first and returns measured outcomes. Each call
has a fresh namespace; retain history through declared memory. See the supplied
API reference and benchmark guide for observations, actions and limits.

The supplied `api_reference/` mirrors the public Python package: `core/` defines
memory declarations and policy execution; `runtime/` provides observation, robot
and demonstration interfaces. Read `api_reference/core/workspace.py` for memory
types; `core/code_policy.py`, `core/tools.py` and `core/sandbox.py` under that same
reference directory define policy execution and supported Python. The benchmark
guide identifies the relevant `runtime/` files.

Generated tools have no imports. `np`, `math` and safe builtins are provided.
Private helpers start with `_`; public functions start with `perceive_` or `act_`.
Do not catch terminal episode exceptions. Preserve every intermediate log and
rollout. The driver freezes code and evaluates it on held-out episodes separately.
