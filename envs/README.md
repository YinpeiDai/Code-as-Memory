# Benchmark environments

The root `pyproject.toml` installs the common Python 3.11 harness and policy
runtime. Native simulators are independent of that environment:

| Benchmark | Python | Setup |
|---|---|---|
| [RoboMME](robomme/README.md) | 3.11 | `uv sync --locked --project envs/robomme --python 3.11` |
| [MIKASA-Robo](mikasa_robo/README.md) | 3.11 | `bash envs/mikasa_robo/install.sh` |
| [RMBench](rmbench/README.md) | 3.10 | `bash envs/rmbench/install.sh` |
| [RoboMemArena](robomemarena/README.md) | 3.10 | `uv sync --locked --project envs/robomemarena --python 3.10` |

Run setup from the repository root. Every environment lives at
`envs/<benchmark>/.venv`; `CAM_<BENCHMARK>_PYTHON` can select a matching external
interpreter. `cam prepare` pins dependency versions and build recipes in each
run. Never install incompatible simulator stacks into the common environment.

On an existing research checkout, use `uv sync --locked --inexact` for the root
environment to retain packages needed by older pinned runs. A fresh checkout
installs only the common dependencies; simulator dependencies live under `envs/`.
