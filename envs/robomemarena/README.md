# RoboMemArena environment

```bash
git submodule update --init third_party/robomemarena
uv sync --locked --project envs/robomemarena --python 3.10
envs/robomemarena/.venv/bin/python -I -c 'import robosuite, mujoco, bddl, h5py'
```

The lock pins the official LIBERO fork’s stack: MuJoCo 3.3.7, robosuite 1.4.0,
and CPU PyTorch 2.4.1. The adapter imports the fork and official scripts from the
pinned benchmark. Its LIBERO assets are linked into prepared runs and hashed.
EGL rendering follows the assigned GPU; no global LIBERO configuration is edited.

`CAM_ROBOMEMARENA_PYTHON` overrides the interpreter. Policies continue to execute
in the separate common Python 3.11 environment.

[Run RoboMemArena](../../docs/benchmarks/robomemarena/README.md) ·
[Simulator interface](../../src/cam/interface/robomemarena/README.md)
