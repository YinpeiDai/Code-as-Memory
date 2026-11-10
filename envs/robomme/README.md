# RoboMME environment

```bash
git submodule update --init third_party/robomme_benchmark
uv sync --locked --project envs/robomme --python 3.11
envs/robomme/.venv/bin/python -I -c 'import robomme, mani_skill, sapien, torch, h5py'
```

This lock retains the pinned RoboMME/ManiSkill source, PyTorch 2.9.1, and SAPIEN
3.0.3 from the former root simulator extra. The common root environment now
contains only the harness and policy libraries. RoboMME tasks use assets shipped
with the pinned benchmark and simulator; NVIDIA/Vulkan rendering must work.

`CAM_ROBOMME_PYTHON` overrides the interpreter. The host pins it and the package
snapshot into every new run. Existing prepared runs retain their original
interpreter and snapshots.

[Run RoboMME](../../docs/benchmarks/robomme/README.md) ·
[Simulator interface](../../src/cam/interface/robomme/README.md)
