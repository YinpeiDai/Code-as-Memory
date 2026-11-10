# MIKASA-Robo environment

```bash
git submodule update --init third_party/mikasa_robo
bash envs/mikasa_robo/install.sh
envs/mikasa_robo/.venv/bin/python -I -c 'import mani_skill, sapien, torch, h5py'

# Required YCB assets, including cups used by ShellGame tasks.
MS_ASSET_DIR="$PWD/envs/mikasa_robo/assets" \
  envs/mikasa_robo/.venv/bin/python -I scripts/download_mikasa_robo_assets.py
```

The installer uses the pinned upstream `pyproject.toml` and `uv.lock`, with the
environment stored here. It pins ManiSkill 3.0.0b15, SAPIEN 3.0.0b1, and PyTorch
2.2.1. Standard runs use CPU physics except `ShellGameTouch-VLA-v0`, which uses
GPU physics. All tasks use GPU rendering.

`CAM_MIKASA_ROBO_PYTHON` overrides the interpreter;
`CAM_MIKASA_ROBO_ASSETS` overrides `envs/mikasa_robo/assets`. Preparation preserves
the upstream build files, package versions, and asset hashes.

[Run MIKASA-Robo](../../docs/benchmarks/mikasa_robo/README.md) ·
[Simulator interface](../../src/cam/interface/mikasa_robo/README.md)
