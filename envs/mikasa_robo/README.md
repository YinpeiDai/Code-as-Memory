# MIKASA-Robo simulator environment

CaM runs the MIKASA-Robo-VLA simulator and its trusted broker in a dedicated uv
environment, `envs/mikasa_robo/.venv`, shared by `mikasa_robo` and
`mikasa_robo_v2`. `CAM_MIKASA_ROBO_PYTHON` and `CAM_MIKASA_ROBO_V2_PYTHON`
override their respective interpreters. Coding commands and
generated policies still use the CaM environment's Python 3.11 sandbox.

For the 50-demo RGB-D experiment on another machine, follow the
[MIKASA-Robo v2 setup guide](../../docs/mikasa_v2_setup.md).

## Install

From the CaM checkout, with the `third_party/mikasa_robo` submodule at
`509b875f3d207c287497c0a897661062de928bb0` (its `uv.lock` SHA-256 is
`afabd6c0…77ec8`):

```bash
git submodule update --init third_party/mikasa_robo
UV_PROJECT_ENVIRONMENT=$PWD/envs/mikasa_robo/.venv \
  uv sync --frozen --no-dev --project third_party/mikasa_robo --python 3.11
envs/mikasa_robo/.venv/bin/python -I -c "import numpy, h5py, imageio, imageio_ffmpeg, mani_skill, torch"
```

Verified with uv 0.6.2: Python 3.11.11, mani_skill 3.0.0b15, sapien 3.0.0b1,
torch 2.2.1+cu121, numpy 1.23.5, gymnasium 0.29.1, h5py 3.16.0, imageio 2.37.3,
imageio-ffmpeg 0.6.0. The lock installs `mikasa_robo_suite` in editable mode;
the adapter instead imports the pinned copy under each run's `inputs/benchmark/`
and rejects any other import. Rendering needs an NVIDIA GPU with Vulkan; the
official protocol uses GPU simulation (`sim_backend="gpu"`, `num_envs=1`).

## Assets

The seven ShellGame tasks load ManiSkill YCB cups. mani_skill 3.0.0b15 pins
SHA-256 `174001ba…02262fb` for `mani_skill2_ycb.zip`, but the Hugging Face
`main` file changed in March 2025, so `python -m mani_skill.utils.download_asset ycb`
fails its checksum. Fetch the matching archive (75 MB extracted) from
`haosulab/ManiSkill2@4c134e2e33b11b1fa751c1d7b6afb2e89231b875`:

```bash
MS_ASSET_DIR=$PWD/data/raw/mikasa_robo/maniskill_data \
  envs/mikasa_robo/.venv/bin/python -I scripts/download_mikasa_robo_assets.py
```

`cam prepare` links this directory into the pinned benchmark as
`maniskill_data` and records its tree hash; the adapter sets `MS_ASSET_DIR` to
the link before importing `mani_skill`. `CAM_MIKASA_ROBO_ASSETS` overrides the
default location. Other tasks need no downloaded assets.

## Demonstrations

The raw data is the lossless RLDS release `mikasa-robo/mikasa-robo-vla-rlds` at
revision `8575624b7e054fc1bceccedc7a8c1b533b4f15d2` (307 GB, 90 task folders),
downloaded with `hf download ... --revision <rev> --local-dir data/raw/mikasa_robo/rlds`.
Do not use the LeRobot copy (AV1, lossy). Then:

```bash
.venv/bin/cam extract --benchmark mikasa_robo --raw-data data/raw/mikasa_robo/rlds \
  --output data/mikasa_robo_25 --count 25 --rgb-only
```

Extraction verifies every file against `src/cam/benchmarks/metadata/mikasa_robo/release.json`
(LFS SHA-256 / git blob SHA-1) and every episode identity against the pinned
training metadata. `scripts/pin_mikasa_robo_metadata.py` (run in this
environment) regenerates that metadata from the release and the simulator registry.

Current learner prompts contain public sensor/action documentation and the
native task goal, without source-derived task contracts or scorer thresholds.
Historical contract metadata remain host-only. The user-authorized
SeqOfColors3 cue-order clarification is recorded in each newly prepared run.

## Checks and measured timing

```bash
.venv/bin/pytest -q tests/test_mikasa_robo.py
CUDA_VISIBLE_DEVICES=0 envs/mikasa_robo/.venv/bin/python -I scripts/smoke_mikasa_robo.py
```

Measured on one RTX 6000 Ada (GPU 0), 2026-11-07:

- Raw canonical env, zero commands, all 90 tasks
  (`outputs/mikasa_robo/sweep/`): 21–62 ms per step (slowest single step
  0.2 s), env creation 2.3–12.7 s, first reset ≤ 2.9 s; the same seed gives an
  identical first frame after resetting another seed.
- Through `broker.execute` with a scripted policy (sandboxed worker, capture,
  video): BatteriesCheckerHard-6-VLA-v0, the longest horizon (2160 steps),
  took 149.5 s (69 ms per step including env creation); 25–30-step tasks take
  7–28 s, mostly env creation and worker start-up.
- `policy_timeout` = max(1800 s CaM default, 2 × 150 s) = 1800 s.
  `max_action_steps` 2200 ≥ 2160, so CaM never truncates before the native
  limit; `max_decisions` is the CaM-wide 40.
