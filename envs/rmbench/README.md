# RMBench simulator environment

Pinned Python 3.10 environment for `cam.benchmarks.rmbench` (RoboTwin 2.0,
SAPIEN 3.0.0b1 ray tracing, CuRobo v0.7.8). It runs the trusted CaM broker and
the native simulator; generated policies still run in the CaM 3.11 sandbox.

```bash
bash envs/rmbench/install.sh                  # idempotent; log: envs/rmbench/install_record.txt
CUDA_VISIBLE_DEVICES=1 envs/rmbench/.venv/bin/python scripts/smoke_rmbench.py \
    --source <pinned rmbench copy with assets/> --output outputs/rmbench/smoke/<tag>
```

`install.sh` does what `third_party/rmbench/scripts/_install.sh` and
`_download_assets.sh` do, without conda, pytorch3d or XPolicyLab:

1. `uv sync --locked --inexact` from `pyproject.toml`/`uv.lock` (torch 2.4.1+cu121,
   sapien 3.0.0b1, mplib 0.2.1, numpy 1.26.4, CuRobo's Python requirements).
   `--inexact` keeps the editable CuRobo build; plain `uv sync` would remove it.
2. The two upstream `sed` patches, applied only to the expected file hashes:
   `sapien/wrapper/urdf_loader.py` (UTF-8 reads)
   `eabba86f…0683 -> 13aad693…8059`; `mplib/planner.py` (screw plan ignores
   collisions) `785f7723…fa0e -> 23c51d08…cf3c`.
3. CuRobo `v0.7.8` (`d64c4b00…`) cloned into `envs/rmbench/curobo` and built with
   `uv pip install --no-deps --no-build-isolation -e`, `CUDA_HOME=/usr/local/cuda-12.8`,
   `TORCH_CUDA_ARCH_LIST=8.9` (RTX 6000 Ada). CuRobo is required even for joint
   actions: each episode starts one planner subprocess per arm.
4. Assets `embodiments/**` and `objects/**` (~1.3 GB) from the
   `TianxingChen/RMBench` dataset at revision `5d8ff210…` into
   `envs/rmbench/assets`, then `${ASSETS_PATH}` in the embodiment `*_tmp.yml`
   templates is expanded to `envs/rmbench` (absolute CuRobo URDF/mesh paths),
   as `scripts/update_embodiment_config_path.py` does.

Overrides: `RMBENCH_ASSETS`, `RMBENCH_ASSET_REVISION`, `CUDA_HOME`,
`TORCH_CUDA_ARCH_LIST` for the installer; `CAM_RMBENCH_ASSETS` for the adapter
and `CAM_RMBENCH_PYTHON` for the broker interpreter.

`cam prepare` copies `third_party/rmbench` into the run without `.git`,
`XPolicyLab/`, `assets/` and raw data, and links `assets -> envs/rmbench/assets`
(its tree hash is recorded). The simulator opens `./assets/...` relative to the
working directory, so the adapter changes into the pinned copy before import.
Native output is verbose; redirect it to log files.

Raw data are not part of this environment: the released `demo_clean.zip`
archives live in `data/raw/rmbench/hf/data/<task>/` and their extracted copies in
`data/raw/rmbench/demo_clean/`. `scripts/rmbench_train_metadata.py` pins the
training seeds from the zips; `scripts/rmbench_seed_search.py` reproduces the
official evaluation-seed search.

For derived RGB-D training demos, use CaM's Python to run
`scripts/prepare_rmbench_rgbd.py start --gpus 0 1 --workers 10 --allow-gpu-sharing`
for the user-requested five workers per GPU (ten total). Sharing is explicit and
checks free memory; omitting the flag waits for idle GPUs. It uses this simulator
interpreter and regenerates the native expert on
the released training seeds because the original dense paths were not released.
Successful-only H5s contain synchronous four-camera 320x240 RGB-D (int16 depth in
millimetres) and public robot/calibration fields. Originals and historical runs
are preserved. The generated job README and manifest record actual counts and
failure shortfalls. `depth: true` enables the matching online sensors for fresh
runs; end-effector control and native success scoring are unchanged.

## Protocol notes (not shown to the learner)

- **Action mode.** CaM exposes only RMBench's official `ee` mode of
  `take_action` (16-D: per arm a world pose `[x,y,z,qw,qx,qy,qz]` and a gripper
  opening; CuRobo plans each arm). Mem-0 and most published RMBench baselines use
  the `qpos`/joint mode, so CaM success rates are not directly comparable to
  joint-action numbers. One `robot.step` is one `take_action`; episodes end at
  native `eval_success` or at the `_eval_step_limit.yml` limit (`timeout`).
- **Failures.** As in `eval_remote_policy`, an action counts only once
  `take_action` returns. A crash before that on the first action (or during
  environment creation, setup or the first observation) is an unscored
  `harness_error`, retried on the same pinned seed with `cam recover-test`
  (the official loop moves to the next seed instead). Any later crash is a
  scored native `error`.
- **Test seeds.** `scripts/rmbench_seed_search.py` reproduces the official
  search (start 100000; keep stable seeds whose scripted expert plans and
  succeeds). We pin our first run; every skipped seed and its reason, the GPU,
  driver and package versions are in each test metadata `provenance`.
  The expert check is not bitwise reproducible for every task (CuRobo/PhysX
  numerics): `provenance.agreement` records, per task, agreement with an
  independent repeat run and with Mem-0's published `eval_log.txt` seeds
  (`scripts/rmbench_seed_agreement.py`). Episodes are deterministic at reset
  (first frame bitwise identical across fresh processes) but `ee` rollouts are
  not bitwise reproducible: two identical scripted 250-action runs diverged from
  the first planned move, by at most 3.4 mm end-effector position and 0.005 rad
  joint target (CuRobo GPU planning), so repeated evaluations can differ slightly.
- **Training seeds.** Episode i of a released `demo_clean.zip` has seed
  `seed.txt[i]` in file order (not sorted). Demonstrations are 25 per task by
  `cam.data.demonstrations.random_select`.
- **Measured cost** (RTX 6000 Ada, GPU shared with 4–5 seed-search jobs):
  ~30–36 s from process start to the first action (two CuRobo planner warm-ups,
  2,500-step stability check, first ray-traced frames), then 0.17–0.2 s per
  `ee` action on an idle GPU; under load 0.51–0.58 s mean, p99 0.81–0.92 s and
  max 1.14 s after the first action (the first takes up to 2.2 s), for 4–15 cm
  moves (`scripts/smoke_rmbench.py`). The longest task (`blocks_ranking_try`,
  3,500 actions) needs ~2,000 s at the loaded mean and ~4,000 s at the observed
  per-action maximum; `policy_timeout` 7200 s = max(1800, ≈2× the loaded worst
  case of ~3,500 s at 1 s/action). Episode observations take about 0.2 MB per
  frame (four gzip RGB cameras plus calibration).
