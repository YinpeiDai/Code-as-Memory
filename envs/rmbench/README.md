# RMBench environment

The trusted simulator uses Python 3.10, SAPIEN 3.0.0b1, PyTorch 2.4.1+cu121 and
CuRobo v0.7.8. The shared harness and isolated policy runtime use root Python 3.11.

```bash
git submodule update --init third_party/rmbench
bash envs/rmbench/install.sh
.venv/bin/cam doctor --benchmark rmbench
```

The installer syncs the lockfile with `--inexact`, applies the upstream SAPIEN/MPLib
patches after checking expected hashes, builds CuRobo, and downloads the pinned
robot/object assets into `envs/rmbench/assets`. Keep `--inexact` on subsequent syncs
so the editable CuRobo build remains installed. Installation evidence is recorded
in `install_record.txt`.

Set `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` for your installed CUDA toolkit and GPU
(the local defaults are CUDA 12.8 and Ada 8.9). Asset download overrides are
`RMBENCH_ASSETS` and `RMBENCH_ASSET_REVISION`; runtime overrides are
`CAM_RMBENCH_ASSETS` and `CAM_RMBENCH_PYTHON`.

The interface uses the official end-effector action mode; its results are not
interchangeable with joint-action baselines. CuRobo warm-up and ray tracing require
GPU memory, and identical rollouts can differ slightly through native numerics.
Preparation pins simulator sources and links assets. No demo replay or seed search
is required to run a published solution or develop from downloaded demonstrations.

Continue with the [RMBench guide](../../docs/benchmarks/rmbench.md) or
[public interface](../../src/cam/interface/rmbench/README.md).
