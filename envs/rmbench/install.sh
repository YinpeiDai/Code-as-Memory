#!/usr/bin/env bash
# Build the pinned RMBench simulator environment for CaM (idempotent).
#
#   bash envs/rmbench/install.sh            # env + patches + CuRobo + assets
#   SKIP_ASSETS=1 bash envs/rmbench/install.sh
#
# Mirrors third_party/rmbench/scripts/_install.sh and scripts/_download_assets.sh
# without conda, pytorch3d or XPolicyLab (CaM uses neither the point-cloud sampler
# nor the XPolicyLab policy server). Everything stays under envs/rmbench/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PY="${HERE}/.venv/bin/python"
RECORD="${HERE}/install_record.txt"
CUROBO_TAG="v0.7.8"
CUROBO_COMMIT="d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
ASSET_REPO="TianxingChen/RMBench"
ASSET_REVISION="${RMBENCH_ASSET_REVISION:-5d8ff21038fba32ea549e9eee8497aaba9aab5cf}"
ASSETS="${RMBENCH_ASSETS:-${HERE}/assets}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PATH="${CUDA_HOME}/bin:${PATH}"

log() { echo "[rmbench-install] $*"; echo "$(date -u +%FT%TZ) $*" >> "${RECORD}"; }

# 1. Locked Python 3.10 environment. --inexact keeps the editable CuRobo build.
uv sync --locked --inexact --project "${HERE}" --python 3.10
log "uv sync --locked ($(sha256sum "${HERE}/uv.lock" | cut -c1-64) uv.lock)"

# 2. The two source patches from scripts/_install.sh, verified by hash.
SITE="$("${PY}" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
patch_file() {  # path before_sha after_sha sed_expression
    local path="$1" before="$2" after="$3" expr="$4" now
    now="$(sha256sum "${path}" | cut -c1-64)"
    if [[ "${now}" == "${after}" ]]; then
        log "already patched ${path#"${SITE}/"} ${after}"
        return
    fi
    if [[ "${now}" != "${before}" ]]; then
        echo "Unexpected ${path} sha256 ${now}; refusing to patch" >&2
        exit 1
    fi
    sed -i -E "${expr}" "${path}"
    now="$(sha256sum "${path}" | cut -c1-64)"
    [[ "${now}" == "${after}" ]] || { echo "Patch of ${path} produced ${now}" >&2; exit 1; }
    log "patched ${path#"${SITE}/"} ${before} -> ${after}"
}
patch_file "${SITE}/sapien/wrapper/urdf_loader.py" \
    eabba86fe7d244ce8d469010f96914010a8883ea5835c86a8bc58be310510683 \
    13aad693f05663472a4f1b59ea34fb313db6ee2f1416fce19ad4830cfa588059 \
    's/("r")(\))( as)/\1, encoding="utf-8") as/g'
patch_file "${SITE}/mplib/planner.py" \
    785f77233baf4159f3327b664fae2507572c189223a6075f42546c1133ddfa0e \
    23c51d083ce1892e5fbd0e1473df432d1c33609e50bfac8bdf9e8bfcf6c2cf3c \
    's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g'

# 3. CuRobo (mandatory: RMBench builds two CuRobo planner subprocesses per episode).
if [[ ! -d "${HERE}/curobo/.git" ]]; then
    git clone --branch "${CUROBO_TAG}" --depth 1 https://github.com/NVlabs/curobo.git "${HERE}/curobo"
fi
[[ "$(git -C "${HERE}/curobo" rev-parse HEAD)" == "${CUROBO_COMMIT}" ]] || {
    echo "CuRobo checkout is not ${CUROBO_TAG} (${CUROBO_COMMIT})" >&2; exit 1; }
git -C "${HERE}/curobo" diff --quiet
git -C "${HERE}/curobo" diff --cached --quiet
if ! (cd /tmp && "${PY}" -I -c 'import torch, curobo.curobolib.kinematics_fused_cu' 2>/dev/null); then
    (cd "${HERE}/curobo" && uv pip install --python "${PY}" --no-deps --no-build-isolation -e .)
fi
(cd /tmp && "${PY}" -I -c 'import torch, curobo, curobo.curobolib.kinematics_fused_cu; print("curobo", curobo.__version__)')
log "curobo ${CUROBO_TAG} ${CUROBO_COMMIT} CUDA_HOME=${CUDA_HOME} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# 4. Simulator assets (embodiments + objects, ~1.4 GB) at a pinned HF revision.
if [[ "${SKIP_ASSETS:-0}" != 1 ]]; then
    mkdir -p "${ASSETS}"
    (cd /tmp && "${PY}" -I - "${ASSET_REPO}" "${ASSET_REVISION}" "${ASSETS}" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, revision, target = sys.argv[1:]
snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                  allow_patterns=["embodiments/**", "objects/**"], local_dir=target)
EOF
    )
    # scripts/update_embodiment_config_path.py: expand ${ASSETS_PATH} in *_tmp.yml to the
    # directory that contains assets/ (absolute CuRobo URDF/mesh paths).
    (cd /tmp && "${PY}" -I - "${ASSETS}" <<'EOF'
import sys
from pathlib import Path
assets = Path(sys.argv[1]).resolve()
base = str(assets.parent)
if assets.name != "assets":
    raise SystemExit("Asset directory must be named assets/ (RMBench joins <base>/assets/...)")
count = 0
for template in sorted((assets / "embodiments").rglob("*_tmp.yml")):
    text = template.read_text()
    template.with_name(template.name.replace("_tmp.yml", ".yml")).write_text(
        text.replace("${ASSETS_PATH}", base).replace("$ASSETS_PATH", base))
    count += 1
if not count:
    raise SystemExit("No embodiment *_tmp.yml templates found")
print(f"expanded {count} embodiment templates with ASSETS_PATH={base}")
EOF
    )
    log "assets ${ASSET_REPO}@${ASSET_REVISION} -> ${ASSETS}"
fi
log "install complete"
