#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
UV_PROJECT_ENVIRONMENT="$repo_root/envs/mikasa_robo/.venv" \
  uv sync --frozen --no-dev --project "$repo_root/third_party/mikasa_robo" --python 3.11
