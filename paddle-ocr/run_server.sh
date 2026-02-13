#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH; please initialize conda (e.g., source ~/miniconda3/etc/profile.d/conda.sh)" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pic_cmp

PORT="${PORT:-8002}"
python -m uvicorn server.api:app --host 0.0.0.0 --port "${PORT}"
