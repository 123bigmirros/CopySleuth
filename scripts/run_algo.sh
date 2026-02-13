#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Source .env if present
if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

ALGO_PORT="${ALGO_PORT:-8001}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH; please initialize conda first." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
ALGO_PATTERN="uvicorn algo_service.app:app"

if pgrep -f "${ALGO_PATTERN}" >/dev/null 2>&1; then
  algo_pid="$(pgrep -f "${ALGO_PATTERN}" | head -n1 || true)"
  echo "Algorithm service already running (pid ${algo_pid})."
  exit 0
fi

nohup bash -lc "
  source \"${CONDA_BASE}/etc/profile.d/conda.sh\"
  conda activate pic_cmp
  export SAM3_DIR=\"${SAM3_DIR:-${ROOT_DIR}/models/sam3}\"
  export ROMA_V2_DIR=\"${ROMA_V2_DIR:-${ROOT_DIR}/models/RoMaV2}\"
  export ROMA_V2_WEIGHTS=\"${ROMA_V2_WEIGHTS:-${ROOT_DIR}/models/RoMaV2/checkpoints/romav2.pt}\"
  export DINOV3_DIR=\"${DINOV3_DIR:-${ROOT_DIR}/models/DINOv3}\"
  export SAM3_CHECKPOINT=\"${SAM3_CHECKPOINT:-${ROOT_DIR}/models/sam3/sam3_checkpoints/sam3.pt}\"
  export ALGO_DATA_DIR=\"${ALGO_DATA_DIR:-${ROOT_DIR}/data/algo}\"
  export DEVICE=\"${DEVICE:-cuda}\"
  exec uvicorn algo_service.app:app --app-dir \"${ROOT_DIR}\" --host 127.0.0.1 --port \"${ALGO_PORT}\"
" > "${LOG_DIR}/algo.log" 2>&1 & echo $! > "${LOG_DIR}/algo.pid"

echo "Algorithm service started on port ${ALGO_PORT} (logs: ${LOG_DIR}/algo.log)"
