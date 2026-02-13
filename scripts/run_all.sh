#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Source .env if present
if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

BACKEND_PORT="${BACKEND_PORT:-8003}"
FRONTEND_PORT="${FRONTEND_PORT:-5178}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
HOST_IP="${HOST_IP:-}"
OCR_PORT="${OCR_PORT:-8002}"
ALGO_PORT="${ALGO_PORT:-8001}"
if [ -n "${HOST_IP}" ]; then
  API_BASE="${API_BASE:-http://${HOST_IP}:${BACKEND_PORT}}"
else
  API_BASE="${API_BASE:-http://127.0.0.1:${BACKEND_PORT}}"
fi
ALGO_SERVICE_URL="${ALGO_SERVICE_URL:-http://127.0.0.1:${ALGO_PORT}}"
MEDIA_SERVICE_URL="${MEDIA_SERVICE_URL:-http://127.0.0.1:8012}"
START_OCR="${START_OCR:-1}"
START_ALGO="${START_ALGO:-1}"
START_BACKEND="${START_BACKEND:-1}"
START_FRONTEND="${START_FRONTEND:-1}"
START_MEDIA="${START_MEDIA:-1}"
MEDIA_PORT="${MEDIA_PORT:-8012}"
MEDIA_STORE_DIR="${MEDIA_STORE_DIR:-${ROOT_DIR}/data/media}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH; please initialize conda first." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"

BACKEND_PATTERN="uvicorn app.main:app"
OCR_PATTERN="uvicorn server.api:app"
FRONTEND_PATTERN="vite/bin/vite.js"
ALGO_PATTERN="uvicorn algo_service.app:app"
MEDIA_PATTERN="uvicorn media_service.app:app"

# --- OCR ---
if [ "${START_OCR}" = "1" ]; then
  if pgrep -f "${OCR_PATTERN}" >/dev/null 2>&1; then
    ocr_pid="$(pgrep -f "${OCR_PATTERN}" | head -n1 || true)"
    echo "${ocr_pid}" > "${LOG_DIR}/ocr.pid"
    echo "OCR service already running (pid ${ocr_pid})."
  else
    nohup bash -lc "
      source \"${CONDA_BASE}/etc/profile.d/conda.sh\"
      conda activate pic_cmp
      export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
      export PORT=\"${OCR_PORT}\"
      cd \"${ROOT_DIR}/paddle-ocr\"
      exec ./run_server.sh
    " > "${LOG_DIR}/ocr.log" 2>&1 & echo $! > "${LOG_DIR}/ocr.pid"
  fi
fi

# --- Algo ---
if [ "${START_ALGO}" = "1" ]; then
  if pgrep -f "${ALGO_PATTERN}" >/dev/null 2>&1; then
    algo_pid="$(pgrep -f "${ALGO_PATTERN}" | head -n1 || true)"
    echo "${algo_pid}" > "${LOG_DIR}/algo.pid"
    echo "Algorithm service already running (pid ${algo_pid})."
  else
    nohup bash -lc "
      source \"${CONDA_BASE}/etc/profile.d/conda.sh\"
      conda activate pic_cmp
      export SAM3_DIR=\"${SAM3_DIR:-${ROOT_DIR}/models/sam3}\"
      export ROMA_V2_DIR=\"${ROMA_V2_DIR:-${ROOT_DIR}/models/RoMaV2}\"
      export ROMA_V2_WEIGHTS=\"${ROMA_V2_WEIGHTS:-${ROOT_DIR}/models/RoMaV2/checkpoints/romav2.pt}\"
      export ROMA_V2_COMPILE=\"${ROMA_V2_COMPILE:-0}\"
      export ALGO_TASK_WORKERS=\"${ALGO_TASK_WORKERS:-1}\"
      export DINOV3_DIR=\"${DINOV3_DIR:-${ROOT_DIR}/models/DINOv3}\"
      export EMBED_MAX_EDGE=\"${EMBED_MAX_EDGE:-518}\"
      export SAM3_CHECKPOINT=\"${SAM3_CHECKPOINT:-${ROOT_DIR}/models/sam3/sam3_checkpoints/sam3.pt}\"
      export ALGO_DATA_DIR=\"${ALGO_DATA_DIR:-${ROOT_DIR}/data/algo}\"
      export DEVICE=\"${DEVICE:-cuda}\"
      exec uvicorn algo_service.app:app --app-dir \"${ROOT_DIR}\" --host 127.0.0.1 --port \"${ALGO_PORT}\"
    " > "${LOG_DIR}/algo.log" 2>&1 & echo $! > "${LOG_DIR}/algo.pid"
  fi
fi

# --- Media ---
if [ "${START_MEDIA}" = "1" ]; then
  if pgrep -f "${MEDIA_PATTERN}" >/dev/null 2>&1; then
    media_pid="$(pgrep -f "${MEDIA_PATTERN}" | head -n1 || true)"
    echo "${media_pid}" > "${LOG_DIR}/media.pid"
    echo "Media service already running (pid ${media_pid})."
  else
    nohup bash -lc "
      source \"${CONDA_BASE}/etc/profile.d/conda.sh\"
      conda activate pic_cmp
      export MEDIA_STORE_DIR=\"${MEDIA_STORE_DIR}\"
      exec uvicorn media_service.app:app --app-dir \"${ROOT_DIR}\" --host 127.0.0.1 --port \"${MEDIA_PORT}\"
    " > "${LOG_DIR}/media.log" 2>&1 & echo $! > "${LOG_DIR}/media.pid"
  fi
fi

# --- Backend ---
if [ "${START_BACKEND}" = "1" ]; then
  if pgrep -f "${BACKEND_PATTERN}" >/dev/null 2>&1; then
    backend_pid="$(pgrep -f "${BACKEND_PATTERN}" | head -n1 || true)"
    echo "${backend_pid}" > "${LOG_DIR}/backend.pid"
    echo "Backend already running (pid ${backend_pid})."
  else
    nohup bash -lc "
      source \"${CONDA_BASE}/etc/profile.d/conda.sh\"
      conda activate pic_cmp
      export SAM3_DIR=\"${SAM3_DIR:-${ROOT_DIR}/models/sam3}\"
      export ROMA_V2_DIR=\"${ROMA_V2_DIR:-${ROOT_DIR}/models/RoMaV2}\"
      export ROMA_V2_WEIGHTS=\"${ROMA_V2_WEIGHTS:-${ROOT_DIR}/models/RoMaV2/checkpoints/romav2.pt}\"
      export ROMA_V2_COMPILE=\"${ROMA_V2_COMPILE:-0}\"
      export DINOV3_DIR=\"${DINOV3_DIR:-${ROOT_DIR}/models/DINOv3}\"
      export SAM3_CHECKPOINT=\"${SAM3_CHECKPOINT:-${ROOT_DIR}/models/sam3/sam3_checkpoints/sam3.pt}\"
      export PADDLE_OCR_DIR=\"${ROOT_DIR}/paddle-ocr\"
      export DEVICE=\"${DEVICE:-cuda}\"
      export ALGO_SERVICE_URL=\"${ALGO_SERVICE_URL}\"
      export OCR_API_BASE=\"${OCR_API_BASE:-http://127.0.0.1:${OCR_PORT}}\"
      export MEDIA_SERVICE_URL=\"${MEDIA_SERVICE_URL}\"
      exec uvicorn app.main:app --app-dir \"${ROOT_DIR}/backend\" --host 0.0.0.0 --port \"${BACKEND_PORT}\"
    " > "${LOG_DIR}/backend.log" 2>&1 & echo $! > "${LOG_DIR}/backend.pid"
  fi
fi

# --- Frontend ---
if [ "${START_FRONTEND}" = "1" ]; then
  if pgrep -f "${FRONTEND_PATTERN}" >/dev/null 2>&1; then
    frontend_pid="$(pgrep -f "${FRONTEND_PATTERN}" | head -n1 || true)"
    echo "${frontend_pid}" > "${LOG_DIR}/frontend.pid"
    echo "Frontend already running (pid ${frontend_pid})."
  else
    nohup bash -lc "
      cd \"${ROOT_DIR}/frontend\"
      unset VITE_API_BASE
      if [ -n \"${API_BASE}\" ]; then
        export VITE_API_BASE=\"${API_BASE}\"
      fi
      exec npm run dev -- --host \"${FRONTEND_HOST}\" --port \"${FRONTEND_PORT}\"
    " > "${LOG_DIR}/frontend.log" 2>&1 & echo $! > "${LOG_DIR}/frontend.pid"
  fi
fi

# --- Summary ---
summary="Started services:\n"
if [ "${START_OCR}" = "1" ]; then
  summary+="- ocr:      http://127.0.0.1:${OCR_PORT} (logs: ${LOG_DIR}/ocr.log)\n"
fi
if [ "${START_ALGO}" = "1" ]; then
  summary+="- algo:     http://127.0.0.1:${ALGO_PORT}  (logs: ${LOG_DIR}/algo.log)\n"
fi
if [ "${START_MEDIA}" = "1" ]; then
  summary+="- media:    http://127.0.0.1:${MEDIA_PORT}  (logs: ${LOG_DIR}/media.log)\n"
fi
if [ "${START_BACKEND}" = "1" ]; then
  summary+="- backend:  http://127.0.0.1:${BACKEND_PORT}  (logs: ${LOG_DIR}/backend.log)\n"
fi
if [ "${START_FRONTEND}" = "1" ]; then
  summary+="- frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT} (logs: ${LOG_DIR}/frontend.log)\n"
  summary+="Frontend API base: ${API_BASE}\n"
fi
summary+="\nPIDs saved to ${LOG_DIR}/*.pid\n"
printf "%b" "${summary}"
