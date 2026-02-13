#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSIONS_FILE="${ROOT_DIR}/scripts/versions.env"

if [[ -f "${VERSIONS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${VERSIONS_FILE}"
fi

hf_download() {
  local model_id="$1"
  local dest="$2"

  # Meta models may require a gated Hugging Face token (export HF_TOKEN=...).
  if command -v huggingface-cli >/dev/null 2>&1; then
    echo "[hf] ${model_id} -> ${dest}"
    huggingface-cli download "${model_id}" \
      --local-dir "${dest}" \
      --local-dir-use-symlinks False \
      ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return 0
  fi

  python - <<PY
import os
import sys

model_id = ${model_id!r}
dest = ${dest!r}

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    print("huggingface_hub not available. Install it or use huggingface-cli.")
    print(f"Import error: {exc}")
    sys.exit(1)

token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

snapshot_download(
    repo_id=model_id,
    local_dir=dest,
    local_dir_use_symlinks=False,
    token=token,
)
print(f"Downloaded {model_id} to {dest}")
PY
}

# SAM3 checkpoint (Hugging Face)
SAM3_DIR="${ROOT_DIR}/models/sam3/sam3_checkpoints"
if [[ -d "${SAM3_DIR}" && -f "${SAM3_DIR}/sam3.pt" ]]; then
  echo "[skip] SAM3 checkpoint already present in ${SAM3_DIR}"
elif [[ -n "${SAM3_MODEL_ID:-}" ]]; then
  mkdir -p "${SAM3_DIR}"
  hf_download "${SAM3_MODEL_ID}" "${SAM3_DIR}"
else
  echo "[warn] SAM3_MODEL_ID not set; skipping SAM3 checkpoint download." >&2
fi

# DINOv3 model (Hugging Face)
DINOV3_DIR="${ROOT_DIR}/models/DINOv3"
if [[ -d "${DINOV3_DIR}" && -f "${DINOV3_DIR}/model.safetensors" ]]; then
  echo "[skip] DINOv3 model already present in ${DINOV3_DIR}"
elif [[ -n "${DINOV3_MODEL_ID:-}" ]]; then
  mkdir -p "${DINOV3_DIR}"
  hf_download "${DINOV3_MODEL_ID}" "${DINOV3_DIR}"
else
  echo "[warn] DINOV3_MODEL_ID not set; skipping DINOv3 download." >&2
fi

# RoMaV2 weights
ROMA_V2_DIR="${ROOT_DIR}/models/RoMaV2"
ROMA_V2_WEIGHTS_PATH="${ROMA_V2_DIR}/checkpoints/romav2.pt"
if [[ -f "${ROMA_V2_WEIGHTS_PATH}" ]]; then
  echo "[skip] RoMaV2 weights already present at ${ROMA_V2_WEIGHTS_PATH}"
elif [[ -n "${ROMA_V2_WEIGHTS_URL:-}" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl not found; cannot download RoMaV2 weights." >&2
  else
    mkdir -p "${ROMA_V2_DIR}/checkpoints"
    echo "[download] RoMaV2 weights"
    curl -L --fail "${ROMA_V2_WEIGHTS_URL}" -o "${ROMA_V2_WEIGHTS_PATH}"
  fi
else
  echo "[warn] ROMA_V2_WEIGHTS_URL not set; skipping RoMaV2 weights download." >&2
fi

# PaddleOCR models
PADDLE_OCR_DIR="${ROOT_DIR}/paddle-ocr"
PADDLE_OCR_SCRIPT="${PADDLE_OCR_DIR}/scripts/download_models.sh"
if [[ -x "${PADDLE_OCR_SCRIPT}" ]]; then
  echo "[paddle-ocr] downloading OCR models"
  "${PADDLE_OCR_SCRIPT}"
else
  echo "[skip] PaddleOCR download script not found at ${PADDLE_OCR_SCRIPT}" >&2
fi

echo "Done."
