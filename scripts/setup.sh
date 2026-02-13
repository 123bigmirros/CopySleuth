#!/usr/bin/env bash
# One-step setup: download models + install all dependencies.
# Usage: ./scripts/setup.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models"
mkdir -p "${MODELS_DIR}"

echo "=== [1/4] Installing pip dependencies ==="
pip install -r "${ROOT_DIR}/backend/requirements.txt"
pip install -e "${ROOT_DIR}[test]"
pip install modelscope

echo ""
echo "=== [2/4] Downloading models ==="

# SAM3 (via ModelScope)
SAM3_DIR="${MODELS_DIR}/sam3"
if [[ -d "${SAM3_DIR}" ]]; then
  echo "[skip] SAM3 already exists at ${SAM3_DIR}"
else
  echo "[download] SAM3 -> ${SAM3_DIR}"
  python -c "
from modelscope import snapshot_download
snapshot_download('facebook/sam3', local_dir='${SAM3_DIR}')
"
fi

# DINOv3 (via ModelScope)
DINOV3_DIR="${MODELS_DIR}/DINOv3"
if [[ -d "${DINOV3_DIR}" ]]; then
  echo "[skip] DINOv3 already exists at ${DINOV3_DIR}"
else
  echo "[download] DINOv3 -> ${DINOV3_DIR}"
  python -c "
from modelscope import snapshot_download
snapshot_download('facebook/dinov3-vitb16-pretrain-lvd1689m', local_dir='${DINOV3_DIR}')
"
fi

# RoMaV2 (via git + weights download)
ROMA_V2_DIR="${MODELS_DIR}/RoMaV2"
if [[ -d "${ROMA_V2_DIR}" ]]; then
  echo "[skip] RoMaV2 already exists at ${ROMA_V2_DIR}"
else
  echo "[download] RoMaV2 -> ${ROMA_V2_DIR}"
  git clone https://github.com/Parskatt/RoMaV2 "${ROMA_V2_DIR}"
  mkdir -p "${ROMA_V2_DIR}/checkpoints"
  if [[ ! -f "${ROMA_V2_DIR}/checkpoints/romav2.pt" ]]; then
    echo "[download] RoMaV2 weights"
    curl -L --fail \
      https://github.com/Parskatt/RoMaV2/releases/download/weights/romav2.pt \
      -o "${ROMA_V2_DIR}/checkpoints/romav2.pt"
  fi
fi

echo ""
echo "=== [3/4] Installing model packages ==="
cd "${MODELS_DIR}/sam3"   && pip install -e ".[all]" && cd "${ROOT_DIR}"
cd "${MODELS_DIR}/RoMaV2" && pip install -e .        && cd "${ROOT_DIR}"

echo ""
echo "=== [4/4] Done ==="
echo "All dependencies installed and models downloaded to ${MODELS_DIR}/"
echo ""
echo "Next: start services with ./scripts/run_algo.sh or ./scripts/run_all.sh"
