#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSIONS_FILE="${ROOT_DIR}/scripts/versions.env"

if [[ -f "${VERSIONS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${VERSIONS_FILE}"
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git not found; please install git first." >&2
  exit 1
fi

clone_repo() {
  local name="$1"
  local url="$2"
  local ref="${3:-}"
  local dest="${ROOT_DIR}/models/${name}"

  if [[ -d "${dest}" ]]; then
    echo "[skip] ${name} already exists at ${dest}"
    return 0
  fi

  echo "[clone] ${name} from ${url}"
  mkdir -p "${ROOT_DIR}/models"
  git clone "${url}" "${dest}"
  if [[ -n "${ref}" ]]; then
    echo "[checkout] ${name} ${ref}"
    git -C "${dest}" checkout "${ref}"
  fi
}

clone_repo "sam3" "${SAM3_REPO_URL:-https://github.com/facebookresearch/sam3}" "${SAM3_REF:-}"
clone_repo "RoMaV2" "${ROMA_V2_REPO_URL:-https://github.com/Parskatt/RoMaV2}" "${ROMA_V2_REF:-}"

# PaddleOCR lives at project root (not under models/)
PADDLE_DEST="${ROOT_DIR}/paddle-ocr"
if [[ -d "${PADDLE_DEST}" ]]; then
  echo "[skip] paddle-ocr already exists at ${PADDLE_DEST}"
else
  echo "[clone] paddle-ocr from ${PADDLE_OCR_REPO_URL:-https://github.com/PaddlePaddle/PaddleOCR}"
  git clone "${PADDLE_OCR_REPO_URL:-https://github.com/PaddlePaddle/PaddleOCR}" "${PADDLE_DEST}"
  if [[ -n "${PADDLE_OCR_REF:-}" ]]; then
    git -C "${PADDLE_DEST}" checkout "${PADDLE_OCR_REF}"
  fi
fi

echo "Done. Model repos are in ${ROOT_DIR}/models/."
