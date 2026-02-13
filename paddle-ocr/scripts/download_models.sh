#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR"

mkdir -p "$MODELS_DIR"

usage() {
  cat <<'USAGE'
Usage: download_models.sh [--with-textline-orientation]

Downloads PP-OCRv5 server detection/recognition inference models into repo root.
Optional: also download the text line orientation model.
USAGE
}

with_textline_orientation=0
if [[ "${1:-}" == "--with-textline-orientation" ]]; then
  with_textline_orientation=1
elif [[ -n "${1:-}" ]]; then
  usage
  exit 1
fi

download_and_extract() {
  local name="$1"
  local url="$2"
  local tar_path="$MODELS_DIR/${name}.tar"

  if [[ -d "$MODELS_DIR/$name" ]]; then
    echo "[skip] $name already exists"
    return 0
  fi

  echo "[download] $name"
  curl -L --fail "$url" -o "$tar_path"
  echo "[extract] $name"
  tar -xf "$tar_path" -C "$MODELS_DIR"
  rm -f "$tar_path"
}

# PP-OCRv5 server models (official inference models)
download_and_extract \
  "PP-OCRv5_server_det_infer" \
  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar"

download_and_extract \
  "PP-OCRv5_server_rec_infer" \
  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar"

if [[ "$with_textline_orientation" -eq 1 ]]; then
  download_and_extract \
    "PP-LCNet_x0_25_textline_ori_infer" \
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar"
fi

echo "Done. Models are in $MODELS_DIR"
