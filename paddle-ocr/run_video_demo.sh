#!/usr/bin/env bash
set -euo pipefail

python video_ocr_from_excel.py \
  /home/dymiao/pic_cmp/video/20250214_104446.mp4 \
  /home/dymiao/pic_cmp/video/demo.xlsx \
  --drop-header \
  --keyword-col 1 \
  --frame-select scenedetect
