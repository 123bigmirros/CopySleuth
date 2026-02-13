#!/usr/bin/env bash
# Test image-to-video matching via the algorithm service (async task + SSE).
# Usage: ./scripts/test_video.sh <query_image> <target_video> [output.json]
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <query_image> <target_video> [output.json]"
  echo ""
  echo "Example:"
  echo "  $0 logo.png video.mp4 result.json"
  echo "  $0 logo.png video.mp4 result.json   # streams SSE progress"
  exit 1
fi

QUERY="$1"
TARGET="$2"
OUT_JSON="${3:-result_video.json}"
BASE_URL="${ALGO_BASE_URL:-http://127.0.0.1:8001}"

echo "Query:  ${QUERY}"
echo "Target: ${TARGET}"
echo "Output: ${OUT_JSON}"
echo "Server: ${BASE_URL}"
echo ""

python -m algo_service.requests_client task \
  --base-url "${BASE_URL}" \
  --query "${QUERY}" \
  --target "${TARGET}" \
  --image-mode b64 \
  --out-json "${OUT_JSON}"

echo ""
echo "Done. Results saved to ${OUT_JSON}"
