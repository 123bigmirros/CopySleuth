#!/usr/bin/env bash
# Test image-to-image matching via the algorithm service.
# Usage: ./scripts/test_image.sh <query_image> <target_image> [output.json]
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <query_image> <target_image> [output.json]"
  echo ""
  echo "Example:"
  echo "  $0 logo.png screenshot.png result.json"
  exit 1
fi

QUERY="$1"
TARGET="$2"
OUT_JSON="${3:-result_image.json}"
BASE_URL="${ALGO_BASE_URL:-http://127.0.0.1:8001}"

echo "Query:  ${QUERY}"
echo "Target: ${TARGET}"
echo "Output: ${OUT_JSON}"
echo "Server: ${BASE_URL}"
echo ""

python -m algo_service.requests_client detect \
  --base-url "${BASE_URL}" \
  --query "${QUERY}" \
  --target "${TARGET}" \
  --image-mode b64 \
  --out-json "${OUT_JSON}" \
  --save-candidates ./candidates

echo ""
echo "Done. Results saved to ${OUT_JSON}"
echo "Candidate crops saved to ./candidates/"
