#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source .env if present
if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

OCR_PORT="${OCR_PORT:-8002}"
ALGO_PORT="${ALGO_PORT:-8001}"
BACKEND_PORT="${BACKEND_PORT:-8003}"
MEDIA_PORT="${MEDIA_PORT:-8012}"

WAIT=0
TIMEOUT=120
for arg in "$@"; do
  case "${arg}" in
    --wait) WAIT=1 ;;
    --timeout=*) TIMEOUT="${arg#*=}" ;;
  esac
done

check_service() {
  local name="$1"
  local url="$2"
  local status
  status="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "${url}" 2>/dev/null || echo "000")"
  if [ "${status}" = "200" ]; then
    printf "  %-12s %s  ok\n" "${name}" "${url}"
    return 0
  else
    printf "  %-12s %s  unreachable (HTTP %s)\n" "${name}" "${url}" "${status}"
    return 1
  fi
}

SERVICES=(
  "ocr|http://127.0.0.1:${OCR_PORT}/v1/health"
  "algo|http://127.0.0.1:${ALGO_PORT}/v1/health"
  "media|http://127.0.0.1:${MEDIA_PORT}/v1/health"
  "backend|http://127.0.0.1:${BACKEND_PORT}/health"
)

run_checks() {
  local all_ok=0
  echo "Service health:"
  for entry in "${SERVICES[@]}"; do
    local name="${entry%%|*}"
    local url="${entry##*|}"
    if ! check_service "${name}" "${url}"; then
      all_ok=1
    fi
  done
  return "${all_ok}"
}

if [ "${WAIT}" = "1" ]; then
  elapsed=0
  while ! run_checks; do
    elapsed=$((elapsed + 5))
    if [ "${elapsed}" -ge "${TIMEOUT}" ]; then
      echo "Timed out after ${TIMEOUT}s waiting for services."
      exit 1
    fi
    echo "Retrying in 5s... (${elapsed}/${TIMEOUT}s)"
    sleep 5
  done
  echo "All services healthy."
else
  if run_checks; then
    echo "All services healthy."
  else
    echo "Some services are not reachable."
    exit 1
  fi
fi
