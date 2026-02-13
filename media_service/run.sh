#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8012}"
MEDIA_STORE_DIR="${MEDIA_STORE_DIR:-./media_store}"
MEDIA_BASE_URL="${MEDIA_BASE_URL:-}"

export MEDIA_STORE_DIR MEDIA_BASE_URL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

uvicorn media_service.app:app --host "$HOST" --port "$PORT"
