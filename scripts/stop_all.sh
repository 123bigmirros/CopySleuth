#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

BACKEND_PORT="${BACKEND_PORT:-8003}"
FRONTEND_PORT="${FRONTEND_PORT:-5178}"
OCR_PORT="${OCR_PORT:-8002}"
ALGO_PORT="${ALGO_PORT:-8001}"
MEDIA_PORT="${MEDIA_PORT:-8012}"
KILL_PORTS="${KILL_PORTS:-1}"

if [ ! -d "${LOG_DIR}" ]; then
  echo "Log directory not found: ${LOG_DIR}" >&2
  exit 1
fi

wait_for_exit() {
  local pid="$1"
  local name="$2"
  local tries=20
  while kill -0 "${pid}" >/dev/null 2>&1; do
    if [ "${tries}" -le 0 ]; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
      echo "${name}: force killed (pid ${pid})"
      return 0
    fi
    tries=$((tries - 1))
    sleep 0.1
  done
  return 0
}

stop_pid() {
  local pid_file="$1"
  local name="$2"
  if [ ! -f "${pid_file}" ]; then
    echo "${name}: pid file not found (${pid_file})"
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [ -z "${pid}" ]; then
    echo "${name}: pid file empty (${pid_file})"
    return 0
  fi
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait_for_exit "${pid}" "${name}"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "${name}: still running (pid ${pid})"
    else
      echo "${name}: stopped (pid ${pid})"
    fi
  else
    echo "${name}: not running (pid ${pid})"
  fi
  rm -f "${pid_file}"
}

stop_pid "${LOG_DIR}/backend.pid" "backend"
stop_pid "${LOG_DIR}/frontend.pid" "frontend"
stop_pid "${LOG_DIR}/media.pid" "media"
stop_pid "${LOG_DIR}/ocr.pid" "ocr"
stop_pid "${LOG_DIR}/algo.pid" "algo"

BACKEND_PATTERN="uvicorn app.main:app"
MEDIA_PATTERN="uvicorn media_service.app:app"
OCR_PATTERN="uvicorn server.api:app"
FRONTEND_PATTERN="vite/bin/vite.js"
ALGO_PATTERN="uvicorn algo_service.app:app"
PORT_KILL_TOOL=""
if command -v ss >/dev/null 2>&1; then
  PORT_KILL_TOOL="ss"
elif command -v lsof >/dev/null 2>&1; then
  PORT_KILL_TOOL="lsof"
fi

kill_by_pattern() {
  local pattern="$1"
  local name="$2"
  local pids
  pids="$(pgrep -f "${pattern}" || true)"
  if [ -z "${pids}" ]; then
    echo "${name}: no matching process"
    return 0
  fi
  echo "${name}: stopping leftover processes (${pids})"
  for pid in ${pids}; do
    kill "${pid}" >/dev/null 2>&1 || true
    wait_for_exit "${pid}" "${name}"
  done
}

kill_by_pattern "${BACKEND_PATTERN}" "backend"
kill_by_pattern "${FRONTEND_PATTERN}" "frontend"
kill_by_pattern "${MEDIA_PATTERN}" "media"
kill_by_pattern "${OCR_PATTERN}" "ocr"
kill_by_pattern "${ALGO_PATTERN}" "algo"

get_port_pids() {
  local port="$1"
  local output=""
  if [ "${PORT_KILL_TOOL}" = "ss" ]; then
    output="$(ss -lntp "sport = :${port}" 2>/dev/null || true)"
    printf '%s\n' "${output}" | grep -o 'pid=[0-9]\+' | cut -d= -f2 || true
  elif [ "${PORT_KILL_TOOL}" = "lsof" ]; then
    lsof -tiTCP:${port} -sTCP:LISTEN 2>/dev/null || true
  fi
}

kill_by_port() {
  local port="$1"
  local name="$2"
  if [ -z "${port}" ]; then
    return 0
  fi
  local pids
  pids="$(get_port_pids "${port}")"
  if [ -z "${pids}" ]; then
    echo "${name}: no process listening on port ${port}"
    return 0
  fi
  echo "${name}: stopping processes on port ${port} (${pids})"
  for pid in ${pids}; do
    kill "${pid}" >/dev/null 2>&1 || true
    wait_for_exit "${pid}" "${name}"
  done
}

if [ "${KILL_PORTS}" = "1" ]; then
  if [ -z "${PORT_KILL_TOOL}" ]; then
    echo "port-kill: skipped (ss/lsof not found)"
  else
    kill_by_port "${BACKEND_PORT}" "backend"
    kill_by_port "${FRONTEND_PORT}" "frontend"
    kill_by_port "${OCR_PORT}" "ocr"
    kill_by_port "${ALGO_PORT}" "algo"
    kill_by_port "${MEDIA_PORT}" "media"
  fi
fi
