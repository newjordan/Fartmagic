#!/usr/bin/env bash
set -euo pipefail

# Serve the Shroud viewer for Midnight 12L and attach a tailnet URL via tailscale serve.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

HTTP_PORT="${HTTP_PORT:-8790}"
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
PID_FILE="${PID_FILE:-${REPO_ROOT}/results/shroud_viewer_http.pid}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/results/shroud_viewer_http.log}"
VIEWER_PATH="/experiments/shroud_nightcrawler/visualizer/shroud_viewer.html?arch=midnight_12l"

mkdir -p "${REPO_ROOT}/results"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && ps -p "${old_pid}" >/dev/null 2>&1; then
    echo "stopping existing local server pid=${old_pid}"
    kill "${old_pid}" || true
    sleep 1
  fi
  rm -f "${PID_FILE}"
fi

echo "starting local static server at http://${BIND_ADDR}:${HTTP_PORT}"
nohup python3 -m http.server "${HTTP_PORT}" --bind "${BIND_ADDR}" --directory "${REPO_ROOT}" \
  >"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"

if command -v tailscale >/dev/null 2>&1; then
  echo "configuring tailscale serve -> http://${BIND_ADDR}:${HTTP_PORT}"
  tailscale serve --bg --yes "http://${BIND_ADDR}:${HTTP_PORT}" >/dev/null
  echo "tailscale serve status:"
  tailscale serve status || true
else
  echo "tailscale not found; skipping tunnel setup"
fi

echo ""
echo "Viewer URLs:"
echo "  local:    http://${BIND_ADDR}:${HTTP_PORT}${VIEWER_PATH}"
echo "  tailnet:  run 'tailscale serve status' and open the ts.net URL + ${VIEWER_PATH}"
echo ""
echo "Dataset:"
echo "  expected: ${REPO_ROOT}/results/shroud_midnight_12l_latest.trace.points.json"
echo ""
