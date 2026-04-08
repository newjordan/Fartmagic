#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/shroud_midnight_12l_$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-8787}"
SHROUD_SERVE="${SHROUD_SERVE:-1}"

SPEC_PATH="${SPEC_PATH:-${SCRIPT_DIR}/visualizer/midnight_12l_spec.json}"
POINTS_PATH="${POINTS_PATH:-${RESULT_DIR}/shroud_midnight_12l_latest.trace.points.json}"
FLOW_PATH="${FLOW_PATH:-${RESULT_DIR}/shroud_midnight_12l_latest.architecture_flow.json}"
LATEST_POINTS="${REPO_ROOT}/results/shroud_midnight_12l_latest.trace.points.json"
LATEST_FLOW="${REPO_ROOT}/results/shroud_midnight_12l_latest.architecture_flow.json"

mkdir -p "${RESULT_DIR}" "${REPO_ROOT}/results"

echo "building Midnight 12L Shroud assets"
python3 "${SCRIPT_DIR}/visualizer/build_midnight_12l_points.py" \
  --spec "${SPEC_PATH}" \
  --output-points "${POINTS_PATH}" \
  --output-flow "${FLOW_PATH}"

cp -f "${POINTS_PATH}" "${LATEST_POINTS}"
cp -f "${FLOW_PATH}" "${LATEST_FLOW}"

echo "wrote:"
echo "  ${POINTS_PATH}"
echo "  ${FLOW_PATH}"
echo "  ${LATEST_POINTS}"
echo "  ${LATEST_FLOW}"

if [[ "${SHROUD_SERVE}" != "0" ]]; then
  HTTP_LOG="${RESULT_DIR}/http.server.log"
  HTTP_PID="${RESULT_DIR}/http.server.pid"

  if ! ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    nohup python3 -m http.server "${PORT}" --bind 127.0.0.1 --directory "${REPO_ROOT}" >"${HTTP_LOG}" 2>&1 &
    echo $! > "${HTTP_PID}"
    echo "started local server on http://127.0.0.1:${PORT}"
  else
    echo "port ${PORT} already in use; reusing existing local server"
  fi

  if command -v tailscale >/dev/null 2>&1; then
    if tailscale status >/dev/null 2>&1; then
      tailscale serve --bg "${PORT}" >/dev/null 2>&1 || true
      echo "tailscale serve requested on port ${PORT}"
    else
      echo "tailscale installed but not active; skip tunnel exposure"
    fi
  fi
fi

echo "open:"
echo "  http://127.0.0.1:${PORT}/experiments/shroud_nightcrawler/visualizer/shroud_viewer.html?arch=midnight_12l"
echo "  load the Midnight 12L points from results/shroud_midnight_12l_latest.trace.points.json"
