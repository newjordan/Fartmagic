#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-2}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
STALL_AFTER_SECONDS="${STALL_AFTER_SECONDS:-180}"
STALL_POLL_SECONDS="${STALL_POLL_SECONDS:-15}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
SERIES_DIR="${REPO_ROOT}/runs/direct_sota_screens_${RUN_STAMP}"
SUMMARY_FILE="${SERIES_DIR}/summary.tsv"
mkdir -p "${SERIES_DIR}"

extract_val_bpb() {
  local line="${1:-}"
  printf '%s\n' "${line}" | sed -n 's/.*val_bpb:\([0-9.][0-9.]*\).*/\1/p'
}

extract_step() {
  local line="${1:-}"
  printf '%s\n' "${line}" | sed -n 's/.*step:\([0-9][0-9]*\)\/[0-9][0-9]*.*/\1/p'
}

extract_train_ms() {
  local line="${1:-}"
  printf '%s\n' "${line}" | sed -n 's/.*train_time:\([0-9][0-9]*\)ms.*/\1/p'
}

extract_step_avg_ms() {
  local line="${1:-}"
  printf '%s\n' "${line}" | sed -n 's/.*step_avg:\([0-9.][0-9.]*\)ms.*/\1/p'
}

find_metric_source() {
  local latest_log="${1:-}"
  local artifact_rel=""
  if [[ -n "${latest_log}" && -f "${latest_log}" ]]; then
    artifact_rel="$(sed -n 's#^\(logs/[^[:space:]]*\.txt\)$#\1#p' "${latest_log}" | tail -n 1)"
  fi
  if [[ -n "${artifact_rel}" && -f "${REPO_ROOT}/${artifact_rel}" ]]; then
    printf '%s\n' "${REPO_ROOT}/${artifact_rel}"
  else
    printf '%s\n' "${latest_log}"
  fi
}

printf "leg\tstatus\tstep\tval_bpb\ttrain_time_ms\tstep_avg_ms\tlog_file\n" > "${SUMMARY_FILE}"

LEGS=(
  "legs/2026-04-10_control_direct"
  "legs/2026-04-10_muon_wd_095_direct"
  "legs/2026-04-10_matrix_lr_022_direct"
  "legs/2026-04-10_parallel_residual_direct"
  "legs/2026-04-10_depth_recurrence_direct"
  "legs/2026-04-10_stack_core_direct"
)

for leg in "${LEGS[@]}"; do
  leg_name="$(basename "${leg}")"
  echo ""
  echo "=== ${leg_name} ==="
  rc=0
  status="OK"
  timeout "${TIMEOUT_SECONDS}" env SEED="${SEED}" NPROC_PER_NODE="${NPROC}" TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY}" bash "${leg}/run.sh" &
  runner_pid=$!
  start_ts="$(date +%s)"
  last_progress_ts="${start_ts}"
  last_step_seen=""
  latest_log=""
  metric_source=""
  while kill -0 "${runner_pid}" 2>/dev/null; do
    latest_log="$(find "${leg}/logs" -maxdepth 1 -type f -name "screen_seed${SEED}_*.log" | sort | tail -n 1)"
    metric_source="$(find_metric_source "${latest_log}")"
    metric_line=""
    if [[ -n "${metric_source}" && -f "${metric_source}" ]]; then
      metric_line="$(grep -aE 'step:[0-9]+/20000 val_loss:.* val_bpb:|step:[0-9]+/20000 train_loss:' "${metric_source}" | tail -n 1 || true)"
    fi
    current_step="$(extract_step "${metric_line}")"
    if [[ -n "${current_step}" && "${current_step}" != "${last_step_seen}" ]]; then
      last_step_seen="${current_step}"
      last_progress_ts="$(date +%s)"
    fi
    now_ts="$(date +%s)"
    if [[ -n "${last_step_seen}" ]] && (( now_ts - last_progress_ts >= STALL_AFTER_SECONDS )); then
      status="STALL"
      kill "${runner_pid}" 2>/dev/null || true
      pkill -f "${REPO_ROOT}/${leg}/train_gpt.py" || true
      break
    fi
    sleep "${STALL_POLL_SECONDS}"
  done
  wait "${runner_pid}" || rc=$?
  sleep 2
  pkill -f "${REPO_ROOT}/${leg}/train_gpt.py" || true
  sleep 2

  latest_log="$(find "${leg}/logs" -maxdepth 1 -type f -name "screen_seed${SEED}_*.log" | sort | tail -n 1)"
  metric_source="$(find_metric_source "${latest_log}")"
  final_line="$(grep -aE 'step:[0-9]+/20000 val_loss:.* val_bpb:' "${metric_source}" | tail -n 1 || true)"
  last_train_line="$(grep -aE 'step:[0-9]+/20000 train_loss:' "${metric_source}" | tail -n 1 || true)"
  metric_line="${final_line}"
  if [[ -z "${metric_line}" ]]; then
    metric_line="${last_train_line}"
  fi
  step="$(extract_step "${metric_line}")"
  val_bpb="$(extract_val_bpb "${final_line}")"
  train_ms="$(extract_train_ms "${metric_line}")"
  step_avg_ms="$(extract_step_avg_ms "${metric_line}")"

  if [[ "${status}" == "STALL" ]]; then
    :
  elif [[ "${rc}" == "124" ]]; then
    status="TIMEOUT"
  elif [[ "${rc}" != "0" ]]; then
    status="FAIL(${rc})"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${leg_name}" "${status}" "${step}" "${val_bpb}" "${train_ms}" "${step_avg_ms}" "${latest_log}" \
    >> "${SUMMARY_FILE}"
done

echo ""
echo "SUMMARY: ${SUMMARY_FILE}"
