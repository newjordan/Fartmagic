#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="${ROOT}/experiments/2026-04-07_midnight_v.2/tests_smart_unet"
RUN_ONE="${EXP_DIR}/run_ablation.sh"
MATRIX="${EXP_DIR}/ablation_matrix.tsv"
RUN_LOG_DIR="${EXP_DIR}/logs/ablation_runs"
OUT_ROOT="${EXP_DIR}/logs/mode_batches"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
PROFILE="${PROFILE:-screen}"
PRIORITY_MAX="${PRIORITY_MAX:-1}"
BUDGET_MINUTES="${BUDGET_MINUTES:-0}"
DRY_RUN="${DRY_RUN:-0}"
REUSE_DONE="${REUSE_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
STREAM_RUNNER_LOGS="${STREAM_RUNNER_LOGS:-1}"

if [[ ! -x "${RUN_ONE}" ]]; then
  echo "FATAL: missing run script: ${RUN_ONE}" >&2
  exit 1
fi

mapfile -t LANES < <(awk -F'\t' -v pmax="${PRIORITY_MAX}" 'NR>1 && ($2+0) <= (pmax+0) {print $1}' "${MATRIX}")
ORDERED=()
ORDERED+=("control")
for lane in "${LANES[@]}"; do
  [[ "${lane}" == "control" ]] && continue
  ORDERED+=("${lane}")
done
ORDERED+=("control")

STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="${OUT_ROOT}/${STAMP}"
mkdir -p "${BATCH_DIR}"
SUMMARY="${BATCH_DIR}/summary.tsv"
printf "idx\tlane\tstatus\tstep\tproxy_val_bpb\tdelta_vs_control\ttrain_time_ms\telapsed_s\tlane_log\tnote\n" > "${SUMMARY}"

batch_start_s="$(date +%s)"
control_bpb=""

echo "batch_dir=${BATCH_DIR}"
echo "profile=${PROFILE} seed=${SEED} nproc=${NPROC} priority_max=${PRIORITY_MAX} budget_minutes=${BUDGET_MINUTES}"
echo "reuse_done=${REUSE_DONE} dry_run=${DRY_RUN}"
echo "stream_runner_logs=${STREAM_RUNNER_LOGS}"
echo "lane_order=${ORDERED[*]}"

afloat() { awk -v a="$1" -v b="$2" 'BEGIN {printf "%.8f", (a - b)}'; }
extract_proxy_line() {
  local f="$1"
  awk 'match($0,/^step:[0-9]+\/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+/){line=$0} END{if(line!="")print line}' "$f"
}
latest_lane_log() {
  local lane="$1"
  ls -1t "${RUN_LOG_DIR}/${lane}_seed${SEED}_"*.log 2>/dev/null | head -n1 || true
}
log_is_done() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  grep -qE "Total submission size mixed\\+brotli|final_eval:skipped|stopping_early: wallclock_cap" "$f"
}

for i in "${!ORDERED[@]}"; do
  lane="${ORDERED[$i]}"
  idx="$((i + 1))"

  if [[ "${BUDGET_MINUTES}" != "0" ]]; then
    now_s="$(date +%s)"
    if (( now_s - batch_start_s >= BUDGET_MINUTES * 60 )); then
      printf "%s\t%s\tSKIPPED\t\t\t\t\t\t\tbudget_reached\n" "$idx" "$lane" >> "${SUMMARY}"
      echo "[${idx}/${#ORDERED[@]}] ${lane} -> SKIPPED (budget_reached)"
      continue
    fi
  fi

  if [[ "${DRY_RUN}" != "1" && "${REUSE_DONE}" == "1" ]]; then
    reused_log="$(latest_lane_log "${lane}")"
    if [[ -n "${reused_log}" ]] && log_is_done "${reused_log}"; then
      proxy_line="$(extract_proxy_line "${reused_log}")"
      step="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*step:\([0-9][0-9]*\)\/[0-9][0-9]*.*/\1/p')"
      proxy_bpb="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*val_bpb:\([0-9.][0-9.]*\).*/\1/p')"
      train_time_ms="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*train_time:\([0-9][0-9]*\)ms.*/\1/p')"
      [[ -z "${control_bpb}" && "${lane}" == "control" ]] && control_bpb="${proxy_bpb}"
      delta=""
      [[ -n "${control_bpb}" && -n "${proxy_bpb}" ]] && delta="$(afloat "${proxy_bpb}" "${control_bpb}")"
      printf "%s\t%s\tREUSED\t%s\t%s\t%s\t%s\t0\t%s\treused_completed_log\n" "$idx" "$lane" "$step" "$proxy_bpb" "$delta" "$train_time_ms" "$reused_log" >> "${SUMMARY}"
      echo "[${idx}/${#ORDERED[@]}] ${lane} -> REUSED (${reused_log})"
      continue
    fi
  fi

  run_start_s="$(date +%s)"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "%s\t%s\tDRYRUN\t\t\t\t\t0\t\tdry_run\n" "$idx" "$lane" >> "${SUMMARY}"
    echo "[${idx}/${#ORDERED[@]}] ${lane} -> DRYRUN"
    continue
  fi

  echo "[${idx}/${#ORDERED[@]}] Running ${lane} (${PROFILE})..."
  set +e
  if [[ "${STREAM_RUNNER_LOGS}" == "1" ]]; then
    SEED="${SEED}" NPROC_PER_NODE="${NPROC}" bash "${RUN_ONE}" "${lane}" "${PROFILE}" 2>&1 | tee "${BATCH_DIR}/${idx}_${lane}.runner.log"
    rc="${PIPESTATUS[0]}"
  else
    SEED="${SEED}" NPROC_PER_NODE="${NPROC}" bash "${RUN_ONE}" "${lane}" "${PROFILE}" > "${BATCH_DIR}/${idx}_${lane}.runner.log" 2>&1
    rc="$?"
  fi
  set -e

  run_end_s="$(date +%s)"
  elapsed_s="$((run_end_s - run_start_s))"
  lane_log="$(latest_lane_log "${lane}")"
  status="OK"
  note=""
  [[ "${rc}" -ne 0 ]] && status="FAIL" && note="runner_exit_${rc}"

  proxy_line="$(extract_proxy_line "${lane_log}")"
  step="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*step:\([0-9][0-9]*\)\/[0-9][0-9]*.*/\1/p')"
  proxy_bpb="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*val_bpb:\([0-9.][0-9.]*\).*/\1/p')"
  train_time_ms="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*train_time:\([0-9][0-9]*\)ms.*/\1/p')"

  [[ -z "${control_bpb}" && "${lane}" == "control" ]] && control_bpb="${proxy_bpb}"
  delta=""
  [[ -n "${control_bpb}" && -n "${proxy_bpb}" ]] && delta="$(afloat "${proxy_bpb}" "${control_bpb}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$idx" "$lane" "$status" "$step" "$proxy_bpb" "$delta" "$train_time_ms" "$elapsed_s" "$lane_log" "$note" >> "${SUMMARY}"
  echo "[${idx}/${#ORDERED[@]}] ${lane} done status=${status} proxy_bpb=${proxy_bpb:-NA} delta_vs_control=${delta:-NA} elapsed=${elapsed_s}s"

  if [[ "${status}" == "FAIL" && "${STOP_ON_FAIL}" == "1" ]]; then
    echo "Stopping batch due to failure"
    break
  fi
done

echo ""
echo "Batch summary: ${SUMMARY}"
if command -v column >/dev/null 2>&1; then
  {
    head -n1 "${SUMMARY}"
    tail -n +2 "${SUMMARY}" | awk -F'\t' '($3=="OK" || $3=="REUSED") && $5!=""' | sort -t$'\t' -k5,5n
  } | column -t -s $'\t'
else
  cat "${SUMMARY}"
fi
