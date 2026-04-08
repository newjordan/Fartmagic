#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="${ROOT}/experiments/2026-04-07_midnight_v.2/tests_smart_unet"
RUN_ONE="${EXP_DIR}/run_ablation.sh"
RUN_LOG_DIR="${EXP_DIR}/logs/ablation_runs"
OUT_ROOT="${EXP_DIR}/logs/build_signal_batches"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
PROFILE="${PROFILE:-screen}"
ITERATIONS="${ITERATIONS:-1200}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-${ITERATIONS}}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-420}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}"
SKIP_QUANT_ROUNDTRIP_EVAL="${SKIP_QUANT_ROUNDTRIP_EVAL:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
LANES="${LANES:-soft_gating hard_routing competitive_routing}"
REUSE_DONE="${REUSE_DONE:-1}"
ALLOW_CONTROL_RERUN="${ALLOW_CONTROL_RERUN:-0}"
BASELINE_CONTROL_LOG="${BASELINE_CONTROL_LOG:-}"
CONTROL_PROXY_BPB="${CONTROL_PROXY_BPB:-}"
CONTROL_TRAIN_TIME_MS="${CONTROL_TRAIN_TIME_MS:-}"
ALLOW_MISSING_BASELINE="${ALLOW_MISSING_BASELINE:-0}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-180}"
LANE_TIMEOUT_SECONDS="${LANE_TIMEOUT_SECONDS:-$((MAX_WALLCLOCK_SECONDS + 210))}"

if [[ ! -x "${RUN_ONE}" ]]; then
  echo "FATAL: missing run script: ${RUN_ONE}" >&2
  exit 1
fi

for lane in ${LANES}; do
  if [[ "${lane}" == "control" && "${ALLOW_CONTROL_RERUN}" != "1" ]]; then
    echo "FATAL: lane list includes control but ALLOW_CONTROL_RERUN!=1 (cost guard)." >&2
    echo "Set ALLOW_CONTROL_RERUN=1 only if you explicitly want to pay for a new control run." >&2
    exit 2
  fi
done

mkdir -p "${OUT_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="${OUT_ROOT}/${STAMP}"
mkdir -p "${BATCH_DIR}"
SUMMARY="${BATCH_DIR}/summary.tsv"
printf "idx\tlane\tstatus\tstep\tproxy_val_bpb\tdelta_vs_control_bpb\ttrain_time_ms\tstep_avg_ms\tspeed_vs_control_x\tsmart_active_decoders\telapsed_s\tlane_log\tnote\n" > "${SUMMARY}"

afloat() { awk -v a="$1" -v b="$2" 'BEGIN {printf "%.8f", (a - b)}'; }
aratio() { awk -v a="$1" -v b="$2" 'BEGIN {if (b <= 0) print ""; else printf "%.6f", (a / b)}'; }
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
log_has_target_iteration() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  grep -qE "^step:[0-9]+/${ITERATIONS} val_loss:[0-9.]+ val_bpb:[0-9.]+" "$f"
}
extract_smart_stats() {
  local f="$1"
  awk '
    {
      if (match($0,/smart_active_decoders:[0-9.]+\/[0-9]+/)) {
        tok = substr($0, RSTART, RLENGTH)
        sub(/^smart_active_decoders:/, "", tok)
        split(tok, a, "/")
        v = a[1] + 0
        d = a[2]
        if (count == 0 || v < min_v) min_v = v
        if (count == 0 || v > max_v) max_v = v
        last_v = v
        denom = d
        count++
      }
    }
    END {
      if (count > 0) {
        printf "last=%.2f/%s,min=%.2f,max=%.2f,n=%d", last_v, denom, min_v, max_v, count
      }
    }
  ' "$f"
}
hydrate_control_baseline() {
  if [[ -n "${CONTROL_PROXY_BPB}" && -n "${CONTROL_TRAIN_TIME_MS}" ]]; then
    return
  fi
  local f=""
  if [[ -n "${BASELINE_CONTROL_LOG}" ]]; then
    f="${BASELINE_CONTROL_LOG}"
  else
    f="$(latest_lane_log control)"
  fi
  [[ -n "${f}" && -f "${f}" ]] || return
  local proxy_line
  proxy_line="$(extract_proxy_line "${f}")"
  local bpb
  local tms
  bpb="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*val_bpb:\([0-9.][0-9.]*\).*/\1/p')"
  tms="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*train_time:\([0-9][0-9]*\)ms.*/\1/p')"
  if [[ -n "${bpb}" && -n "${tms}" ]]; then
    CONTROL_PROXY_BPB="${bpb}"
    CONTROL_TRAIN_TIME_MS="${tms}"
  fi
}

hydrate_control_baseline
echo "build_signal_batch_dir=${BATCH_DIR}"
echo "profile=${PROFILE} seed=${SEED} nproc=${NPROC} iterations=${ITERATIONS} val_loss_every=${VAL_LOSS_EVERY}"
echo "max_wallclock_seconds=${MAX_WALLCLOCK_SECONDS} lane_timeout_seconds=${LANE_TIMEOUT_SECONDS} stop_on_fail=${STOP_ON_FAIL}"
echo "reuse_done=${REUSE_DONE}"
echo "lanes=${LANES}"
echo "nccl_async_error_handling=${TORCH_NCCL_ASYNC_ERROR_HANDLING} nccl_heartbeat_timeout_sec=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}"
echo "control_baseline_bpb=${CONTROL_PROXY_BPB:-NA} control_baseline_train_time_ms=${CONTROL_TRAIN_TIME_MS:-NA}"
echo "allow_missing_baseline=${ALLOW_MISSING_BASELINE}"

idx=0
control_bpb="${CONTROL_PROXY_BPB}"
control_train_ms="${CONTROL_TRAIN_TIME_MS}"
if [[ "${ALLOW_MISSING_BASELINE}" != "1" && ( -z "${control_bpb}" || -z "${control_train_ms}" ) ]]; then
  echo "FATAL: missing control baseline (cost guard)." >&2
  echo "Provide BASELINE_CONTROL_LOG or CONTROL_PROXY_BPB + CONTROL_TRAIN_TIME_MS." >&2
  echo "Or explicitly allow spending for baseline refresh: ALLOW_CONTROL_RERUN=1 LANES='control ...'" >&2
  exit 3
fi
for lane in ${LANES}; do
  idx="$((idx + 1))"
  lane_log=""
  if [[ "${REUSE_DONE}" == "1" ]]; then
    maybe_log="$(latest_lane_log "${lane}")"
    if [[ -n "${maybe_log}" ]] && log_is_done "${maybe_log}" && log_has_target_iteration "${maybe_log}"; then
      lane_log="${maybe_log}"
      proxy_line="$(extract_proxy_line "${lane_log}")"
      step="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*step:\([0-9][0-9]*\)\/[0-9][0-9]*.*/\1/p')"
      proxy_bpb="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*val_bpb:\([0-9.][0-9.]*\).*/\1/p')"
      train_time_ms="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*train_time:\([0-9][0-9]*\)ms.*/\1/p')"
      step_avg_ms="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*step_avg:\([0-9.][0-9.]*\)ms.*/\1/p')"
      smart_stats="$(extract_smart_stats "${lane_log}")"
      [[ -z "${smart_stats}" ]] && smart_stats="NA"
      delta_bpb=""
      [[ -n "${control_bpb}" && -n "${proxy_bpb}" ]] && delta_bpb="$(afloat "${proxy_bpb}" "${control_bpb}")"
      speed_vs_control=""
      [[ -n "${control_train_ms}" && -n "${train_time_ms}" ]] && speed_vs_control="$(aratio "${control_train_ms}" "${train_time_ms}")"
      printf "%s\t%s\tREUSED\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t0\t%s\treused_completed_log\n" \
        "${idx}" "${lane}" "${step}" "${proxy_bpb}" "${delta_bpb}" "${train_time_ms}" \
        "${step_avg_ms}" "${speed_vs_control}" "${smart_stats}" "${lane_log}" >> "${SUMMARY}"
      echo "[${idx}] ${lane} status=REUSED proxy_bpb=${proxy_bpb:-NA} train_time_ms=${train_time_ms:-NA} smart=${smart_stats}"
      continue
    fi
  fi

  run_start_s="$(date +%s)"
  echo "[${idx}] running ${lane}..."

  set +e
  if command -v timeout >/dev/null 2>&1; then
    env \
      ITERATIONS="${ITERATIONS}" \
      SEED="${SEED}" \
      NPROC_PER_NODE="${NPROC}" \
      VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
      SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL}" \
      POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC}" \
      SKIP_QUANT_ROUNDTRIP_EVAL="${SKIP_QUANT_ROUNDTRIP_EVAL}" \
      MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
      TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
      TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}" \
      timeout --signal=TERM --kill-after=30s "${LANE_TIMEOUT_SECONDS}s" \
      bash "${RUN_ONE}" "${lane}" "${PROFILE}" 2>&1 | tee "${BATCH_DIR}/${idx}_${lane}.runner.log"
    rc="${PIPESTATUS[0]}"
  else
    env \
      ITERATIONS="${ITERATIONS}" \
      SEED="${SEED}" \
      NPROC_PER_NODE="${NPROC}" \
      VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
      SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL}" \
      POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC}" \
      SKIP_QUANT_ROUNDTRIP_EVAL="${SKIP_QUANT_ROUNDTRIP_EVAL}" \
      MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
      TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
      TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}" \
      bash "${RUN_ONE}" "${lane}" "${PROFILE}" 2>&1 | tee "${BATCH_DIR}/${idx}_${lane}.runner.log"
    rc="${PIPESTATUS[0]}"
  fi
  set -e

  run_end_s="$(date +%s)"
  elapsed_s="$((run_end_s - run_start_s))"
  lane_log="$(latest_lane_log "${lane}")"
  status="OK"
  note=""
  if [[ "${rc}" -ne 0 ]]; then
    status="FAIL"
    note="runner_exit_${rc}"
    if [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then
      note="${note},lane_timeout_${LANE_TIMEOUT_SECONDS}s"
    fi
  fi

  proxy_line="$(extract_proxy_line "${lane_log}")"
  step="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*step:\([0-9][0-9]*\)\/[0-9][0-9]*.*/\1/p')"
  proxy_bpb="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*val_bpb:\([0-9.][0-9.]*\).*/\1/p')"
  train_time_ms="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*train_time:\([0-9][0-9]*\)ms.*/\1/p')"
  step_avg_ms="$(printf '%s\n' "${proxy_line}" | sed -n 's/.*step_avg:\([0-9.][0-9.]*\)ms.*/\1/p')"
  smart_stats="$(extract_smart_stats "${lane_log}")"
  [[ -z "${smart_stats}" ]] && smart_stats="NA"

  delta_bpb=""
  [[ -n "${control_bpb}" && -n "${proxy_bpb}" ]] && delta_bpb="$(afloat "${proxy_bpb}" "${control_bpb}")"
  speed_vs_control=""
  [[ -n "${control_train_ms}" && -n "${train_time_ms}" ]] && speed_vs_control="$(aratio "${control_train_ms}" "${train_time_ms}")"

  if [[ "${status}" == "OK" && -z "${proxy_bpb}" ]]; then
    status="FAIL"
    if [[ -n "${note}" ]]; then
      note="${note},missing_proxy_line"
    else
      note="missing_proxy_line"
    fi
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${idx}" "${lane}" "${status}" "${step}" "${proxy_bpb}" "${delta_bpb}" \
    "${train_time_ms}" "${step_avg_ms}" "${speed_vs_control}" "${smart_stats}" "${elapsed_s}" "${lane_log}" "${note}" >> "${SUMMARY}"

  echo "[${idx}] ${lane} status=${status} proxy_bpb=${proxy_bpb:-NA} train_time_ms=${train_time_ms:-NA} smart=${smart_stats}"
  if [[ "${status}" == "FAIL" && "${STOP_ON_FAIL}" == "1" ]]; then
    echo "Stopping due to failure (STOP_ON_FAIL=1)"
    break
  fi
done

echo ""
echo "Build-signal summary: ${SUMMARY}"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${SUMMARY}"
else
  cat "${SUMMARY}"
fi

best_lane="$(
  tail -n +2 "${SUMMARY}" \
    | awk -F'\t' '($3=="OK" || $3=="REUSED"){if($5!="" && $7!=""){if(!best || ($7+0)<best_t){best=$2;best_t=$7+0;best_bpb=$5;best_d=$6;best_s=$10}}} END{if(best!="") printf "%s\t%.4f\t%s\t%.0f\t%s", best, best_bpb, (best_d==""?"NA":best_d), best_t, best_s}'
)"
if [[ -n "${best_lane}" ]]; then
  IFS=$'\t' read -r blane bbpb bdelta btime bsmart <<< "${best_lane}"
  echo ""
  echo "Fastest OK lane by train_time_ms: ${blane} (val_bpb=${bbpb}, delta_vs_control=${bdelta}, train_time_ms=${btime}, smart=${bsmart})"
fi
