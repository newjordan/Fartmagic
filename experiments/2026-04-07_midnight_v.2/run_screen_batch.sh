#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT}/experiments/2026-04-07_midnight_v.2"
RUN_ONE="${EXP_DIR}/run_ablation.sh"
MATRIX="${EXP_DIR}/ablation_matrix.tsv"
OUT_ROOT="${EXP_DIR}/logs/screen_batches"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
PROFILE="${PROFILE:-screen}"
PRIORITY_MAX="${PRIORITY_MAX:-1}"
BUDGET_MINUTES="${BUDGET_MINUTES:-0}"
SCREEN_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-}"
CONTROL_AT_END="${CONTROL_AT_END:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
DRY_RUN="${DRY_RUN:-0}"

# If combo arm doesn't look promising from parent signals, skip it to save cost.
GATE_COMBO="${GATE_COMBO:-1}"
COMBO_GATE_MARGIN="${COMBO_GATE_MARGIN:-0.0000}"

if [[ ! -x "${RUN_ONE}" ]]; then
  echo "FATAL: missing run script: ${RUN_ONE}" >&2
  exit 1
fi
if [[ ! -f "${MATRIX}" ]]; then
  echo "FATAL: missing matrix file: ${MATRIX}" >&2
  exit 1
fi

if [[ "${PROFILE}" != "screen" && "${PROFILE}" != "ultra_cheap" ]]; then
  echo "FATAL: PROFILE must be screen|ultra_cheap for economical batch runs (got ${PROFILE})" >&2
  exit 2
fi

mapfile -t LANES < <(awk -F'\t' -v pmax="${PRIORITY_MAX}" 'NR>1 && ($2+0) <= (pmax+0) {print $1}' "${MATRIX}")
if [[ "${#LANES[@]}" -eq 0 ]]; then
  echo "FATAL: no lanes selected from matrix with PRIORITY_MAX=${PRIORITY_MAX}" >&2
  exit 2
fi

ORDERED=()
ORDERED+=("control")
for lane in "${LANES[@]}"; do
  [[ "${lane}" == "control" ]] && continue
  ORDERED+=("${lane}")
done
if [[ "${CONTROL_AT_END}" == "1" ]]; then
  ORDERED+=("control")
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="${OUT_ROOT}/${STAMP}"
mkdir -p "${BATCH_DIR}"
SUMMARY="${BATCH_DIR}/summary.tsv"

printf "idx\tlane\tstatus\tstep\tproxy_val_bpb\tdelta_vs_control\ttrain_time_ms\telapsed_s\tlane_log\tbatch_log\tnote\n" > "${SUMMARY}"

echo "batch_dir=${BATCH_DIR}"
echo "profile=${PROFILE}"
echo "seed=${SEED}"
echo "nproc=${NPROC}"
echo "priority_max=${PRIORITY_MAX}"
echo "budget_minutes=${BUDGET_MINUTES}"
echo "dry_run=${DRY_RUN}"
echo "stop_on_fail=${STOP_ON_FAIL}"
echo "gate_combo=${GATE_COMBO} margin=${COMBO_GATE_MARGIN}"
if [[ -n "${SCREEN_WALLCLOCK_SECONDS}" ]]; then
  echo "override_max_wallclock_seconds=${SCREEN_WALLCLOCK_SECONDS}"
else
  echo "override_max_wallclock_seconds=(profile default)"
fi
echo "lane_order=${ORDERED[*]}"
echo ""

batch_start_s="$(date +%s)"
control_bpb=""
declare -A lane_proxy_bpb=()

append_row() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${1}" "${2}" "${3}" "${4}" "${5}" "${6}" "${7}" "${8}" "${9}" "${10}" "${11}" \
    >> "${SUMMARY}"
}

float_delta() {
  awk -v a="${1}" -v b="${2}" 'BEGIN {printf "%.8f", (a - b)}'
}

is_lt() {
  awk -v a="${1}" -v b="${2}" 'BEGIN {exit !(a < b)}'
}

extract_proxy_line() {
  local f="${1}"
  awk '
    {
      if (match($0, /^step:[0-9]+\/20000 val_loss:[0-9.]+ val_bpb:[0-9.]+/)) {
        line=$0
      }
    }
    END {
      if (line != "") print line;
    }
  ' "${f}"
}

for i in "${!ORDERED[@]}"; do
  lane="${ORDERED[$i]}"
  idx="$((i + 1))"

  now_s="$(date +%s)"
  if [[ "${BUDGET_MINUTES}" != "0" ]]; then
    elapsed_budget=$((now_s - batch_start_s))
    if (( elapsed_budget >= BUDGET_MINUTES * 60 )); then
      note="budget_reached"
      append_row "${idx}" "${lane}" "SKIPPED" "" "" "" "" "" "" "" "${note}"
      echo "[${idx}/${#ORDERED[@]}] ${lane} -> SKIPPED (${note})"
      continue
    fi
  fi

  if [[ "${GATE_COMBO}" == "1" && "${lane}" == "adaptive_embed_25_6_attn6" && -n "${control_bpb}" ]]; then
    a="${lane_proxy_bpb[adaptive_embed_25_6]:-}"
    b="${lane_proxy_bpb[attn6]:-}"
    run_combo=0
    if [[ -n "${a}" ]]; then
      gate_a="$(awk -v c="${control_bpb}" -v m="${COMBO_GATE_MARGIN}" 'BEGIN {printf "%.8f", c - m}')"
      if is_lt "${a}" "${gate_a}"; then run_combo=1; fi
    fi
    if [[ -n "${b}" ]]; then
      gate_b="$(awk -v c="${control_bpb}" -v m="${COMBO_GATE_MARGIN}" 'BEGIN {printf "%.8f", c - m}')"
      if is_lt "${b}" "${gate_b}"; then run_combo=1; fi
    fi
    if [[ "${run_combo}" == "0" ]]; then
      note="combo_gated_off_no_parent_improvement"
      append_row "${idx}" "${lane}" "SKIPPED" "" "" "" "" "" "" "" "${note}"
      echo "[${idx}/${#ORDERED[@]}] ${lane} -> SKIPPED (${note})"
      continue
    fi
  fi

  run_start_s="$(date +%s)"
  batch_log="${BATCH_DIR}/${idx}_${lane}.runner.log"

  echo "[${idx}/${#ORDERED[@]}] Running ${lane} (${PROFILE})..."
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1 -> skipping execution" | tee "${batch_log}"
    elapsed_s=0
    note="dry_run"
    append_row "${idx}" "${lane}" "DRYRUN" "" "" "" "" "${elapsed_s}" "" "${batch_log}" "${note}"
    continue
  fi

  set +e
  if [[ -n "${SCREEN_WALLCLOCK_SECONDS}" ]]; then
    SEED="${SEED}" NPROC_PER_NODE="${NPROC}" MAX_WALLCLOCK_SECONDS="${SCREEN_WALLCLOCK_SECONDS}" \
      bash "${RUN_ONE}" "${lane}" "${PROFILE}" 2>&1 | tee "${batch_log}"
    rc="${PIPESTATUS[0]}"
  else
    SEED="${SEED}" NPROC_PER_NODE="${NPROC}" \
      bash "${RUN_ONE}" "${lane}" "${PROFILE}" 2>&1 | tee "${batch_log}"
    rc="${PIPESTATUS[0]}"
  fi
  set -e

  run_end_s="$(date +%s)"
  elapsed_s="$((run_end_s - run_start_s))"

  lane_log="$(awk -F'=' '/^log=/{print $2; exit}' "${batch_log}")"
  if [[ -z "${lane_log}" ]]; then
    lane_log="(unknown)"
  fi

  status="OK"
  note=""
  if [[ "${rc}" -ne 0 ]]; then
    status="FAIL"
    note="runner_exit_${rc}"
  fi

  step=""
  proxy_bpb=""
  delta=""
  train_time_ms=""

  if [[ -f "${lane_log}" ]]; then
    proxy_line="$(extract_proxy_line "${lane_log}")"
    if [[ -n "${proxy_line}" ]]; then
      step="$(awk 'match($0,/step:([0-9]+)\/20000/,m){print m[1]}' <<< "${proxy_line}")"
      proxy_bpb="$(awk 'match($0,/val_bpb:([0-9.]+)/,m){print m[1]}' <<< "${proxy_line}")"
      train_time_ms="$(awk 'match($0,/train_time:([0-9]+)ms/,m){print m[1]}' <<< "${proxy_line}")"
    else
      note="${note:+${note};}no_proxy_line"
    fi
  else
    note="${note:+${note};}missing_lane_log"
  fi

  if [[ -z "${control_bpb}" && "${lane}" == "control" && -n "${proxy_bpb}" ]]; then
    control_bpb="${proxy_bpb}"
  fi
  if [[ -n "${proxy_bpb}" ]]; then
    lane_proxy_bpb["${lane}"]="${proxy_bpb}"
  fi
  if [[ -n "${control_bpb}" && -n "${proxy_bpb}" ]]; then
    delta="$(float_delta "${proxy_bpb}" "${control_bpb}")"
  fi

  append_row "${idx}" "${lane}" "${status}" "${step}" "${proxy_bpb}" "${delta}" "${train_time_ms}" "${elapsed_s}" "${lane_log}" "${batch_log}" "${note}"

  echo "[${idx}/${#ORDERED[@]}] ${lane} done status=${status} proxy_bpb=${proxy_bpb:-NA} delta_vs_control=${delta:-NA} elapsed=${elapsed_s}s"

  if [[ "${status}" == "FAIL" && "${STOP_ON_FAIL}" == "1" ]]; then
    echo "Stopping batch due to failure (STOP_ON_FAIL=1)"
    break
  fi
done

echo ""
echo "Batch summary: ${SUMMARY}"
echo "Sorted by proxy_val_bpb (lower is better):"
{
  head -n 1 "${SUMMARY}"
  tail -n +2 "${SUMMARY}" | awk -F'\t' '$3=="OK" && $5!=""' | sort -t$'\t' -k5,5n
} | column -t -s $'\t' || cat "${SUMMARY}"
