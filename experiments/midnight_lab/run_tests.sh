#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_DIR="${ROOT}/experiments/midnight_lab"
MATRIX_DEFAULT="${LAB_DIR}/tests.tsv"
MATRIX="${MIDNIGHT_TEST_MATRIX:-${MATRIX_DEFAULT}}"
TRAIN_SCRIPT_DEFAULT="${ROOT}/experiments/midnight/train_gpt.py"
TRAIN_SCRIPT="${MIDNIGHT_TRAIN_SCRIPT:-${TRAIN_SCRIPT_DEFAULT}}"
LOG_ROOT="${LAB_DIR}/logs/runs"

ACTION="${1:-list}"
ARG2="${2:-}"
ARG3="${3:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"

show_usage() {
  cat <<'USAGE'
Usage:
  bash experiments/midnight_lab/run_tests.sh list
  bash experiments/midnight_lab/run_tests.sh run <test_id>
  bash experiments/midnight_lab/run_tests.sh batch [priority_max]

Behavior:
  - Test registry source: experiments/midnight_lab/tests.tsv
  - Baseline train script: experiments/midnight/train_gpt.py
  - Forces LOADER_MODE=sequential
  - Writes logs + summary.tsv to experiments/midnight_lab/logs/runs/<run_stamp>/

Optional env:
  - MIDNIGHT_TEST_MATRIX=<absolute_path> to select a non-default test registry
  - MIDNIGHT_TRAIN_SCRIPT=<absolute_path> to override train script
  - SEED, NPROC_PER_NODE, COMPILE_ENABLED, SKIP_GPTQ, COMPRESSOR, STOP_ON_FAIL
USAGE
}

if [[ ! -f "${MATRIX}" ]]; then
  echo "FATAL: missing test registry: ${MATRIX}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "FATAL: missing train script: ${TRAIN_SCRIPT}" >&2
  echo "Set MIDNIGHT_TRAIN_SCRIPT or ensure experiments/midnight/train_gpt.py exists." >&2
  exit 1
fi

list_tests() {
  echo "Registry: ${MATRIX}"
  echo "Train script: ${TRAIN_SCRIPT}"
  awk -F'\t' '
    NR==1 {next}
    {
      printf "%-42s p=%s lane=%-26s profile=%-10s max_s=%-4s %s\n", $1, $2, $3, $4, $5, $6
    }
  ' "${MATRIX}"
}

row_for_test() {
  local id="${1}"
  awk -F'\t' -v id="${id}" 'NR>1 && $1==id {print; exit}' "${MATRIX}"
}

set_profile_defaults() {
  local profile="${1}"
  local max_s_from_row="${2:-}"
  case "${profile}" in
    full)
      PROFILE_MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-${max_s_from_row:-600}}"
      PROFILE_SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
      PROFILE_POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-1}"
      PROFILE_VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}"
      PROFILE_SKIP_QUANT_ROUNDTRIP_EVAL="${SKIP_QUANT_ROUNDTRIP_EVAL:-0}"
      PROFILE_NOTE="full-fidelity"
      ;;
    screen)
      PROFILE_MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-${max_s_from_row:-240}}"
      PROFILE_SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
      PROFILE_POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}"
      PROFILE_VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
      PROFILE_SKIP_QUANT_ROUNDTRIP_EVAL="${SKIP_QUANT_ROUNDTRIP_EVAL:-1}"
      PROFILE_NOTE="screen"
      ;;
    ultra_cheap)
      PROFILE_MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-${max_s_from_row:-150}}"
      PROFILE_SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
      PROFILE_POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}"
      PROFILE_VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-750}"
      PROFILE_SKIP_QUANT_ROUNDTRIP_EVAL="${SKIP_QUANT_ROUNDTRIP_EVAL:-1}"
      PROFILE_NOTE="ultra_cheap"
      ;;
    *)
      echo "FATAL: unknown profile '${profile}'" >&2
      exit 2
      ;;
  esac
}

lane_overrides() {
  local lane="${1}"
  local -n ref="${2}"
  ref=()
  case "${lane}" in
    control)
      ;;
    warmdown_3000)
      ref+=(WARMDOWN_ITERS=3000)
      ;;
    warmdown_2500)
      ref+=(WARMDOWN_ITERS=2500)
      ;;
    warmdown_4200)
      ref+=(WARMDOWN_ITERS=4200)
      ;;
    xsa_12)
      ref+=(XSA_LAST_N=12)
      ;;
    xsa_10)
      ref+=(XSA_LAST_N=10)
      ;;
    ve_10_11)
      ref+=(VE_LAYERS=10,11)
      ;;
    ve_8_9_10)
      ref+=(VE_LAYERS=8,9,10)
      ;;
    adaptive_embed_25_6)
      ref+=(
        ADAPTIVE_EMBED_PRECISION=1
        ADAPTIVE_EMBED_KEEP_FRAC=0.25
        ADAPTIVE_EMBED_LOW_BITS=6
      )
      ;;
    adaptive_embed_40_6)
      ref+=(
        ADAPTIVE_EMBED_PRECISION=1
        ADAPTIVE_EMBED_KEEP_FRAC=0.40
        ADAPTIVE_EMBED_LOW_BITS=6
      )
      ;;
    attn6)
      ref+=(QUANT_ATTN_BITS=6)
      ;;
    adaptive_embed_25_6_attn6)
      ref+=(
        ADAPTIVE_EMBED_PRECISION=1
        ADAPTIVE_EMBED_KEEP_FRAC=0.25
        ADAPTIVE_EMBED_LOW_BITS=6
        QUANT_ATTN_BITS=6
      )
      ;;
    *)
      echo "FATAL: unknown lane '${lane}' in tests.tsv" >&2
      exit 2
      ;;
  esac
}

extract_val_bpb() {
  local line="${1}"
  printf '%s\n' "${line}" | sed -n 's/.*val_bpb:\([0-9.][0-9.]*\).*/\1/p'
}

extract_step() {
  local line="${1}"
  printf '%s\n' "${line}" | sed -n 's/.*step:\([0-9][0-9]*\)\/20000.*/\1/p'
}

extract_train_ms() {
  local line="${1}"
  printf '%s\n' "${line}" | sed -n 's/.*train_time:\([0-9][0-9]*\)ms.*/\1/p'
}

append_summary() {
  local summary="${1}"
  local test_id="${2}"
  local lane="${3}"
  local profile="${4}"
  local status="${5}"
  local step="${6}"
  local proxy_bpb="${7}"
  local final_bpb="${8}"
  local roundtrip_bpb="${9}"
  local size_bytes="${10}"
  local train_ms="${11}"
  local log_file="${12}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${test_id}" "${lane}" "${profile}" "${SEED}" "${status}" "${step}" "${proxy_bpb}" \
    "${final_bpb}" "${roundtrip_bpb}" "${size_bytes}" "${train_ms}" "${log_file}" \
    >> "${summary}"
}

run_one() {
  local test_id="${1}"
  local run_stamp="${2}"
  local row
  row="$(row_for_test "${test_id}")"
  if [[ -z "${row}" ]]; then
    echo "FATAL: test_id not found: ${test_id}" >&2
    exit 2
  fi

  local id priority lane profile max_s description env_overrides
  IFS=$'\t' read -r id priority lane profile max_s description env_overrides <<< "${row}"
  set_profile_defaults "${profile}" "${max_s}"

  local lane_env=()
  lane_overrides "${lane}" lane_env

  local matrix_env=()
  if [[ -n "${env_overrides}" && "${env_overrides}" != "(none)" ]]; then
    read -r -a matrix_env <<< "${env_overrides}"
  fi

  local run_dir="${LOG_ROOT}/${run_stamp}"
  mkdir -p "${run_dir}"
  local file_stamp
  file_stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="${run_dir}/${test_id}_seed${SEED}_${file_stamp}.log"
  local summary="${run_dir}/summary.tsv"
  if [[ ! -f "${summary}" ]]; then
    printf "test_id\tlane\tprofile\tseed\tstatus\tstep\tproxy_val_bpb\tfinal_exact_bpb\troundtrip_exact_bpb\tsize_bytes\ttrain_time_ms\tlog_file\n" > "${summary}"
  fi

  echo "test_id=${test_id}"
  echo "priority=${priority}"
  echo "description=${description}"
  echo "train_script=${TRAIN_SCRIPT}"
  echo "lane=${lane}"
  echo "profile=${profile} (${PROFILE_NOTE})"
  echo "seed=${SEED}"
  echo "nproc=${NPROC}"
  echo "max_wallclock_seconds=${PROFILE_MAX_WALLCLOCK_SECONDS}"
  echo "loader_mode=sequential"
  echo "log=${log_file}"
  if [[ "${#lane_env[@]}" -gt 0 ]]; then
    echo "lane_overrides=${lane_env[*]}"
  else
    echo "lane_overrides=(none)"
  fi
  if [[ "${#matrix_env[@]}" -gt 0 ]]; then
    echo "matrix_overrides=${matrix_env[*]}"
  else
    echo "matrix_overrides=(none)"
  fi
  echo ""

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    append_summary "${summary}" "${test_id}" "${lane}" "${profile}" "DRYRUN" "" "" "" "" "" "" "${log_file}"
    return 0
  fi

  cd "${ROOT}"
  export PYTHONPATH="${ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
  export TMPDIR="${TMPDIR:-/tmp}"

  set +e
  env \
    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS="${PROFILE_MAX_WALLCLOCK_SECONDS}" \
    SKIP_FINAL_EVAL="${PROFILE_SKIP_FINAL_EVAL}" \
    POST_EMA_DIAGNOSTIC="${PROFILE_POST_EMA_DIAGNOSTIC}" \
    VAL_LOSS_EVERY="${PROFILE_VAL_LOSS_EVERY}" \
    SKIP_QUANT_ROUNDTRIP_EVAL="${PROFILE_SKIP_QUANT_ROUNDTRIP_EVAL}" \
    SKIP_GPTQ="${SKIP_GPTQ:-1}" \
    COMPRESSOR="${COMPRESSOR:-brotli}" \
    COMPILE_ENABLED="${COMPILE_ENABLED:-0}" \
    NUM_LAYERS="${NUM_LAYERS:-12}" \
    QUANT_ATTN_BITS="${QUANT_ATTN_BITS:-5}" \
    QUANT_MLP_BITS="${QUANT_MLP_BITS:-6}" \
    QUANT_AUX_BITS="${QUANT_AUX_BITS:-6}" \
    QUANT_EMBED_BITS="${QUANT_EMBED_BITS:-8}" \
    QUANT_OTHER_BITS="${QUANT_OTHER_BITS:-8}" \
    LOADER_MODE=sequential \
    COPRIME_MAX_LOADED_SHARDS="${COPRIME_MAX_LOADED_SHARDS:-1}" \
    COPRIME_SHARDS_PER_BATCH="${COPRIME_SHARDS_PER_BATCH:-1}" \
    COPRIME_SHARD_HOLD_STEPS="${COPRIME_SHARD_HOLD_STEPS:-64}" \
    COMPLEMENT_ALPHA="${COMPLEMENT_ALPHA:-0}" \
    XSA_LAST_N="${XSA_LAST_N:-11}" \
    BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}" \
    ROPE_DIMS="${ROPE_DIMS:-16}" \
    SWA_EVERY="${SWA_EVERY:-50}" \
    MTP_NUM_HEADS="${MTP_NUM_HEADS:-0}" \
    TRIGRAM="${TRIGRAM:-0}" \
    NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-0}" \
    CUBRIC_CADENCE="${CUBRIC_CADENCE:-0}" \
    NGRAM_ENTROPY_SHIFT="${NGRAM_ENTROPY_SHIFT:-0}" \
    "${lane_env[@]}" \
    "${matrix_env[@]}" \
    torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
    2>&1 | tee "${log_file}"
  local rc="${PIPESTATUS[0]}"
  set -e

  local status="OK"
  if [[ "${rc}" != "0" ]]; then
    status="FAIL(${rc})"
  fi

  local proxy_line final_line roundtrip_line size_line
  proxy_line="$(grep -E '^step:[0-9]+/20000 val_loss:[0-9.]+ val_bpb:[0-9.]+' "${log_file}" | tail -1 || true)"
  final_line="$(grep -E 'final_sliding_window_exact' "${log_file}" | tail -1 || true)"
  roundtrip_line="$(grep -E 'final_quant_roundtrip_exact' "${log_file}" | tail -1 || true)"
  size_line="$(grep -E 'Total submission size mixed\+zlib:' "${log_file}" | tail -1 || true)"

  local step proxy_bpb final_bpb roundtrip_bpb size_bytes train_ms
  step="$(extract_step "${proxy_line}")"
  proxy_bpb="$(extract_val_bpb "${proxy_line}")"
  final_bpb="$(extract_val_bpb "${final_line}")"
  roundtrip_bpb="$(extract_val_bpb "${roundtrip_line}")"
  size_bytes="$(printf '%s\n' "${size_line}" | sed -n 's/.*: \([0-9][0-9]*\) bytes.*/\1/p')"
  train_ms="$(extract_train_ms "${proxy_line}")"

  append_summary "${summary}" "${test_id}" "${lane}" "${profile}" "${status}" "${step}" "${proxy_bpb}" "${final_bpb}" "${roundtrip_bpb}" "${size_bytes}" "${train_ms}" "${log_file}"

  echo ""
  echo "Summary (${test_id}):"
  grep -E 'loader:|step:500/|step:1000/|stopping_early|late_qat:enabled|Total submission size mixed\+zlib|final_quant_roundtrip_exact|final_sliding_window_exact' "${log_file}" | tail -40 || true
  echo ""

  return "${rc}"
}

run_batch() {
  local priority_max="${1}"
  local run_stamp="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
  local stop_on_fail="${STOP_ON_FAIL:-1}"

  mapfile -t ids < <(awk -F'\t' -v p="${priority_max}" 'NR>1 && ($2+0) <= (p+0) {print $1}' "${MATRIX}")
  if [[ "${#ids[@]}" -eq 0 ]]; then
    echo "FATAL: no tests selected for PRIORITY_MAX=${priority_max}" >&2
    exit 2
  fi

  echo "run_stamp=${run_stamp}"
  echo "priority_max=${priority_max}"
  echo "selected_tests=${#ids[@]}"
  echo "log_dir=${LOG_ROOT}/${run_stamp}"
  echo ""

  local id rc
  for id in "${ids[@]}"; do
    echo "=== Running ${id} ==="
    set +e
    run_one "${id}" "${run_stamp}"
    rc="$?"
    set -e
    if [[ "${rc}" != "0" && "${stop_on_fail}" == "1" ]]; then
      echo "STOP_ON_FAIL=1 -> aborting batch after ${id}" >&2
      return "${rc}"
    fi
  done
}

case "${ACTION}" in
  list|--list)
    list_tests
    ;;
  run)
    if [[ -z "${ARG2}" ]]; then
      echo "FATAL: run requires <test_id>" >&2
      show_usage >&2
      exit 2
    fi
    run_one "${ARG2}" "${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
    ;;
  batch)
    run_batch "${ARG2:-${PRIORITY_MAX:-1}}"
    ;;
  help|-h|--help)
    show_usage
    ;;
  *)
    echo "FATAL: unknown action '${ACTION}'" >&2
    show_usage >&2
    exit 2
    ;;
esac
