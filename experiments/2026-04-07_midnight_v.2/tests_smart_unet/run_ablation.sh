#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="${ROOT}/experiments/2026-04-07_midnight_v.2/tests_smart_unet"
TRAIN_SCRIPT="${EXP_DIR}/train_gpt_smart_unet_test.py"
MATRIX="${EXP_DIR}/ablation_matrix.tsv"
LOG_DIR="${EXP_DIR}/logs/ablation_runs"

LANE="${1:-}"
PROFILE="${2:-${AB_PROFILE:-screen}}"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
USER_MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "FATAL: missing train script: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

show_usage() {
  echo "Usage: bash experiments/2026-04-07_midnight_v.2/tests_smart_unet/run_ablation.sh <lane> [profile]"
  echo "Profiles: full | screen | ultra_cheap"
  echo ""
  echo "Available lanes:"
  if [[ -f "${MATRIX}" ]]; then
    awk -F'\t' 'NR==1{next} {printf "  %-22s priority=%s  %s\n", $1, $2, $3}' "${MATRIX}"
  else
    echo "  control"
  fi
}

if [[ -z "${LANE}" || "${LANE}" == "list" || "${LANE}" == "--help" || "${LANE}" == "-h" ]]; then
  show_usage
  exit 0
fi

case "${PROFILE}" in
  full)
    PROFILE_MAX_WALLCLOCK_SECONDS="${USER_MAX_WALLCLOCK_SECONDS:-600}"
    PROFILE_SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
    PROFILE_POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-1}"
    PROFILE_VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}"
    PROFILE_SKIP_QUANT_ROUNDTRIP_EVAL="${SKIP_QUANT_ROUNDTRIP_EVAL:-0}"
    PROFILE_NOTE="full-fidelity"
    ;;
  screen)
    PROFILE_MAX_WALLCLOCK_SECONDS="${USER_MAX_WALLCLOCK_SECONDS:-240}"
    PROFILE_SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
    PROFILE_POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}"
    PROFILE_VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
    PROFILE_SKIP_QUANT_ROUNDTRIP_EVAL="${SKIP_QUANT_ROUNDTRIP_EVAL:-1}"
    PROFILE_NOTE="cheap screen"
    ;;
  ultra_cheap)
    PROFILE_MAX_WALLCLOCK_SECONDS="${USER_MAX_WALLCLOCK_SECONDS:-150}"
    PROFILE_SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
    PROFILE_POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}"
    PROFILE_VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-750}"
    PROFILE_SKIP_QUANT_ROUNDTRIP_EVAL="${SKIP_QUANT_ROUNDTRIP_EVAL:-1}"
    PROFILE_NOTE="minimum-cost smoke"
    ;;
  *)
    echo "FATAL: unknown profile: ${PROFILE}" >&2
    show_usage >&2
    exit 2
    ;;
esac

EXTRA_ENV=()
case "${LANE}" in
  control)
    EXTRA_ENV+=(SMART_UNET_MODE=off)
    ;;
  soft_gating)
    EXTRA_ENV+=(
      SMART_UNET_MODE=soft
      SMART_UNET_QUALITY_THRESHOLD="${SMART_UNET_QUALITY_THRESHOLD:-0.995}"
      SMART_UNET_SKIP_MARGIN="${SMART_UNET_SKIP_MARGIN:-0.004}"
      SMART_UNET_QUALITY_SHARPNESS="${SMART_UNET_QUALITY_SHARPNESS:-24}"
      SMART_UNET_MIN_GATE="${SMART_UNET_MIN_GATE:-0.35}"
    )
    ;;
  hard_routing)
    EXTRA_ENV+=(
      SMART_UNET_MODE=hard
      SMART_UNET_QUALITY_THRESHOLD="${SMART_UNET_QUALITY_THRESHOLD:-0.996}"
      SMART_UNET_SKIP_MARGIN="${SMART_UNET_SKIP_MARGIN:-0.004}"
      SMART_UNET_MAX_DECODER_SKIPS="${SMART_UNET_MAX_DECODER_SKIPS:-4}"
      COMPILE_ENABLED=0
    )
    ;;
  competitive_routing)
    EXTRA_ENV+=(
      SMART_UNET_MODE=competitive
      SMART_UNET_QUALITY_THRESHOLD="${SMART_UNET_QUALITY_THRESHOLD:-0.996}"
      SMART_UNET_SKIP_MARGIN="${SMART_UNET_SKIP_MARGIN:-0.004}"
      SMART_UNET_MAX_DECODER_SKIPS="${SMART_UNET_MAX_DECODER_SKIPS:-4}"
      COMPILE_ENABLED=0
    )
    ;;
  *)
    echo "FATAL: unknown lane: ${LANE}" >&2
    show_usage >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${LANE}_seed${SEED}_${STAMP}.log"

echo "lane=${LANE}"
echo "profile=${PROFILE} (${PROFILE_NOTE})"
echo "seed=${SEED}"
echo "nproc=${NPROC}"
echo "max_wallclock_seconds=${PROFILE_MAX_WALLCLOCK_SECONDS}"
echo "skip_final_eval=${PROFILE_SKIP_FINAL_EVAL}"
echo "post_ema_diagnostic=${PROFILE_POST_EMA_DIAGNOSTIC}"
echo "val_loss_every=${PROFILE_VAL_LOSS_EVERY}"
echo "skip_quant_roundtrip_eval=${PROFILE_SKIP_QUANT_ROUNDTRIP_EVAL}"
echo "log=${LOG_FILE}"
echo "overrides=${EXTRA_ENV[*]}"
echo ""

cd "${ROOT}"
export PYTHONPATH="${ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

env \
  SEED="${SEED}" \
  MAX_WALLCLOCK_SECONDS="${PROFILE_MAX_WALLCLOCK_SECONDS}" \
  SKIP_FINAL_EVAL="${PROFILE_SKIP_FINAL_EVAL}" \
  POST_EMA_DIAGNOSTIC="${PROFILE_POST_EMA_DIAGNOSTIC}" \
  VAL_LOSS_EVERY="${PROFILE_VAL_LOSS_EVERY}" \
  SKIP_QUANT_ROUNDTRIP_EVAL="${PROFILE_SKIP_QUANT_ROUNDTRIP_EVAL}" \
  SKIP_GPTQ=1 \
  COMPRESSOR=brotli \
  NUM_LAYERS=12 \
  QUANT_ATTN_BITS=5 \
  QUANT_MLP_BITS=6 \
  QUANT_AUX_BITS=6 \
  QUANT_EMBED_BITS=8 \
  QUANT_OTHER_BITS=8 \
  LOADER_MODE=coprime \
  COPRIME_MAX_LOADED_SHARDS=1 \
  COPRIME_SHARDS_PER_BATCH=1 \
  COPRIME_SHARD_HOLD_STEPS=64 \
  COMPLEMENT_ALPHA=0 \
  XSA_LAST_N=11 \
  BIGRAM_VOCAB_SIZE=2048 \
  ROPE_DIMS=16 \
  SWA_EVERY=50 \
  MTP_NUM_HEADS=0 \
  TRIGRAM=0 \
  NGRAM_EVAL_ORDER=0 \
  CUBRIC_CADENCE=0 \
  NGRAM_ENTROPY_SHIFT=0 \
  "${EXTRA_ENV[@]}" \
  torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
  2>&1 | tee "${LOG_FILE}"

echo ""
echo "Summary (${LOG_FILE}):"
grep -E "smart_unet:|step:500/|step:1000/|step:2000/|stopping_early|late_qat:enabled|Total submission size mixed\+" "${LOG_FILE}" | tail -60 || true
echo ""
proxy_line="$(grep -E '^step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+' "${LOG_FILE}" | tail -1 || true)"
if [[ -n "${proxy_line}" ]]; then
  echo "Proxy metric (latest train-time val): ${proxy_line}"
fi
