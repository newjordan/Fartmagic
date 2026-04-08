#!/usr/bin/env bash
set -euo pipefail

# Midnight 12L -> Shroud micronized architecture trace (DGX Spark friendly)
# - Runs a short 1-GPU trace job with 12-layer settings.
# - Builds points/edges payload with explicit Midnight 12L architecture tags.
# - Writes latest symlike copies under results/shroud_midnight_12l_latest.*

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
RUN_TAG="${RUN_TAG:-SHROUD_MIDNIGHT_12L}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/shroud_midnight_12l_${TS}}"
mkdir -p "${RESULT_DIR}" logs

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/junkyard/experiments/pr779_asap_test/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
SPEC_PATH="${SPEC_PATH:-${REPO_ROOT}/experiments/shroud_nightcrawler/visualizer/midnight_12l_spec.json}"

TRACE_PATH="${TRACE_PATH:-${RESULT_DIR}/${RUN_TAG}.trace.jsonl}"
LOG_PATH="${LOG_PATH:-${RESULT_DIR}/${RUN_TAG}.log}"
POINTS_PATH="${POINTS_PATH:-${RESULT_DIR}/${RUN_TAG}.trace.points.json}"
FLOW_PATH="${FLOW_PATH:-${RESULT_DIR}/${RUN_TAG}.architecture_flow.json}"

if [[ ! -f "${DATA_PATH}/fineweb_train_000000.bin" || ! -f "${DATA_PATH}/fineweb_val_000000.bin" ]]; then
  echo "missing local smoke data in ${DATA_PATH}" >&2
  echo "expected: fineweb_train_000000.bin and fineweb_val_000000.bin" >&2
  exit 1
fi
if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "missing tokenizer at ${TOKENIZER_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SPEC_PATH}" ]]; then
  echo "missing midnight spec at ${SPEC_PATH}" >&2
  exit 1
fi

echo "============================================"
echo "  SHROUD MIDNIGHT 12L — micronized trace"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RESULT_DIR=${RESULT_DIR}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  SPEC_PATH=${SPEC_PATH}"
echo "============================================"

# 12L micronized run (keeps architecture semantics while reducing runtime/load)
SEED="${SEED}" \
RUN_ID="${RUN_TAG}_${TS}" \
DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-90}" \
ITERATIONS="${ITERATIONS:-40}" \
WARMUP_STEPS="${WARMUP_STEPS:-0}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-40}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-256}" \
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-256}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8192}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
MODEL_DIM="${MODEL_DIM:-256}" \
NUM_LAYERS="${NUM_LAYERS:-12}" \
NUM_HEADS="${NUM_HEADS:-8}" \
NUM_KV_HEADS="${NUM_KV_HEADS:-4}" \
MLP_MULT="${MLP_MULT:-3.0}" \
BIGRAM_DIM="${BIGRAM_DIM:-128}" \
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}" \
ROPE_DIMS="${ROPE_DIMS:-16}" \
XSA_LAST_N="${XSA_LAST_N:-11}" \
USE_CRAWLER=0 \
MTP_NUM_HEADS=0 \
NGRAM_EVAL_ORDER=0 \
CUBRIC_CADENCE=0 \
COMPILE_ENABLED=0 \
COMPILE_FULLGRAPH=0 \
TORCHDYNAMO_OPTIMIZE_DDP=1 \
DDP_FIND_UNUSED_PARAMETERS=1 \
SHROUD_ENABLE=1 \
SHROUD_FLOW_TRACE=1 \
SHROUD_STEP_EVERY=1 \
SHROUD_HEAD_TRACE=1 \
SHROUD_HEAD_MAX_TOKENS=64 \
SHROUD_MAX_EVENTS="${SHROUD_MAX_EVENTS:-600000}" \
SHROUD_TRACE_PATH="${TRACE_PATH}" \
SKIP_GPTQ=1 \
SKIP_EMA=1 \
python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  experiments/shroud_nightcrawler/train_gpt.py \
  2>&1 | tee "${LOG_PATH}"

echo "building Midnight points payload..."
python3 experiments/shroud_nightcrawler/visualizer/build_shroud_points.py \
  --input "${TRACE_PATH}" \
  --output "${POINTS_PATH}" \
  --spec-json "${SPEC_PATH}" \
  --max-points "${MAX_POINTS:-300000}" \
  --max-edges "${MAX_EDGES:-300000}"

echo "building architecture flow graph..."
python3 experiments/shroud_nightcrawler/visualizer/build_architecture_flow.py \
  --input "${TRACE_PATH}" \
  --output "${FLOW_PATH}"

cp -f "${TRACE_PATH}" "${REPO_ROOT}/results/shroud_midnight_12l_latest.trace.jsonl"
cp -f "${POINTS_PATH}" "${REPO_ROOT}/results/shroud_midnight_12l_latest.trace.points.json"
cp -f "${FLOW_PATH}" "${REPO_ROOT}/results/shroud_midnight_12l_latest.architecture_flow.json"
cp -f "${SPEC_PATH}" "${REPO_ROOT}/results/shroud_midnight_12l_latest.spec.json"

echo "============================================"
echo "  DONE"
echo "  trace_jsonl=${TRACE_PATH}"
echo "  points_json=${POINTS_PATH}"
echo "  flow_json=${FLOW_PATH}"
echo "  latest: results/shroud_midnight_12l_latest.*"
echo "============================================"
