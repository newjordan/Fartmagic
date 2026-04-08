#!/usr/bin/env bash
set -euo pipefail

# shroud_nightcrawler — DGX Spark micro-run
# Preserves Nightcrawler's 5F+1C×3+5F architecture shape with reduced dims
# for SHROUD trace capture and data-flow visualization.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
RUN_TAG="${RUN_TAG:-SHROUD_NIGHTCRAWLER}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/shroud_nightcrawler_${TS}}"
mkdir -p "${RESULT_DIR}" logs

DATA_PATH="${DATA_PATH:-/tmp/shroud_loopviz_smoke_data}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
TRACE_PATH="${TRACE_PATH:-${RESULT_DIR}/${RUN_TAG}.trace.jsonl}"
LOG_PATH="${LOG_PATH:-${RESULT_DIR}/${RUN_TAG}.log}"
POINTS_PATH="${POINTS_PATH:-${RESULT_DIR}/${RUN_TAG}.trace.points.json}"
FLOW_PATH="${FLOW_PATH:-${RESULT_DIR}/${RUN_TAG}.architecture_flow.json}"

if [[ ! -f "${DATA_PATH}/fineweb_train_000000.bin" || ! -f "${DATA_PATH}/fineweb_val_000000.bin" ]]; then
  echo "missing smoke data in ${DATA_PATH}; expected fineweb_train_000000.bin and fineweb_val_000000.bin" >&2
  exit 1
fi
if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "missing tokenizer at ${TOKENIZER_PATH}" >&2
  exit 1
fi

echo "============================================"
echo "  SHROUD NIGHTCRAWLER — 5F+1C×3+5F flow viz"
echo "  RUN_TAG=${RUN_TAG}"
echo "  RESULT_DIR=${RESULT_DIR}"
echo "  DATA_PATH=${DATA_PATH}"
echo "============================================"

# --- Training: reduced dims, Nightcrawler 5F shape ---
SEED="${SEED}" \
RUN_ID="${RUN_TAG}_${TS}" \
DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-75}" \
ITERATIONS="${ITERATIONS:-28}" \
WARMUP_STEPS="${WARMUP_STEPS:-0}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-28}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-256}" \
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-256}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
MODEL_DIM="${MODEL_DIM:-256}" \
NUM_LAYERS="${NUM_LAYERS:-6}" \
NUM_HEADS="${NUM_HEADS:-4}" \
NUM_KV_HEADS="${NUM_KV_HEADS:-2}" \
MLP_MULT="${MLP_MULT:-2.0}" \
COMPILE_ENABLED=0 \
COMPILE_FULLGRAPH=0 \
TORCHDYNAMO_OPTIMIZE_DDP=1 \
DDP_FIND_UNUSED_PARAMETERS=1 \
MTP_NUM_HEADS=0 \
XSA_LAST_N=6 \
COMPLEMENT_ALPHA=0 \
NGRAM_EVAL_ORDER=0 \
CUBRIC_CADENCE=0 \
USE_CRAWLER=1 \
NUM_FLAT_LAYERS=5 \
NUM_CRAWLER_LAYERS=1 \
CRAWLER_LOOPS=3 \
CRAWLER_MLP_MULT=6.0 \
INST_DIM=16 \
CRAWLER_QUANT_INT8=0 \
DELTA_NET_HEADS=0 \
SHROUD_ENABLE=1 \
SHROUD_FLOW_TRACE=1 \
SHROUD_STEP_EVERY=1 \
SHROUD_HEAD_TRACE=1 \
SHROUD_HEAD_MAX_TOKENS=64 \
SHROUD_MAX_EVENTS=600000 \
SHROUD_TRACE_PATH="${TRACE_PATH}" \
SKIP_GPTQ=1 \
SKIP_EMA=1 \
python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  experiments/shroud_nightcrawler/train_gpt.py \
  2>&1 | tee "${LOG_PATH}"

# --- Build visualization payloads ---
echo "building point cloud..."
python3 experiments/shroud_nightcrawler/visualizer/build_shroud_points.py \
  --input "${TRACE_PATH}" \
  --output "${POINTS_PATH}" \
  --max-points "${MAX_POINTS:-260000}" \
  --max-edges "${MAX_EDGES:-260000}"

echo "building architecture flow graph..."
python3 experiments/shroud_nightcrawler/visualizer/build_architecture_flow.py \
  --input "${TRACE_PATH}" \
  --output "${FLOW_PATH}"

# --- Copy latest symlinks ---
cp -f "${TRACE_PATH}" "${REPO_ROOT}/results/shroud_nightcrawler_latest.trace.jsonl"
cp -f "${POINTS_PATH}" "${REPO_ROOT}/results/shroud_nightcrawler_latest.trace.points.json"
cp -f "${FLOW_PATH}" "${REPO_ROOT}/results/shroud_nightcrawler_latest.architecture_flow.json"

echo "============================================"
echo "  DONE"
echo "  trace_jsonl=${TRACE_PATH}"
echo "  points_json=${POINTS_PATH}"
echo "  flow_json=${FLOW_PATH}"
echo "  latest: results/shroud_nightcrawler_latest.*"
echo "============================================"
