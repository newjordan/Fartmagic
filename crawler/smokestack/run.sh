#!/bin/bash
set -euo pipefail
# ================================================================
# Smokestack — Full 8xH100 600s run
#
# Run AFTER gate passes. Set CRAWLER_LOOPS to the winning arm value.
#
# Usage:
#   SEED=444 CRAWLER_LOOPS=2 NPROC_PER_NODE=8 bash experiments/smokestack/run.sh
#   SEED=300 CRAWLER_LOOPS=2 NPROC_PER_NODE=8 bash experiments/smokestack/run.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
LEGAL_SIZE_LIMIT="${LEGAL_SIZE_LIMIT:-16000000}"

# The ONE variable — set via env
CRAWLER_LOOPS="${CRAWLER_LOOPS:?ERROR: must set CRAWLER_LOOPS to winning gate value (1 or 2)}"

# Derive rope scales from loop count
case "${CRAWLER_LOOPS}" in
    1) ROPE_SCALES="9" ;;
    2) ROPE_SCALES="9,1" ;;
    3) ROPE_SCALES="9,1,1" ;;
    *) echo "ERROR: unexpected CRAWLER_LOOPS=${CRAWLER_LOOPS}"; exit 1 ;;
esac

mkdir -p "${SCRIPT_DIR}/results"
LOG="${SCRIPT_DIR}/results/fullrun_loops${CRAWLER_LOOPS}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

# Preflight
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || { echo "  ERROR: zstandard missing"; exit 1; }

echo "[preflight] checking flash_attn..."
python3 - <<'PY'
try:
    import flash_attn_interface
    print("  FA3 (hopper) OK")
except Exception:
    try:
        import flash_attn
        v = flash_attn.__version__
        if str(v).startswith("3"):
            print(f"  FA3 v{v} OK")
        else:
            print(f"  WARNING: flash-attn v{v} detected (want v3)")
    except Exception:
        raise SystemExit("  ERROR: flash-attn not importable")
PY

echo "[preflight] checking dataset + tokenizer..."
python3 - <<'PY'
import glob, os
tok = "./data/tokenizers/fineweb_1024_bpe.model"
assert os.path.isfile(tok), f"missing tokenizer: {tok}"
shards = glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
assert len(shards) >= 8, f"need >=8 train shards, found {len(shards)}"
print(f"  tokenizer OK, train shards={len(shards)}")
PY

echo ""
echo "============================================"
echo "  Smokestack — Full Run"
echo "  seed=${SEED} GPUs=${NPROC} wallclock=600s"
echo "  CRAWLER_LOOPS=${CRAWLER_LOOPS}  ROPE_SCALES=${ROPE_SCALES}"
echo "  log: ${LOG}"
echo "============================================"
echo ""

env \
    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS=600 \
    WARMDOWN_ITERS=2000 \
    COMPLEMENT_ALPHA=0 \
    XSA_LAST_N=11 \
    BIGRAM_VOCAB_SIZE=2048 \
    ROPE_DIMS=16 \
    SWA_EVERY=50 \
    MTP_NUM_HEADS=0 \
    LATE_QAT_THRESHOLD=0 \
    MATRIX_LR=0.03 \
    TORCHDYNAMO_OPTIMIZE_DDP=0 \
    COMPILE_FULLGRAPH=1 \
    NGRAM_EVAL_ORDER=0 \
    MODEL_DIM=512 \
    USE_CRAWLER=1 \
    NUM_FLAT_LAYERS=9 \
    NUM_CRAWLER_LAYERS=1 \
    CRAWLER_LOOPS="${CRAWLER_LOOPS}" \
    CRAWLER_MLP_MULT=6.0 \
    INST_DIM=32 \
    CRAWLER_QUANT_INT8=0 \
    DELTA_NET_HEADS=0 \
    SKIP_EMA=1 \
    SKIP_GPTQ=1 \
    LOOP_AWARE_GPTQ=0 \
    MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_CHOKE_DIM=0 \
    CRAWLER_MLP_CHOKE_SHAPE=flat \
    CRAWLER_MLP_CHOKE_GROUPS=8 \
    CRAWLER_LOOP_ROPE_SCALES="${ROPE_SCALES}" \
    CRAWLER_LOOP_SMEAR=0 \
    CRAWLER_TAP_DIM=0 \
    CRAWLER_TAP_LOOP_SPECIFIC=1 \
    CRAWLER_TAP_LAYERS=all \
    ANCHOR_DIM=0 \
    FLAT_WEIGHT_SHARE=0 \
    NPROC_PER_NODE="${NPROC}" \
    "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
    2>&1 | tee "${LOG}"

# Metrics extraction
raw_bpb="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || true)"
int6_sw_bpb="$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || true)"
bytes_total="$(grep -oP 'Total submission size int6\+(?:zstd|zlib): \K[0-9]+' "${LOG}" | tail -1 || true)"
step_ms="$(grep -oP 'step_avg:\K[0-9.]+' "${LOG}" | tail -1 || true)"
steps="$(grep -oP 'stopping_early:.*step:\K[0-9]+' "${LOG}" | tail -1 || true)"
if [[ -z "${steps}" ]]; then
    steps="$(grep -oP 'step:\K[0-9]+(?=/[0-9]+ val_loss:)' "${LOG}" | tail -1 || true)"
fi

artifact_ok="unknown"
if [[ -n "${bytes_total}" && "${bytes_total}" =~ ^[0-9]+$ ]]; then
    if (( bytes_total <= LEGAL_SIZE_LIMIT )); then
        artifact_ok="yes"
    else
        artifact_ok="no"
    fi
fi

# Save artifacts
if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
    cp -f "${REPO_ROOT}/final_model.pt" "${SCRIPT_DIR}/results/fullrun_loops${CRAWLER_LOOPS}_s${SEED}.final_model.pt"
fi
if [[ -f "${REPO_ROOT}/final_model.int6.ptz" ]]; then
    cp -f "${REPO_ROOT}/final_model.int6.ptz" "${SCRIPT_DIR}/results/fullrun_loops${CRAWLER_LOOPS}_s${SEED}.int6.ptz"
fi

echo ""
echo "============================================"
echo "  RESULT — Smokestack loops=${CRAWLER_LOOPS} seed=${SEED}"
echo "  raw_bpb:       ${raw_bpb:-?}"
echo "  int6_sw_bpb:   ${int6_sw_bpb:-?}"
echo "  step_avg_ms:   ${step_ms:-?}"
echo "  steps:         ${steps:-?}"
echo "  bytes_total:   ${bytes_total:-?}  (limit ${LEGAL_SIZE_LIMIT})"
echo "  artifact_legal:${artifact_ok}"
echo "  vs BWX 9F:     1.13867894 (target to beat)"
echo "  log:           ${LOG}"
echo "============================================"

if [[ "${artifact_ok}" == "no" ]]; then
    echo "WARNING: artifact exceeds ${LEGAL_SIZE_LIMIT} bytes!"
    exit 2
fi

exit 0
