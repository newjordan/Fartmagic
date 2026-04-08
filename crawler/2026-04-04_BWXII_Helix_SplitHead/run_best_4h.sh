#!/bin/bash
set -euo pipefail

# Launch best validated arm for a 4-hour run.
#
# Usage:
#   # Auto-pick from latest closeness summary:
#   NPROC_PER_NODE=8 bash crawler/2026-04-04_BWXII_Helix_SplitHead/run_best_4h.sh
#
#   # Force a specific arm from closeness matrix (H0/H1/H2):
#   NPROC_PER_NODE=8 ARM=H1 bash crawler/2026-04-04_BWXII_Helix_SplitHead/run_best_4h.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

SEED="${SEED:-300}"
NPROC="${NPROC_PER_NODE:-8}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-14400}"
WARMDOWN_ITERS="${WARMDOWN_ITERS:-12000}"
ITERATIONS="${ITERATIONS:-200000}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
TARGET_BPB="${TARGET_BPB:-1.1364}"

pick_latest_summary() {
    find "${SCRIPT_DIR}/results" -maxdepth 2 -type f -name "summary.tsv" 2>/dev/null | sort | tail -1
}

ARM="${ARM:-}"
SUMMARY="${SUMMARY:-$(pick_latest_summary || true)}"

if [[ -z "${ARM}" ]]; then
    if [[ -z "${SUMMARY}" || ! -f "${SUMMARY}" ]]; then
        echo "ERROR: no summary.tsv found; set ARM=H0|H1|H2 or run run_closeness_matrix.sh first."
        exit 2
    fi
    ARM="$(python3 - <<PY
import csv
from pathlib import Path
p = Path("${SUMMARY}")
rows = []
with p.open() as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        g = row.get("gap_to_ouroboros", "")
        try:
            row["_abs_gap"] = abs(float(g))
        except Exception:
            continue
        rows.append(row)
if not rows:
    print("")
else:
    rows.sort(key=lambda x: x["_abs_gap"])
    print(rows[0]["arm"])
PY
)"
fi

if [[ -z "${ARM}" ]]; then
    echo "ERROR: unable to determine ARM."
    exit 2
fi

case "${ARM}" in
    H0)
        HELIX=0
        CRAWLER_LOOPS=2
        CRAWLER_CROSS_HEADS=0
        HELIX_DIM=0
        MUON_WD=0.04
        ;;
    H1)
        HELIX=1
        CRAWLER_LOOPS=2
        CRAWLER_CROSS_HEADS=4
        HELIX_DIM=192
        MUON_WD=0.12
        ;;
    H2)
        HELIX=1
        CRAWLER_LOOPS=2
        CRAWLER_CROSS_HEADS=4
        HELIX_DIM=192
        MUON_WD=0.04
        ;;
    *)
        echo "ERROR: unknown ARM=${ARM}. Expected H0/H1/H2."
        exit 2
        ;;
esac

OUT_DIR="${SCRIPT_DIR}/results"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/run4h_${ARM}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "4H RUN ARM=${ARM}"
echo "target(Ouroboros)=${TARGET_BPB}"
echo "HELIX=${HELIX} CRAWLER_LOOPS=${CRAWLER_LOOPS} CRAWLER_CROSS_HEADS=${CRAWLER_CROSS_HEADS} HELIX_DIM=${HELIX_DIM} MUON_WD=${MUON_WD}"
echo "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS} ITERATIONS=${ITERATIONS}"
echo "LOG=${LOG}"
echo "============================================"

env \
    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
    ITERATIONS="${ITERATIONS}" \
    WARMDOWN_ITERS="${WARMDOWN_ITERS}" \
    COMPLEMENT_ALPHA=0 \
    XSA_LAST_N=11 \
    BIGRAM_VOCAB_SIZE=2048 \
    ROPE_DIMS=16 \
    SWA_EVERY=50 \
    MTP_NUM_HEADS=0 \
    LATE_QAT_THRESHOLD=0 \
    MATRIX_LR=0.03 \
    EMBED_LR=0.035 \
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
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=1 \
    QK_GAIN_INIT=4.0 \
    GPTQ_CAL_SAMPLES=256 \
    GPTQ_CAL_SEQ_LEN=2048 \
    MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_CHOKE_DIM=0 \
    CRAWLER_MLP_CHOKE_SHAPE=flat \
    CRAWLER_MLP_CHOKE_GROUPS=8 \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
    CRAWLER_LOOP_SMEAR=0 \
    CRAWLER_TAP_DIM=0 \
    CRAWLER_TAP_LOOP_SPECIFIC=1 \
    CRAWLER_TAP_LAYERS=all \
    ANCHOR_DIM=0 \
    FLAT_WEIGHT_SHARE=0 \
    HELIX="${HELIX}" \
    HELIX_DIM="${HELIX_DIM}" \
    HELIX_STRIDE=1 \
    CRAWLER_CROSS_HEADS="${CRAWLER_CROSS_HEADS}" \
    MUON_WD="${MUON_WD}" \
    "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
    2>&1 | tee "${LOG}"

echo ""
echo "4h run complete: ${LOG}"

