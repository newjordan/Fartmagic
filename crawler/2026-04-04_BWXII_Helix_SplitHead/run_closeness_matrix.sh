#!/bin/bash
set -euo pipefail

# Validate whether Helix SplitHead can close the gap to Ouroboros.
# No preflight checks here by design: this is a fast, pod-native runner.
#
# Usage:
#   NPROC_PER_NODE=8 MAX_WALLCLOCK_SECONDS=600 bash crawler/2026-04-04_BWXII_Helix_SplitHead/run_closeness_matrix.sh
#   NPROC_PER_NODE=8 MAX_WALLCLOCK_SECONDS=900 ARM_FILTER=H1,H2 bash crawler/2026-04-04_BWXII_Helix_SplitHead/run_closeness_matrix.sh

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
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
WARMDOWN_ITERS="${WARMDOWN_ITERS:-2000}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SCRIPT_DIR}/results/closeness_${TS}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
TARGET_BPB="${TARGET_BPB:-1.1364}"  # Ouroboros submission value

echo -e "arm\tseed\twallclock_s\traw_bpb\tint6_sw_bpb\tgap_to_ouroboros\tstep_ms\tsteps\tbytes_total\tlog" > "${SUMMARY}"

run_arm() {
    local arm="$1"
    local helix="$2"
    local loops="$3"
    local cross="$4"
    local hdim="$5"
    local muon_wd="$6"

    local log="${OUT_DIR}/${arm}_seed${SEED}.log"
    echo ""
    echo "=================================================="
    echo "ARM ${arm}"
    echo "  HELIX=${helix} CRAWLER_LOOPS=${loops} CRAWLER_CROSS_HEADS=${cross} HELIX_DIM=${hdim} MUON_WD=${muon_wd}"
    echo "  target(Ouroboros)=${TARGET_BPB}"
    echo "  log=${log}"
    echo "=================================================="

    env \
        SEED="${SEED}" \
        MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
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
        CRAWLER_LOOPS="${loops}" \
        CRAWLER_MLP_MULT=6.0 \
        INST_DIM=32 \
        CRAWLER_QUANT_INT8=0 \
        DELTA_NET_HEADS=0 \
        SKIP_EMA=1 \
        SKIP_GPTQ=0 \
        LOOP_AWARE_GPTQ=1 \
        QK_GAIN_INIT=4.0 \
        GPTQ_CAL_SAMPLES=128 \
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
        HELIX="${helix}" \
        HELIX_DIM="${hdim}" \
        HELIX_STRIDE=1 \
        CRAWLER_CROSS_HEADS="${cross}" \
        MUON_WD="${muon_wd}" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"

    local raw_bpb int6_sw_bpb step_ms steps bytes_total gap
    raw_bpb="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || true)"
    int6_sw_bpb="$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || true)"
    step_ms="$(grep -oP 'step_avg:\K[0-9.]+' "${log}" | tail -1 || true)"
    steps="$(grep -oP 'stopping_early:.*step:\K[0-9]+' "${log}" | tail -1 || true)"
    if [[ -z "${steps}" ]]; then
        steps="$(grep -oP 'step:\K[0-9]+(?=/[0-9]+ val_loss:)' "${log}" | tail -1 || true)"
    fi
    bytes_total="$(grep -oP 'Total submission size int6\+(?:brotli|zlib): \K[0-9]+' "${log}" | tail -1 || true)"
    gap="$(python3 - <<PY
target=${TARGET_BPB}
val='${int6_sw_bpb}'
try:
    v=float(val)
    print(f"{v-target:+.6f}")
except Exception:
    print("NA")
PY
)"

    echo -e "${arm}\t${SEED}\t${MAX_WALLCLOCK_SECONDS}\t${raw_bpb:-?}\t${int6_sw_bpb:-?}\t${gap}\t${step_ms:-?}\t${steps:-?}\t${bytes_total:-?}\t${log}" >> "${SUMMARY}"
}

ARM_FILTER="${ARM_FILTER:-}"
want_arm() {
    local arm="$1"
    if [[ -z "${ARM_FILTER}" ]]; then
        return 0
    fi
    [[ ",${ARM_FILTER}," == *",${arm},"* ]]
}

# H0: Ouroboros-like control (no helix, 2-loop cadence)
if want_arm "H0"; then
    run_arm "H0" 0 2 0 0 0.04
fi

# H1: Helix splithead with Ouroboros cadence (2 loops) and high WD
if want_arm "H1"; then
    run_arm "H1" 1 2 4 192 0.12
fi

# H2: Helix splithead with Ouroboros cadence (2 loops) and standard WD
if want_arm "H2"; then
    run_arm "H2" 1 2 4 192 0.04
fi

echo ""
echo "==== SUMMARY (${SUMMARY}) ===="
column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"

echo ""
echo "Best arm by smallest absolute gap_to_ouroboros:"
python3 - <<PY
import csv
from pathlib import Path
path = Path("${SUMMARY}")
rows = []
with path.open() as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        g = row["gap_to_ouroboros"]
        try:
            row["_abs_gap"] = abs(float(g))
        except Exception:
            continue
        rows.append(row)
if not rows:
    print("No valid rows.")
else:
    rows.sort(key=lambda x: x["_abs_gap"])
    b = rows[0]
    print(f"{b['arm']} int6_sw_bpb={b['int6_sw_bpb']} gap={b['gap_to_ouroboros']} log={b['log']}")
PY

