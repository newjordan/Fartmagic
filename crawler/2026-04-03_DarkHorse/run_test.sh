#!/bin/bash
set -euo pipefail
# Dark Horse — A/B test: base (PR #1296) vs helix on their architecture
#
# Usage:
#   NPROC_PER_NODE=4 bash crawler/2026-04-03_DarkHorse/run_test.sh
#   NPROC_PER_NODE=8 bash crawler/2026-04-03_DarkHorse/run_test.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TS="$(date +%Y%m%d_%H%M%S)"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/darkhorse_summary_${TS}.tsv"

pip install brotli -q 2>/dev/null || true

# Use our SP1024 data — testing architecture, not tokenizer
echo "[preflight] Using SP1024 (our standard data)..."
ls ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l | xargs -I{} echo "  train shards: {}"

# 500 steps, production config
BASE_ENV=(
    SEED="${SEED}"
    ITERATIONS=2000
    MAX_WALLCLOCK_SECONDS=0
    VOCAB_SIZE=1024
    RECUR_LAYERS=4,5
    RECUR_START_STEP=500
    PARALLEL_START_LAYER=7
    TTT_ENABLED=0
)

run_arm() {
    local tag="$1"; shift
    local train_py="$1"; shift
    local logfile="${RESULTS_DIR}/${tag}_s${SEED}_${TS}.log"
    echo ""
    echo "================================================================"
    echo "  ARM: ${tag} — $(date)"
    echo "================================================================"
    env "${BASE_ENV[@]}" "$@" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${train_py}" 2>&1 | tee "${logfile}"
    local bpb=$(grep -oP 'val_bpb:\K[0-9.]+' "${logfile}" | tail -1 2>/dev/null || echo "?")
    local int6=$(grep -oP 'final_int6_sliding_window_exact.*val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local step_ms=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 2>/dev/null || echo "?")
    local params=$(grep -oP 'model_params:\K[0-9]+' "${logfile}" 2>/dev/null || echo "?")
    echo -e "${tag}\t${params}\t${bpb}\t${int6}\t${step_ms}" >> "${SUMMARY}"
    echo "  >>> ${tag}: bpb=${bpb} int6=${int6} step_ms=${step_ms} params=${params}"
}

echo -e "arm\tparams\traw_bpb\tint6_sw_bpb\tstep_ms" > "${SUMMARY}"

echo ""
echo "================================================================"
echo "  DARK HORSE — Helix on SOTA Recursion Base"
echo "  PR #1296 base (SP4096 + recur + parallel + MuonEqR + QK5)"
echo "================================================================"

# A: Their base, unmodified
run_arm "A_base" "${SCRIPT_DIR}/train_gpt_base.py" \
    HELIX=0

# B: Their base + Helix dim=64
run_arm "B_helix_dim64" "${SCRIPT_DIR}/train_gpt_helix.py" \
    HELIX=1 HELIX_DIM=64 HELIX_STRIDE=1

# C: Their base + Helix dim=192
run_arm "C_helix_dim192" "${SCRIPT_DIR}/train_gpt_helix.py" \
    HELIX=1 HELIX_DIM=192 HELIX_STRIDE=1

# D: Their base + Helix dim=64, no depth recurrence (isolate helix from recur)
run_arm "D_helix_norecur" "${SCRIPT_DIR}/train_gpt_helix.py" \
    HELIX=1 HELIX_DIM=64 HELIX_STRIDE=1 RECUR_LAYERS=""

echo ""
echo "================================================================"
echo "  DARK HORSE COMPLETE — $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""
cat "${SUMMARY}" | column -t -s$'\t'
