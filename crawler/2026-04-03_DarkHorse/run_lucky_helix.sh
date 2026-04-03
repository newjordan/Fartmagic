#!/bin/bash
set -euo pipefail
# Dark Horse — Helix on Lucky IV (our 1.0963 SOTA)
# Cheap MLP crawler, not full transformer block
#
# Usage:
#   NPROC_PER_NODE=8 bash crawler/2026-04-03_DarkHorse/run_lucky_helix.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
TS="$(date +%Y%m%d_%H%M%S)"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

RESULTS_DIR="${SCRIPT_DIR}/results/lucky_helix"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary_${TS}.tsv"

run_arm() {
    local tag="$1"; shift
    local train_py="$1"; shift
    local logfile="${RESULTS_DIR}/${tag}_s${SEED}_${TS}.log"
    echo ""
    echo "================================================================"
    echo "  ARM: ${tag} — $(date)"
    echo "================================================================"
    env SEED="${SEED}" "$@" \
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
echo "  DARK HORSE — Helix on Lucky IV (1.0963 BPB)"
echo "  Cheap MLP crawler, ${NPROC} GPUs, 2k steps"
echo "================================================================"

# A: Lucky IV base — no helix
run_arm "A_lucky_base" "${SCRIPT_DIR}/train_gpt_lucky_base.py" \
    HELIX=0 ITERATIONS=2000 MAX_WALLCLOCK_SECONDS=0

# B: Lucky + helix dim=64 stride=2 (conservative)
run_arm "B_helix_d64_s2" "${SCRIPT_DIR}/train_gpt_lucky_helix.py" \
    HELIX=1 HELIX_DIM=64 HELIX_STRIDE=2 ITERATIONS=2000 MAX_WALLCLOCK_SECONDS=0

# C: Lucky + helix dim=128 stride=2
run_arm "C_helix_d128_s2" "${SCRIPT_DIR}/train_gpt_lucky_helix.py" \
    HELIX=1 HELIX_DIM=128 HELIX_STRIDE=2 ITERATIONS=2000 MAX_WALLCLOCK_SECONDS=0

# D: Lucky + helix dim=64 stride=1 (every layer)
run_arm "D_helix_d64_s1" "${SCRIPT_DIR}/train_gpt_lucky_helix.py" \
    HELIX=1 HELIX_DIM=64 HELIX_STRIDE=1 ITERATIONS=2000 MAX_WALLCLOCK_SECONDS=0

echo ""
echo "================================================================"
echo "  LUCKY HELIX COMPLETE — $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""
cat "${SUMMARY}" | column -t -s$'\t'
