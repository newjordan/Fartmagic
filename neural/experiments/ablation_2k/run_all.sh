#!/usr/bin/env bash
set -euo pipefail

# Ablation series: 3 arms, 2k steps each, 4 GPUs, seed 444
# Control = Lucky IV baseline data (already collected)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../ && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/flash-attention/hopper${PYTHONPATH:+:${PYTHONPATH}}"
export SEED=444
export ITERATIONS=2000
export WARMDOWN_ITERS=0
export SKIP_FINAL_EVAL=1
export POST_EMA_DIAGNOSTIC=1
export COPRIME_MAX_LOADED_SHARDS=1
export COPRIME_SHARDS_PER_BATCH=1

# Ensure brotli is available
pip install brotli 2>/dev/null || true

NPROC="${NPROC_PER_NODE:-4}"

echo "============================================"
echo "ABLATION SERIES — $(date)"
echo "Arms: qk4, depth_recur, muoneq_r"
echo "Steps: 2000 | GPUs: ${NPROC} | Seed: ${SEED}"
echo "============================================"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# --- ARM 1: QK4 (QK_GAIN_INIT=4.0) ---
echo ""
echo ">>> ARM 1/3: qk4 (QK_GAIN_INIT=4.0)"
echo ">>> Started: $(date)"
python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
    "${SCRIPT_DIR}/qk4/train_gpt.py" 2>&1 | tee "${SCRIPT_DIR}/qk4/log_seed444.txt"
echo ">>> ARM 1 DONE: $(date)"
echo ""

# --- ARM 2: DEPTH_RECUR (layers 4,5 fire twice) ---
echo ">>> ARM 2/3: depth_recur (RECUR_LAYERS=4,5)"
echo ">>> Started: $(date)"
python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
    "${SCRIPT_DIR}/depth_recur/train_gpt.py" 2>&1 | tee "${SCRIPT_DIR}/depth_recur/log_seed444.txt"
echo ">>> ARM 2 DONE: $(date)"
echo ""

# --- ARM 3: MUONEQ_R (row-normalize before Newton-Schulz) ---
echo ">>> ARM 3/3: muoneq_r (MuonEq-R row normalization)"
echo ">>> Started: $(date)"
python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
    "${SCRIPT_DIR}/muoneq_r/train_gpt.py" 2>&1 | tee "${SCRIPT_DIR}/muoneq_r/log_seed444.txt"
echo ">>> ARM 3 DONE: $(date)"
echo ""

echo "============================================"
echo "ALL ARMS COMPLETE — $(date)"
echo "Logs:"
echo "  qk4:         ${SCRIPT_DIR}/qk4/log_seed444.txt"
echo "  depth_recur: ${SCRIPT_DIR}/depth_recur/log_seed444.txt"
echo "  muoneq_r:    ${SCRIPT_DIR}/muoneq_r/log_seed444.txt"
echo "============================================"
