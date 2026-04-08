#!/usr/bin/env bash
set -euo pipefail

# SLOT LR sweep: 2 arms on existing checkpoint, no training
# Baseline: SLOT_LR=0.005 (Lucky V result: 1.09273413)
# Uses the quantized model from Lucky V run

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../ && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/flash-attention/hopper${PYTHONPATH:+:${PYTHONPATH}}"

pip install brotli 2>/dev/null || true

NPROC="${NPROC_PER_NODE:-8}"
SCRIPT="${ROOT}/neural/experiments/Lucky_V/train_gpt.py"

echo "============================================"
echo "SLOT LR SWEEP — eval only, no training"
echo "Arms: LR=0.010, LR=0.020"
echo "Baseline: LR=0.005 → 1.09273413 BPB"
echo "============================================"

# ARM 1: SLOT_LR=0.010
echo ""
echo ">>> SLOT_LR=0.010"
echo ">>> Started: $(date)"
SLOT_LR=0.010 SKIP_TRAIN=1 python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
    "${SCRIPT}" 2>&1 | grep -E "slot|sliding|val_bpb|exact|eval_time"
echo ">>> DONE: $(date)"

# ARM 2: SLOT_LR=0.020
echo ""
echo ">>> SLOT_LR=0.020"
echo ">>> Started: $(date)"
SLOT_LR=0.020 SKIP_TRAIN=1 python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
    "${SCRIPT}" 2>&1 | grep -E "slot|sliding|val_bpb|exact|eval_time"
echo ">>> DONE: $(date)"

echo ""
echo "============================================"
echo "SWEEP DONE — $(date)"
echo "Compare vs baseline SLOT_LR=0.005 → 1.09273413"
echo "============================================"
