#!/usr/bin/env bash
LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WHALE_BWD_VARIANT=fused_delta_tma
export WHALE_FUSED_DELTA_T_MAX=3072

export BENCH_SHAPE="2,2048,8,4,64"
export BENCH_BACKENDS="whale_fast,fa3"
export BENCH_PROBE_ROUNDS=5
export BENCH_PROBE_ITERS=100
export BENCH_VALIDATE_ROUNDS=10
export BENCH_VALIDATE_ITERS=200

export CROSS_GPU_WORKER_GPUS="${CROSS_GPU_WORKER_GPUS:-0 1}"
export CROSS_GPU_CACHE_ROOT="/tmp/triton_cache_whale_cross_gpu"
export CROSS_GPU_STATE_DIR="state"
export CROSS_GPU_LOG_DIR="logs"
export CROSS_GPU_EVIDENCE_DIR="evidence"
export CROSS_GPU_ANCHOR_FILE="${LEG_DIR}/winner_anchor.json"

anchor_json() {
  jq -r "$1" "${CROSS_GPU_ANCHOR_FILE}"
}

export CROSS_GPU_WINNER_STATUS="$(anchor_json '.status')"
export CROSS_GPU_WINNER_FAMILY_ID="$(anchor_json '.family_id')"
export CROSS_GPU_WINNER_DQ_TILE="$(anchor_json '.dq_tile')"
export CROSS_GPU_WINNER_DKDV_TILE="$(anchor_json '.dkdv_tile')"
export CROSS_GPU_WINNER_DKDV_MAXNREG="$(anchor_json '.dkdv_maxnreg')"
export CROSS_GPU_WINNER_QMR="$(anchor_json '.qmr')"
export CROSS_GPU_WINNER_SOURCE_EVIDENCE="$(anchor_json '.source_evidence')"
export CROSS_GPU_WINNER_MULTI_GPU_READY="$(anchor_json '.downstream_ready.multi_gpu')"
export CROSS_GPU_WINNER_DONOR_READY="$(anchor_json '.downstream_ready.whale_donor')"

export CROSS_GPU_RIDGE_SWEEP_QMRS="1984"
export CROSS_GPU_VALLEY_QMRS="1984"
export CROSS_GPU_CHALLENGER_ROWS=$'fallback|128,64,8,2|128,128,8,3|256|928
hybrid_a|128,64,8,2|128,64,8,3|256|2048
hybrid_b|128,64,8,5|128,128,8,3|256|1952 2192'
export CROSS_GPU_BENCH_SCRIPT="/workspace/SOTA_FINAL/legs/2026-04-17_whale_2gpu_pruned_search_runtime/bench_stable_scratch.py"
