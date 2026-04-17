#!/usr/bin/env bash
# Fixed validation settings for the combined run.
export WHALE_BWD_VARIANT=fused_delta_tma
export WHALE_FUSED_DELTA_T_MAX=3072
export WHALE_BWD_KV_TMA_MAXNREG=224
export CUDA_VISIBLE_DEVICES=0

export BENCH_SHAPE="2,2048,8,4,64"
export BENCH_BACKENDS="whale_fast,fa3"
export BENCH_ROUNDS=10
export BENCH_ITERS=200

export DKDV_WINNER_CONFIG="128,128,8,3"
export DQ_WINNER_CONFIG="128,64,8,5"
export DQ_WINNER_MAXNREG="224"

export WHALE_BWD_KV_TMA_CONFIG="${DKDV_WINNER_CONFIG}"
export WHALE_BWD_Q_CONFIG="${DQ_WINNER_CONFIG}"
export WHALE_BWD_Q_MAXNREG="${DQ_WINNER_MAXNREG}"
