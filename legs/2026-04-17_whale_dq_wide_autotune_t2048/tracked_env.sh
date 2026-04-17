#!/usr/bin/env bash
# Fixed env. WHALE_BWD_Q_CONFIG varies per run. WHALE_BWD_KV_TMA_CONFIG is
# populated from the dkdv sweep winner (sourced by the queue runner).
export WHALE_BWD_VARIANT=fused_delta_tma
export WHALE_FUSED_DELTA_T_MAX=3072
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/SOTA_FINAL:${PYTHONPATH:-}
