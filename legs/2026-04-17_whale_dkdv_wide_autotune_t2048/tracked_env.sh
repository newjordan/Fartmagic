#!/usr/bin/env bash
# Fixed env (varies nothing across the sweep). WHALE_BWD_KV_TMA_CONFIG is set
# per-run by sweep_dkdv.sh.
export WHALE_BWD_VARIANT=fused_delta_tma
export WHALE_FUSED_DELTA_T_MAX=3072
export WHALE_BWD_KV_TMA_MAXNREG=224
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/SOTA_FINAL:${PYTHONPATH:-}
