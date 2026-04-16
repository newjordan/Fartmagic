#!/bin/bash
# whale fwd TMA leg: opt-in TMA forward. Parity with non-TMA at headline; 2us gain.
export WHALE_FWD_VARIANT="${WHALE_FWD_VARIANT:-default}"   # set to 'tma' to enable
export WHALE_BWD_VARIANT=fused_delta
export WHALE_FUSED_DELTA_T_MAX=3072
