#!/usr/bin/env bash
# Tracked env for whale_bwd_tma. Opt-in knob for the TMA variant of the
# dkdv inline-delta kernel. Not part of default dispatch.
export WHALE_BWD_VARIANT="${WHALE_BWD_VARIANT:-fused_delta_tma}"
export WHALE_FUSED_DELTA_T_MAX="${WHALE_FUSED_DELTA_T_MAX:-1048576}"
