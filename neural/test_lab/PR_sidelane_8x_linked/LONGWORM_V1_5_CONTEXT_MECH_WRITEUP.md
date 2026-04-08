# Longworm V1.5 Context Mechanism

This variant prioritizes long-context mechanism quality over generic depth expansion.

## Core changes

- Dedicated Longworm trainer path: `neural/experiments/Longworm/train_longworm.py`
- Multi-scale causal context taps inside each block:
  - tap offsets: `16, 64, 256, 1024`
  - optional prefix-mean memory tap
  - learned tap gates + learned per-dim context scale
  - optional RMS normalization on composed context signal
- Hybrid transition integrator:
  - `rk2` on early layers
  - `rk4` on last 4 layers

## Default submission arm

- `35_v1_5_longworm_context_mech_l11_d528_h12_kv4_non_ngram_brotli`
- Shape: `11L x 528d`, heads `12/4`
- Submission-safe defaults:
  - no forced 4K ranking metric
  - primary leaderboard metric remains submission-path `sw_bpb`

## Intent

Shift budget from generic “more layers” scaling into explicit long-context relationship modeling that judges can attribute to a novel, causal mechanism.
