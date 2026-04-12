# Hypothesis — Midnight_Rider

## Concept: Progressive Depth with Per-Layer Quant Policy

Train with 12 core layers at full speed, then activate late layers at 70% wallclock.
Late layers use gentler quantization (int7 or int8) to avoid quant sensitivity from
undertrained weights. Core layers keep int5/int6 as before.

## Parent
- Source: `vault/train_gpt_midnight_iii_base.py`
- SHA256: `4d265579556279e3b0d652abf078fe762117227cd2408c9eca1afd81bdb15365`

## Architecture
- Late layers appended after the U-net decoder (no skip connections)
- Zero-init output projections → start as identity (no-op residual)
- Activation similar to `ENABLE_LOOPING_AT` mechanism
- Per-layer quant policy: core layers int5/int6, late layers int7 or int8

## New Env Vars
- `LATE_LAYERS` (int, default 0) — extra layers to activate late
- `LATE_ACTIVATE_AT` (float, default 0.7) — wallclock fraction for activation
- `LATE_QUANT_ATTN_BITS` (int, default 8) — quant bits for late layer attn weights
- `LATE_QUANT_MLP_BITS` (int, default 8) — quant bits for late layer MLP weights

## Ablation Arms

| Arm | Late Layers | Late Quant | Est. Artifact | Est. Steps |
|-----|-------------|------------|---------------|------------|
| control | 0 (12L only) | — | ~12.3MB | ~5131 |
| 2L_int8 | +2 | int8 | ~14.9MB | ~4900 |
| 2L_int7 | +2 | int7 | ~14.3MB | ~4900 |
| 3L_int8 | +3 | int8 | ~16.2MB | ~4820 |
| 3L_int7 | +3 | int7 | ~15.3MB | ~4820 |

## Gate Pass Criteria
- Any arm with final_sliding_window_exact < 1.11325 (iii_lean baseline) is a win
- Artifact size must stay under 16,000,000 bytes
- Step parity: late activation should not crash step_avg above 160ms
