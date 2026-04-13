# Hypothesis — 2026-04-13_midnight_iii_v_qattn6

Parent: legs/2026-04-12_midnight_iii_v_submission/train_gpt.py

## One Variable
- Name: `QUANT_ATTN_BITS`
- New value: `6`
- Baseline value: `5`

## Why
- Keep the winning Midnight III.V SP8192 training profile fixed and test whether a 1-bit attn quant relaxation improves post-quant quality while staying legal on bytes.

## Pass Criteria
- `Total submission size mixed+brotli` remains below `16,000,000` bytes.
- `final_sliding_window_exact` is at or better than current legal leader.
- `final_quant_roundtrip_exact` improves vs the parent qattn5 profile.
