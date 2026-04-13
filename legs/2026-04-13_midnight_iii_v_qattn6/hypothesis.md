# Hypothesis — 2026-04-13_midnight_iii_v_qattn6

Parent: legs/2026-04-12_midnight_iii_v_submission/train_gpt.py

## One Variable
- `QUANT_ATTN_BITS`: `5 -> 6`

## Why
- Prior III.V run showed large quantization roundtrip degradation while artifact size remained comfortably below cap.
- Increasing attention quant bits by one step is the narrowest compression-relaxation change to reduce roundtrip loss without changing architecture/data path.

## Pass Criteria
- Improved `final_quant_roundtrip_exact` vs parent III.V baseline.
- Submission size remains below cap.
- No regression in final sliding-window competitiveness beyond expected noise.
