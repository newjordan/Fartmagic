# RESULTS

**PROMOTES**

## Summary
- Hypothesis: relaxing `QUANT_ATTN_BITS` from `5` to `6` on frozen Midnight III.V SP8192 profile improves deployed quality while staying under the byte cap.
- Result: promoted as the current legal SOTA in this repo corpus.
- Delta vs prior documented leader (`PIPELINE.md` 1.10567949): `-0.01408923` bpb.

## Evidence
- seed 444 log: `legs/2026-04-13_midnight_iii_v_qattn6/logs/full_seed444_20260413_044438.log`
  - `gptq:calibrated 0 layers in 2.9s` (line 93)
  - `DIAGNOSTIC post_ema val_bpb:1.1081` (line 95)
  - `Total submission size mixed+brotli: 15576180 bytes` (line 107)
  - `final_quant_roundtrip_exact val_bpb:1.46704988` (line 109)
  - `final_sliding_window_exact val_bpb:1.09159026` (line 111)
- seed 300 confirmation log: `legs/2026-04-13_midnight_iii_v_qattn6/logs/full_seed300_20260413_050254.log`
  - `gptq:calibrated 0 layers in 3.0s` (line 93)
  - `DIAGNOSTIC post_ema val_bpb:1.1087` (line 95)
  - `Total submission size mixed+brotli: 15569556 bytes` (line 107)
  - `final_quant_roundtrip_exact val_bpb:1.47483853` (line 109)
  - `final_sliding_window_exact val_bpb:1.09226433` (line 111)

## Verdict
- PROMOTES
- What to carry forward: Midnight III.V SP8192 + `QUANT_ATTN_BITS=6` as legal deployment lane.
- What to avoid next: seed churn on this exact lane before post-quant sweep legs are isolated and tracked.
