# Ablation Log

## v1 Run (8xH100, 600s, seed=444) — BROKEN GPTQ
- Command: `bash neural/legs/2026-04-11_iii_loop/run.sh`
- pre-quant val_bpb: 1.1348 (post_ema)
- quant_roundtrip val_bpb: 1.53989427
- GPTQ tensors: 0 / 4 naive — Hessians collected for 2 layers but never matched banks
- Verdict: **INVALID** — GPTQ plumbing broken, all weights fell through to naive quant
- Fix: v2 patches gptq_calibrate (F.linear interception for bank slices),
  mixed_quantize_gptq (per-slice GPTQ for 3D banks), dequantize_mixed_quant (3D scale)

## v2 Run (8xH100, 600s, seed=444) — PARTIAL GPTQ
- Command: `bash neural/legs/2026-04-11_iii_loop/run.sh` (after git pull 01e6d1a)
- Steps: 3894 @ 146.44ms avg
- pre-quant (post_ema) val_bpb: 1.1344
- quant_roundtrip_exact val_bpb: 1.53449871 (barely improved — only 12/72 slices GPTQ'd)
- final_sliding_window_exact val_bpb: **1.11039111**
- Delta vs leader: +0.00471 (WORSE)
- GPTQ tensors: 12 / 60 naive — autocast `.to(x.dtype)` creates new tensor, data_ptr mismatch
- Verdict: **PARTIAL** — GPTQ working for some slices but autocast breaks data_ptr matching
- Fix: v3 removes autocast from calibration so `.to(x.dtype)` is a no-op in float32

## Full Run (8xH100, 600s, seed=444)
- Command:
- final_sliding_window_exact val_bpb:
- Delta vs leader:
- Verdict: PROMOTION_CANDIDATE / FAIL

## Confirmation (8xH100, 600s, seed=300)
- final_sliding_window_exact val_bpb:
- Verdict: CONFIRMED / NOT_CONFIRMED
