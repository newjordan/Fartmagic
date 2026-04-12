# Ablation Log

## v1 Run (8xH100, 600s, seed=444) — BROKEN GPTQ
- Command: `bash neural/legs/2026-04-11_iii_loop/run.sh`
- pre-quant val_bpb: 1.1348 (post_ema)
- quant_roundtrip val_bpb: 1.53989427
- GPTQ tensors: 0 / 4 naive — Hessians collected for 2 layers but never matched banks
- Verdict: **INVALID** — GPTQ plumbing broken, all weights fell through to naive quant
- Fix: v2 patches gptq_calibrate (F.linear interception for bank slices),
  mixed_quantize_gptq (per-slice GPTQ for 3D banks), dequantize_mixed_quant (3D scale)

## Full Run (8xH100, 600s, seed=444)
- Command:
- final_sliding_window_exact val_bpb:
- Delta vs leader:
- Verdict: PROMOTION_CANDIDATE / FAIL

## Confirmation (8xH100, 600s, seed=300)
- final_sliding_window_exact val_bpb:
- Verdict: CONFIRMED / NOT_CONFIRMED
