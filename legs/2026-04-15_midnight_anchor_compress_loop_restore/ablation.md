# Ablation Log

## Gate (1xGPU export sanity)
- Command: not run
- Checkpoint path:
- `init_model:loop_state_restored`:
- DIAGNOSTIC post_ema:
- final_quant_roundtrip_exact:
- final_sliding_window_exact:
- Verdict: not run

## Full Run (8xH100 export-only, seed=444)
- Command: `bash legs/2026-04-15_midnight_anchor_compress_loop_restore/run.sh`
- Checkpoint path: `/workspace/SOTA_FINAL/artifacts/midnight_anchor/final_model_anchor.pt`
- quant_policy: `attn=6 mlp=6 aux=6 embed=8 other=8`
- `init_model:loop_state_restored`: `looping:1`
- gptq:calibrated: `66 layers`
- DIAGNOSTIC post_ema: `1.1055`
- Total submission size mixed+brotli: `15450196`
- final_quant_roundtrip_exact val_bpb: `1.12205231`
- final_sliding_window_exact val_bpb: `1.10506462`
- Verdict: PASS

## Confirmation (8xH100 export-only, seed=300)
- final_sliding_window_exact val_bpb:
- Verdict: CONFIRMED / NOT_CONFIRMED
