# Ablation Log

## Gate (1xGPU export sanity)
- Command:
- Checkpoint path:
- DIAGNOSTIC post_ema:
- final_quant_roundtrip_exact:
- final_sliding_window_exact:
- Verdict: PASS / FAIL

## Full Run (8xH100 export-only, seed=444)
- Command:
- Checkpoint path:
- quant_policy:
- gptq:calibrated:
- DIAGNOSTIC post_ema:
- Total submission size mixed+brotli:
- final_quant_roundtrip_exact val_bpb:
- final_sliding_window_exact val_bpb:
- Verdict: PASS / FAIL

## Confirmation (8xH100 export-only, seed=300)
- final_sliding_window_exact val_bpb:
- Verdict: CONFIRMED / NOT_CONFIRMED
