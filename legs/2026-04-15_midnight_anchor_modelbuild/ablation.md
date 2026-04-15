# Ablation Log

## Gate (1xGPU, 2000 steps)
- Command: pending
- `DIAGNOSTIC post_ema`: pending
- Stop message: pending
- Verdict: not run

## Full Run (8xGPU, 600s, seed=444)
- Command: `bash legs/2026-04-15_midnight_anchor_modelbuild/run.sh`
- `DIAGNOSTIC post_ema`: `1.1055`
- Stop message: `anchor:modelbuild_complete export_only=1`
- Checkpoint: `./artifacts/midnight_anchor/final_model_anchor.pt`
- Verdict: ANCHOR_READY
