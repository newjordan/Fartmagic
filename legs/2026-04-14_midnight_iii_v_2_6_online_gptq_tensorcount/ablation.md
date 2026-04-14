# Ablation Log

## Gate (optional sanity only, 1xGPU, 2000 steps)
- Command:
- Seed:
- Best proxy metric:
- Verdict: PASS / FAIL
- Notes:

## Full Run (4xH100, 1200s, seed=444)
- Command:
- steps reached before wallclock cap:
- `recompile_limit` warnings:
- final_sliding_window_exact val_bpb:
- Delta vs leader:
- Verdict: PROMOTION_CANDIDATE / FAIL

## Confirmation (4xH100, 1200s, seed=300)
- final_sliding_window_exact val_bpb:
- Verdict: CONFIRMED / NOT_CONFIRMED
