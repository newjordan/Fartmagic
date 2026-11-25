# Ablation Log

## Gate (1xGPU, 2000 steps)
- Command:
- Seed:
- Best proxy metric:
- Verdict: PASS / FAIL
- Notes: no gate log is stored in this leg.

## Full Run (8xH100, 600s, seed=444)
- Command:
- final_sliding_window_exact val_bpb:
- Delta vs leader:
- Verdict: PROMOTION_CANDIDATE / FAIL
- Notes: an earlier draft of this file cited `legs/2026-04-14_midnight_iii_v_2_3_online_gptq_cpu/logs/full_seed444_20260414_031919.log`. That log is not in this checkout, and it is not a run of this leg.

## Confirmation (8xH100, 600s, seed=300)
- final_sliding_window_exact val_bpb:
- Verdict: CONFIRMED / NOT_CONFIRMED
