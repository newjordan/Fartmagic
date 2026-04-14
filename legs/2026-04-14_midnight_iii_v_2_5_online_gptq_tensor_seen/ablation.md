# Ablation Log

## Gate (1xGPU, 2000 steps)
- Command:
- Seed:
- Best proxy metric:
- Verdict: PASS / FAIL
- Notes:

## Full Run (8xH100, 600s, seed=444)
- Command: `cd /workspace/parameter-golf_testlab && NPROC_PER_NODE=4 bash legs/2026-04-14_midnight_iii_v_2_3_online_gptq_cpu/run.sh`
- Profile evidence: `max_wallclock_seconds:600.000` (`legs/2026-04-14_midnight_iii_v_2_3_online_gptq_cpu/logs/full_seed444_20260414_031919.log:16`)
- Online mode evidence: `gptq:online mode:online_cpu every_steps:4 max_rows:4096 require_looping:1` (`.../full_seed444_20260414_031919.log:23`)
- Compile instability evidence: `torch._dynamo hit config.recompile_limit (8)` (`.../full_seed444_20260414_031919.log:80`, `:94`, `:108`, `:122`)
- stop evidence: `stopping_early ... step:433/20000` (`.../full_seed444_20260414_031919.log:137`)
- `gptq:calibrated 66 layers in 0.2s (online)` (`.../full_seed444_20260414_031919.log:140`)
- DIAGNOSTIC post_ema val_bpb: `1.8545` (`.../full_seed444_20260414_031919.log:141`)
- Total submission size mixed+brotli: `8099908 bytes` (`.../full_seed444_20260414_031919.log:149`)
- final_quant_roundtrip_exact val_bpb: `2.74505006` (`.../full_seed444_20260414_031919.log:151`)
- final_sliding_window_exact val_bpb: `1.85046810` (`.../full_seed444_20260414_031919.log:153`)
- Delta vs leader: regresses heavily
- Verdict: FAIL

## Confirmation (8xH100, 600s, seed=300)
- final_sliding_window_exact val_bpb:
- Verdict: CONFIRMED / NOT_CONFIRMED
