# RESULTS

**UPDATED: 2026-04-14 seed=444 full run (online GPTQ warmdown)**

## Summary
- Hypothesis: online GPTQ Hessian accumulation during training should reduce post-EMA quant gap versus offline-only calibration.
- Result: run used online mode but regressed badly (`final_sliding_window_exact val_bpb=1.85046810`).
- Delta vs leader: negative (exact leader delta not computed in this leg artifact).

## Evidence
- seed 444 log: `legs/2026-04-14_midnight_iii_v_2_3_online_gptq_cpu/logs/full_seed444_20260414_031919.log`
- `max_wallclock_seconds:600.000` (`.../full_seed444_20260414_031919.log:16`)
- `gptq:online mode:online_cpu ...` (`.../full_seed444_20260414_031919.log:23`)
- `torch._dynamo hit config.recompile_limit (8)` (`.../full_seed444_20260414_031919.log:80`, `:94`, `:108`, `:122`)
- `stopping_early: wallclock_cap ... step:433/20000` (`.../full_seed444_20260414_031919.log:137`)
- `gptq:calibrated 66 layers in 0.2s (online)` (`.../full_seed444_20260414_031919.log:140`)
- `DIAGNOSTIC post_ema ... val_bpb:1.8545` (`.../full_seed444_20260414_031919.log:141`)
- `Total submission size mixed+brotli: 8099908 bytes` (`.../full_seed444_20260414_031919.log:149`)
- `final_quant_roundtrip_exact ... val_bpb:2.74505006` (`.../full_seed444_20260414_031919.log:151`)
- `final_sliding_window_exact ... val_bpb:1.85046810` (`.../full_seed444_20260414_031919.log:153`)
- seed 300: not run for this leg configuration.

## Verdict
- DOES NOT PROMOTE.
- What to carry forward: online collection path is active and produces calibrated GPTQ tensors.
- What to avoid next: collector mutation inside compiled forward currently correlates with repeated `recompile_limit` warnings and severe bpb regression.
