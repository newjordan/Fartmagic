# RESULTS

**UPDATED: 2026-04-14 seed=444 checkpoint eval**

## Summary
- Hypothesis: Run quantization-only checkpoint evaluation (`ITERATIONS=0`, `WARMUP_STEPS=0`) from a shape-matched SP8192 reference checkpoint.
- Result: `post_ema val_bpb=1.4420`, `final_quant_roundtrip_exact val_bpb=1.46565755`, `final_sliding_window_exact val_bpb=1.42387173`.
- Delta vs leader: TBD (leader reference metric not recorded in this leg artifact).

## Evidence
- seed 444 log: `legs/2026-04-14_midnight_iii_v_2_2_ckpt_eval/logs/full_seed444_20260414_024259.log`
- `quant_policy:attn=6 mlp=6 aux=6 embed=8 other=8 artifact=final_model.mixed.ptz` (`.../full_seed444_20260414_024259.log:22`)
- `gptq:calibrated 66 layers in 2.9s` (`.../full_seed444_20260414_024259.log:29`)
- `DIAGNOSTIC post_ema val_loss:3.7250 val_bpb:1.4420` (`.../full_seed444_20260414_024259.log:30`)
- `Total submission size mixed+brotli: 15582015 bytes` (`.../full_seed444_20260414_024259.log:38`)
- `final_quant_roundtrip_exact val_loss:3.78594567 val_bpb:1.46565755` (`.../full_seed444_20260414_024259.log:40`)
- `final_sliding_window_exact val_loss:3.67799308 val_bpb:1.42387173` (`.../full_seed444_20260414_024259.log:42`)
- seed 300: not run in this leg.

## Verdict
- DOES NOT PROMOTE yet (missing seed-300 confirmation).
- What to carry forward: shape-matched init checkpoint and `COMPILE_FULLGRAPH=0` path are stable for post-EMA GPTQ diagnostics.
- What to avoid next: mismatched checkpoint/tokenizer pairs (seen earlier as embedding and bank shape mismatch with inflated bpb).
