# Promotion Evidence — 2026-04-13_midnight_iii_v_qattn6

## Corpus Paths
- `legs/2026-04-13_midnight_iii_v_qattn6/logs/full_seed444_20260413_044438.log`
- `legs/2026-04-13_midnight_iii_v_qattn6/logs/full_seed300_20260413_050254.log`

## Extracted Metric Lines

```text
$ rg -n "seed:|quant_policy:|gptq:calibrated|DIAGNOSTIC post_ema|Total submission size|final_quant_roundtrip_exact|final_sliding_window_exact" legs/2026-04-13_midnight_iii_v_qattn6/logs/full_seed444_20260413_044438.log legs/2026-04-13_midnight_iii_v_qattn6/logs/full_seed300_20260413_050254.log

.../full_seed444_20260413_044438.log:20:quant_policy:attn=6 mlp=6 aux=6 embed=8 other=8 artifact=final_model.mixed.ptz
.../full_seed444_20260413_044438.log:21:seed:444
.../full_seed444_20260413_044438.log:93:gptq:calibrated 0 layers in 2.9s
.../full_seed444_20260413_044438.log:95:DIAGNOSTIC post_ema val_loss:2.8625 val_bpb:1.1081 eval_time:2320ms
.../full_seed444_20260413_044438.log:107:Total submission size mixed+brotli: 15576180 bytes
.../full_seed444_20260413_044438.log:109:final_quant_roundtrip_exact val_loss:3.78954221 val_bpb:1.46704988
.../full_seed444_20260413_044438.log:111:final_sliding_window_exact val_loss:2.81967914 val_bpb:1.09159026

.../full_seed300_20260413_050254.log:20:quant_policy:attn=6 mlp=6 aux=6 embed=8 other=8 artifact=final_model.mixed.ptz
.../full_seed300_20260413_050254.log:21:seed:300
.../full_seed300_20260413_050254.log:93:gptq:calibrated 0 layers in 3.0s
.../full_seed300_20260413_050254.log:95:DIAGNOSTIC post_ema val_loss:2.8638 val_bpb:1.1087 eval_time:2206ms
.../full_seed300_20260413_050254.log:107:Total submission size mixed+brotli: 15569556 bytes
.../full_seed300_20260413_050254.log:109:final_quant_roundtrip_exact val_loss:3.80966109 val_bpb:1.47483853
.../full_seed300_20260413_050254.log:111:final_sliding_window_exact val_loss:2.82142034 val_bpb:1.09226433
```

## Comparator
- Prior documented leader in `PIPELINE.md` line 6: `1.10567949`.
- New promoted value: `1.09159026`.
- Improvement: `0.01408923` bpb.
