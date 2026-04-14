# Ablation Log

## Gate (1xGPU, 2000 steps)
- Command:
- Seed:
- Best proxy metric:
- Verdict: PASS / FAIL
- Notes:

## Full Run (8xH100, 600s, seed=444)
- Command: `cd /workspace/parameter-golf_testlab && NPROC_PER_NODE=4 bash legs/2026-04-14_midnight_iii_v_2_2_ckpt_eval/run.sh`
- quant_policy: `attn=6 mlp=6 aux=6 embed=8 other=8 artifact=final_model.mixed.ptz` (`legs/2026-04-14_midnight_iii_v_2_2_ckpt_eval/logs/full_seed444_20260414_024259.log:22`)
- gptq:calibrated: `66 layers in 2.9s` (`legs/2026-04-14_midnight_iii_v_2_2_ckpt_eval/logs/full_seed444_20260414_024259.log:29`)
- DIAGNOSTIC post_ema: `val_loss:3.7250 val_bpb:1.4420 eval_time:8231ms` (`legs/2026-04-14_midnight_iii_v_2_2_ckpt_eval/logs/full_seed444_20260414_024259.log:30`)
- Total submission size mixed+brotli: `15582015 bytes` (`legs/2026-04-14_midnight_iii_v_2_2_ckpt_eval/logs/full_seed444_20260414_024259.log:38`)
- final_quant_roundtrip_exact val_bpb: `1.46565755` (`legs/2026-04-14_midnight_iii_v_2_2_ckpt_eval/logs/full_seed444_20260414_024259.log:40`)
- final_sliding_window_exact val_bpb: `1.42387173` (`legs/2026-04-14_midnight_iii_v_2_2_ckpt_eval/logs/full_seed444_20260414_024259.log:42`)
- Delta vs leader: TBD
- Verdict: PROMOTION_CANDIDATE (seed-300 confirmation pending)

## Confirmation (8xH100, 600s, seed=300)
- final_sliding_window_exact val_bpb:
- Verdict: CONFIRMED / NOT_CONFIRMED
