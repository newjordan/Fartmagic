# Ablation Log

## Gate (1xGPU, 2000 steps)
- Command: not used for this promotion snapshot (promotion based on full-run artifacts imported from pod corpus)
- Seed: n/a
- Best proxy metric: n/a
- Verdict: n/a
- Notes: this leg is a one-variable child of `2026-04-12_midnight_iii_v_submission` with `QUANT_ATTN_BITS=6`.

## Full Run (8xH100, 600s, seed=444)
- Command: `cd /workspace/parameter-golf && SEED=444 NPROC_PER_NODE=8 bash legs/2026-04-13_midnight_iii_v_qattn6/run.sh`
- final_sliding_window_exact val_bpb: `1.09159026` (`logs/full_seed444_20260413_044438.log:111`)
- final_quant_roundtrip_exact val_bpb: `1.46704988` (`logs/full_seed444_20260413_044438.log:109`)
- Total submission size: `15576180` bytes (`logs/full_seed444_20260413_044438.log:107`)
- Delta vs prior documented leader (`1.10567949`): `-0.01408923`
- Verdict: PROMOTION_CANDIDATE

## Confirmation (8xH100, 600s, seed=300)
- Command: `cd /workspace/parameter-golf && SEED=300 NPROC_PER_NODE=8 bash legs/2026-04-13_midnight_iii_v_qattn6/run.sh`
- final_sliding_window_exact val_bpb: `1.09226433` (`logs/full_seed300_20260413_050254.log:111`)
- final_quant_roundtrip_exact val_bpb: `1.47483853` (`logs/full_seed300_20260413_050254.log:109`)
- Total submission size: `15569556` bytes (`logs/full_seed300_20260413_050254.log:107`)
- Verdict: CONFIRMED
