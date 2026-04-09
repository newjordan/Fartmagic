# Ablation: crawler_9f_corpus_ablations_v1

Date: 2026-04-08
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Screen (4xGPU, 1500 steps, seed=444)

Status: [ ] pending  [ ] pass  [ ] fail

### Battery / Recurrence Arms

| Arm | Change | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_ctrl | verdict |
|-----|--------|-------------|---------|-------------|---------|-------|---------------|---------|
| A00 | Control (BWX 9F) | | | | | | -- | |
| A01 | TAP_DIM=32 shared | | | | | | | |
| A02 | ANCHOR_DIM=32 | | | | | | | |
| A03 | LOOPS=4 naive (9,1,1,1) | | | | | | | |
| A04 | LOOPS=4 diff (9,3,1,1) | | | | | | | |
| A05 | LOOPS=5 prog (9,5,3,1,1) | | | | | | | |
| A06 | INST_DIM=64 | | | | | | | |
| A07 | NUM_CRAWLER_LAYERS=2 | | | | | | | |
| A08 | CRAWLER_QUANT_INT8=1 | | | | | | | |

### QAT Arms

| Arm | Change | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_ctrl | verdict |
|-----|--------|-------------|---------|-------------|---------|-------|---------------|---------|
| A09 | QAT legacy | | | | | | | |
| A10 | QAT softclamp | | | | | | | |
| A11 | QAT sigmoidste | | | | | | | |

### Quant Policy Arms (post-train on best A00 checkpoint)

| Arm | INT6_CATS | raw_bpb | int6_sw_bpb | bytes | delta_vs_Q00 | verdict |
|-----|-----------|---------|-------------|-------|-------------|---------|
| Q00 | mlp,attn,aux | | | | -- | |
| Q01 | mlp,attn | | | | | |
| Q02 | attn | | | | | |
| Q03 | mlp | | | | | |

Screen target: delta <= -0.001 = signal. delta <= -0.002 = strong signal.

## Promotion candidates

(Fill after screen completes. Winners go to 2000-step 8xGPU gate.)

| Rank | Arm | delta | Next step |
|------|-----|-------|-----------|
| | | | |

## Notes

- Summary TSV: `crawler/2026-04-08_crawler_9f_corpus_ablations_v1/results/summary_screen_s444_*.tsv`
- 4xGPU 1500-step screen is directionally valid (BW7 MegaGate proved cross-arm deltas valid on 4x)
- Noise floor higher than 2000-step gate; marginal signals (<0.001) may be ambiguous
