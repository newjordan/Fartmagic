# B06: Universal Transformer (4-hour Run)

## Base

- Base ID: `BASE_4H`
- Path: `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/train_gpt.py`
- Current reference score: `val_bpb=1.2074` (4-hour non-record baseline)

## TV0 (4h Control)

```bash
cd /home/frosty40/parameter-golf-lab/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3
MAX_WALLCLOCK_SECONDS=14400 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## TV1 (UT Recurrence; Requires Code Path)

- Replace per-layer unique weights with one shared block unrolled with learned time embedding.
- Proposed flags:
  - `UT_ENABLED=1`
  - `UT_STEPS=12`
  - `UT_ACT_ENABLED=1`
  - `UT_MAX_PONDER=16`

## Ablation Ladder

| Arm | Change | Goal |
|---|---|---|
| A0 | 4h baseline control | Baseline under same wallclock |
| A1 | Shared UT block, fixed depth | Parameter efficiency test |
| A2 | Adaptive compute time (ponder) | Dynamic depth by token difficulty |
| A3 | UT + QAT schedule | Quantization compatibility under recurrence |

## Code Touchpoints

- `train_gpt.py`: block definition, recurrence loop, ACT halting loss

## References (Local Notes)

- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2601.10007_continuous-depth-transformers-with-learned-control-dynamics_20260210_164016.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2603.01914_adaponderlm-gated-pondering-language-models-with-token-wise-adaptive-depth_20260403_024015.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2603.05573_why-depth-matters-in-parallelizable-sequence-models-a-lie-algebraic-view_20260401_123544.md`

