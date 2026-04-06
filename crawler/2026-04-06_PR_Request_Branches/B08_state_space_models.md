# B08: State-Space Models

## Base

- Base ID: `BASE_NIGHTCRAWLER`
- Path: `records/track_10min_16mb/2026-04-01_Nightcrawler_8xH100/train_gpt.py`
- Current reference score: `val_bpb=1.1761`

## TV0 (Control Run)

```bash
cd /home/frosty40/parameter-golf-lab
USE_CRAWLER=1 NUM_FLAT_LAYERS=9 NUM_CRAWLER_LAYERS=1 CRAWLER_LOOPS=1 \
SEED=42 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-01_Nightcrawler_8xH100/train_gpt.py
```

## TV1 (SSM Hybrid; Requires Code Path)

- Add SSM mixer block (Mamba/S4 style) in crawler path, keep attention in flat path.
- Proposed flags:
  - `SSM_ENABLED=1`
  - `SSM_IN_CRAWLER=1`
  - `SSM_STATE_DIM=256`
  - `SSM_GATING=1`

## Ablation Ladder

| Arm | Change | Goal |
|---|---|---|
| A0 | Nightcrawler control | Baseline |
| A1 | Replace crawler MLP with SSM mixer | Sequence memory gain test |
| A2 | Hybrid SSM + attention crossover | Balance local/global dependencies |
| A3 | Loop-aware GPTQ on SSM params | Quantization survivability |

## Code Touchpoints

- `train_gpt.py`: crawler block internals, recurrent state threading, quant path

## References (Local Notes)

- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2601.19467_on-the-expressiveness-of-state-space-models-via-temporal-logics_20260210_012538.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.22719_interpreting-and-steering-state-space-models-via-activation-subspace-bottlenecks_20260401_091111.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2601.13048_analysis-of-long-range-dependency-understanding-in-state-space-models_20260210_233019.md`

