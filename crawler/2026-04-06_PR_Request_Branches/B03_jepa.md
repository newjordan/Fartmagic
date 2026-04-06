# B03: JEPA

## Base

- Base ID: `BASE_RASCAL`
- Path: `records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py`
- Current reference score: `val_bpb=1.1099`

## TV0 (Control Run)

```bash
cd /home/frosty40/parameter-golf-lab
SKIP_GPTQ=1 SEED=42 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py
```

## TV1 (First JEPA Variant; Requires Code Path)

- Add JEPA latent prediction loss on masked future chunks of hidden states.
- Proposed flags:
  - `JEPA_ENABLED=1`
  - `JEPA_WEIGHT=0.05`
  - `JEPA_MASK_SPAN=8`
  - `JEPA_STOPGRAD_TARGET=1`

## Ablation Ladder

| Arm | Change | Goal |
|---|---|---|
| A0 | Rascal control | Establish matched-step baseline |
| A1 | Top-layer JEPA aux loss only | Check representation signal |
| A2 | Multi-layer JEPA (`layers=8,9,10`) | Test depth of predictive target |
| A3 | JEPA + QAT on/off | Evaluate quantization compatibility |

## Code Touchpoints

- `train_gpt.py`: block outputs, aux-head definition, training loss aggregation

## References (Local Notes)

- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.11832_jepa-vla-video-predictive-embedding-is-needed-for-vla-models_20260401_074953.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.19322_us-jepa-a-joint-embedding-predictive-architecture-for-medical-ultrasound_20260402_091114.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.14771_got-jepa-generic-object-tracking-with-model-adaptation-and-occlusion-handling-using-joint-embedding-predictive-architecture_20260404_023357.md`

