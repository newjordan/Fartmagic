# B04: Text Diffusion

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

## TV1 (Hybrid AR + Diffusion; Requires Code Path)

- Keep AR CE objective and add masked diffusion denoise loss on token embeddings.
- Proposed flags:
  - `DIFFLM_ENABLED=1`
  - `DIFFLM_STEPS=16`
  - `DIFFLM_WEIGHT=0.15`
  - `DIFFLM_SCHEDULE=cosine`

## Ablation Ladder

| Arm | Change | Goal |
|---|---|---|
| A0 | Pure AR (control) | Baseline |
| A1 | AR + diffusion aux loss | Sign-of-life under same steps |
| A2 | Diffusion plan-conditioning (`plan_tokens`) | Improve multi-step consistency |
| A3 | Diffusion-only decode at eval (optional) | Compare quality vs latency tradeoff |

## Code Touchpoints

- `train_gpt.py`: noising schedule, denoiser head, hybrid loss weighting

## References (Local Notes)

- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2601.22527_texttt-eos-training-free-bidirectional-variable-length-control-for-masked-diffusion-llms_20260210_012708.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.19066_idlm-inverse-distilled-diffusion-language-models_20260331_162451.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2603.13243_think-first-diffuse-fast-improving-diffusion-language-model-reasoning-via-autoregressive-plan-conditioning_20260401_091454.md`

