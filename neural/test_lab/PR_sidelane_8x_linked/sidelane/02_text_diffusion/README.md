# Text Diffusion

## Core Hypothesis

A discrete diffusion-style denoising objective can improve compression performance if used as an auxiliary objective or as a short refinement stage over AR logits.

## Phase Plan

1. Phase A (proof): add denoising-on-token-noise auxiliary loss on top of AR model.
2. Phase B (artifact): compress denoising components into shared weights (no separate heavy backbone).
3. Phase C (runtime): restrict denoising steps to tiny count or test-time-only where allowed.

## Cross-Repo Tasks

- `parameter-golf-lab`: implement `train_gpt_diffaux.py` with shared trunk.
- `sota_crawler`: sweep noise schedule, denoise depth, auxiliary weight.
- `ml-research-analysis`: shortlist practical discrete text diffusion recipes under parameter limits.

## Initial Ablation Grid

- Noise rates: 0.05, 0.1, 0.2
- Denoise steps (train objective): 1, 2, 4
- Aux weight: 0.05, 0.1, 0.25

## Promotion Criteria

- Better val_bpb trend than matched AR baseline with same wallclock class.
- Shared-weight approach preserves compressed size budget headroom.

## Kill Criteria

- Requires too many denoising steps to be competitive.
