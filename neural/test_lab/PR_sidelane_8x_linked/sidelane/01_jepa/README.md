# JEPA

## Core Hypothesis

A latent prediction objective can improve sample efficiency under tight parameter budgets by forcing compressed, predictive representations before token-level decoding.

## Phase Plan

1. Phase A (proof): add a lightweight BYOL/JEPA-style latent target branch to existing transformer trunk.
2. Phase B (artifact): retain only minimal extra heads needed at train time; keep export path clean.
3. Phase C (runtime): fuse branch computations or prune branch-only params for 8xH100 speed viability.

## Cross-Repo Tasks

- `parameter-golf-lab`: implement `train_gpt_jepa.py` and export-safe toggles.
- `sota_crawler`: sweep latent dim, EMA decay, loss weight.
- `ml-research-analysis`: summarize JEPA design constraints for autoregressive text.

## Initial Ablation Grid

- Latent dim: 64, 128, 256
- EMA decay: 0.99, 0.995, 0.999
- JEPA loss weight: 0.05, 0.1, 0.2

## Promotion Criteria

- Beats naive baseline trend in matched-step experiments.
- No artifact-size regression beyond recoverable quantization.
- Stable across at least 3 seeds.

## Kill Criteria

- Consistent optimization instability or no gain after two full ablation rounds.
