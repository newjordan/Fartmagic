# Learning Adapters on Random Linear Maps

## Core Hypothesis

Frozen random linear projections plus tiny learned adapters can create high-capacity function classes with much lower trainable-parameter cost than full fine-tuning.

## Phase Plan

1. Phase A (proof): insert fixed random projections with trainable low-rank adapters.
2. Phase B (artifact): quantize adapters aggressively and test compression sensitivity.
3. Phase C (runtime): merge/fold adapter math where possible to reduce launch overhead.

## Cross-Repo Tasks

- `parameter-golf-lab`: adapter module + export codepath.
- `sota_crawler`: projection dimension, adapter rank, and sparsity sweeps.
- `ml-research-analysis`: theory notes on random features + linear adapters in low-budget LMs.

## Initial Ablation Grid

- Projection dim: 64, 128, 256
- Adapter rank: 2, 4, 8
- Adapter sparsity: 0.0, 0.5, 0.8

## Promotion Criteria

- Better than equal-parameter baseline at matched train steps.
- Compression remains competitive after quantization.

## Kill Criteria

- Projection noise dominates and requires larger adapters than budget allows.
