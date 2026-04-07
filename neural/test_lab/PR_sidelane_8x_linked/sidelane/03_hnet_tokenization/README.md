# H-net Tokenization

## Core Hypothesis

Hierarchical tokenization can lower effective sequence entropy by mixing coarse and fine symbols, reducing bits-per-byte pressure for fixed model size.

## Phase Plan

1. Phase A (proof): dual-level tokenizer prototype (coarse + refinement) with deterministic encoding.
2. Phase B (artifact): validate byte-accurate val_bpb accounting and deterministic decode.
3. Phase C (runtime): streamline tokenization pipeline for reproducible training/eval speed.

## Cross-Repo Tasks

- `parameter-golf-lab`: training/eval plumbing for hierarchical token streams.
- `sota_crawler`: controlled tokenizer sweeps and paired baseline comparisons.
- `ml-research-analysis`: prior-art review on hierarchical/discrete tokenization for LMs.

## Initial Ablation Grid

- Coarse vocab sizes: 512, 1024, 2048
- Refinement vocab sizes: 256, 512, 1024
- Mix ratios: 25/75, 50/50, 75/25 (coarse/fine)

## Promotion Criteria

- Verified val_bpb correctness under challenge accounting.
- Comparable or better trend than best same-compute tokenizer baseline.

## Kill Criteria

- Accounting complexity creates unacceptable verification risk.
