# Hypothesis — The Witch

Parent: vault/train_gpt_midnight_iii_base.py

## Architecture

3 physical layers at dim=1024, looped 4× = 12 effective layers.
Same parameter budget as 12L (dim=512, 12 layers), ~29.3M params.

| | 12L Clean | The Witch |
|---|-----------|-----------|
| model_dim | 512 | 1024 |
| physical layers | 12 | 3 |
| effective layers | 12 | 12 (3 × 4 loops) |
| num_heads / kv | 8 / 4 | 16 / 8 |
| head_dim | 64 | 64 |
| mlp_dim | 1536 | 3072 |
| bank params | 28.31M | 28.31M |
| matmul arithmetic intensity | 256 FLOP/byte | 512 FLOP/byte |
| H100 tensor core utilization | ~43% | ~87% |

## Why

The 12L at dim=512 is memory-bandwidth-bound. H100 tensor cores idle at 43% waiting
on DRAM to deliver activations. Every matmul shape `[98304, 512] × [512, 512]` has
arithmetic intensity 256 — well below the H100 ridge point of 590 FLOP/byte.

Doubling model_dim to 1024 doubles the arithmetic intensity to 512 (87% of ridge).
Cutting to 3 physical layers keeps the same parameter count. Looping 4× recovers
12 effective layers of depth. The weight reuse from looping is nearly free — 3 layers
of bank weights (9 MB) fit entirely in H100 L2 cache (50 MB).

The tradeoff: 3 unique weight matrices vs 12. Looping provides depth but not feature
diversity. The hypothesis is that 87% hardware utilization on wider representations
outweighs the diversity loss from fewer physical layers.

## Pass Criteria

1. Training converges — loss decreases normally through the 600s window.
2. `step_avg` after warmup is comparable to or better than 12L (≤ 110ms target).
3. `final_quant_roundtrip_exact` has a quant gap ≤ 0.05 BPB vs post-EMA quality.
4. `final_sliding_window_exact` on the deployed artifact is competitive with 12L (≤ 1.15 BPB).
