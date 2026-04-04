# Helix SplitHead — Hypothesis

## Core Concept
The crawler's attention heads split between self-attend and cross-attend.
Self-heads maintain crawler coherence. Cross-heads read the flat stream to
find what needs correction. This is the custom crawler transformer — a
cross-referencing correction block, not a standard transformer.

Cross-heads use NO RoPE on their K projections — position-agnostic content
routing. Self-heads keep RoPE. The crawler learns WHERE to look (self-heads)
and WHAT to fix (cross-heads) simultaneously.

## What This Replaces
- Marco-Polo (failed): was a separate cross-attention bridge mechanism
- TAP (marginal): was a frozen encoder signal injection
- Anchor (regressed on tap-off): was a per-loop causal write state

SplitHead replaces all of these with one architectural change: rewire
where some attention heads get their K/V from.

## Ablation Suite (30 arms)

### Section 1: Controls (2 arms)
- S0: Helix standard (no split heads)
- S1: No helix at all

### Section 2: Split-Head Sweep (4 arms)
- H1-H4: 1/2/3/4 of 4 heads cross-attend
- Key question: what's the optimal self/cross ratio?

### Section 3: Split-Head + Bridge Dim (4 arms)
- D1-D4: cross × dim combinations
- Does split-head change the optimal bridge width?

### Section 4: Competition Tech (6 arms)
- W1-W2: Weight decay 0.09, 0.12 (quant gap)
- Q1-Q2: QK gain 5.0, 6.0
- L1-L2: Matrix LR 0.02, 0.04

### Section 5: Crawler MLP (3 arms)
- M1-M3: MLP mult 2.0, 6.0 + 7F depth

### Section 6: Depth Scaling (3 arms)
- K1-K3: 7F with cross=2, cross=4, no cross (control)

### Section 7: RoPE Variations (2 arms)
- R1-R2: Different RoPE scales for crawler with split heads

### Section 8: Best Combo Stacking (4 arms)
- B1-B4: Stack the winners from above
