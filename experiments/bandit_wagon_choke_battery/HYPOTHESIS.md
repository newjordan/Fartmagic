# bandit_wagon_choke_battery (BWCB) — Battery on Pyramid

## Background

BWCS established pyramid-512 as the dominant choke shape (-0.01037 vs control, quant_gap
collapsed to -0.0001). The mega ablation (BWB series, flat/no-choke) shows battery alone
has signal but with unexpected scale ordering: 1,2,4 beats 1,3,9 in isolation.

This series answers: **does battery stack with pyramid-512, and which scale wins in combo?**

## References (from prior runs)

| Run | Config | INT6_SW_BPB | Quant Gap |
|-----|--------|-------------|-----------|
| BWCS-00 | flat ctrl (1 shard) | 1.45761 | +0.0013 |
| BWCS-02 | pyramid-512 (1 shard) | 1.44724 | -0.0001 |
| BWC-04 | flat choke=512 (80 shards) | 1.42887 | -0.0009 |
| BWB-01 | battery 1,2,4 flat (80 shards) | 1.43769 | -0.0010 |
| BWB-02 | battery 1,3,9 flat (80 shards) | 1.44470 | +0.0028 |

## Arms

| ID | Shape | Rope Scales | Purpose |
|----|-------|-------------|---------|
| BWCB-00 | pyramid-512 | 1,2,4 | Gentle combo — BWB-01 scale winner on pyramid |
| BWCB-01 | pyramid-512 | 1,3,9 | Core hypothesis combo |
| BWCB-02 | pyramid-512 | 1,5,25 | Aggressive combo |

References from BWCS-00 and BWCS-02 — no control repin needed.

## Results (seed=444, 500 steps, 1 shard)

| ID | Scales | Raw BPB | INT6_SW_BPB | Quant Gap | vs BWCS-00 | vs BWCS-02 |
|----|--------|---------|-------------|-----------|------------|------------|
| BWCS-02 | (pyramid-512 reference) | 1.4473 | **1.44724** | **-0.0001** | -0.01037 | — |
| BWCB-00 | 1,2,4 | 1.4473 | 1.44850 | +0.0012 | -0.00911 | +0.00126 |
| BWCB-01 | 1,3,9 | 1.4492 | 1.45016 | +0.0010 | -0.00745 | +0.00292 |
| BWCB-02 | 1,5,25 | 1.4525 | 1.45534 | +0.0028 | -0.00227 | +0.00810 |

## Verdict: Ascending Battery Hurts Pyramid

**No ascending config beats pyramid-512 alone.** All three are strictly worse, and quant_gap
goes positive for all (vs -0.0001 for pyramid alone).

Ascending battery and pyramid work in opposite directions on inter-loop distribution coherence:
- Pyramid converges distributions → forces universal stage1 commitment → clean quantization
- Ascending battery diverges distributions → each loop attends a wider horizon → quantization stress

Together they partially cancel. Pyramid does absorb some battery stress (1,3,9 standalone: +0.0028,
pyramid+1,3,9: +0.0010 — partial improvement) but not enough to match pyramid alone.

**Scale → damage monotonic:** 1,2,4 (+0.00126) < 1,3,9 (+0.00292) < 1,5,25 (+0.00810)
vs pyramid reference. Wider ascending spread = more distribution divergence = more damage.

## Follow-On: BWCD (Descending)

Descending battery (9,3,1) had near-zero quant gap standalone (+0.0001 on flat MLP) vs
ascending 1,3,9 (+0.0028). Hypothesis: descending also converges distributions (wide→narrow
is a progressive refinement sequence) and may therefore STACK with pyramid rather than fight it.

BWCD tests: 9,3,1 | 4,2,1 | 9,1,1 | 9,3,9 — all on pyramid-512.
