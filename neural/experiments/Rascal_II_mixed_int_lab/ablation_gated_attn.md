# Gated Attention Ablation — DEAD

**Date:** 2026-04-06
**Variable:** GATED_ATTENTION=1
**Base:** Rascal II mixed-int (attn=5, mlp=6, embed=8)
**Config:** 4×GPU, seed=300, 600s wallclock

## Results

| Metric | Mixed-int control | Gated Attention | Delta |
|--------|-------------------|-----------------|-------|
| Post-EMA val_bpb | 1.1736 | 1.1732 | -0.0004 (noise) |
| Roundtrip BPB | 1.1907 | 1.1928 | +0.0021 (worse) |
| Sliding BPB | 1.1500 | 1.1497 | -0.0003 (noise) |
| Steps completed | 3292 | 3205 | -87 (slower) |
| Artifact size | 14.12 MB | 13.51 MB | -0.61 MB |
| Step avg | ~177ms | ~178ms | +1ms overhead |

## Verdict

DEAD. No signal above noise on any metric. The per-head sigmoid gate adds ~1ms/step overhead, costing 87 steps over 600s. Sliding window delta is -0.0003 — well within run-to-run variance (~0.0003 BPB).

The gate mechanism (learned per-head sigmoid scaling of attention output) does not improve Rascal II. The model's existing attn_scale + resid_mix parameters already provide sufficient gating.
