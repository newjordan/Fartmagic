# Rascal II Research-Driven Ablation Sweep — Final Results

**Date:** 2026-04-06
**Base:** Rascal II (1.1099 BPB 3-seed mean, vault baseline)
**Config:** 4×GPU, seed=300, 600s wallclock, mixed-int quant (attn=5, mlp=6, embed=8)
**Source:** 121K paper analyses from memgrafter/analysis corpus + internal hypotheses

## Control Baselines

| Metric | All int6 control | Mixed-int (attn=5) |
|--------|-----------------|-------------------|
| Post-EMA val_bpb | 1.1763 | 1.1736 |
| Roundtrip BPB | 1.1902 | 1.1907 |
| Sliding window BPB | 1.1528 | 1.1500 |
| Artifact size | 14.66 MB | 14.12 MB |
| Steps completed | 3214 | 3292 |

## Arm Results Summary

| Arm | Variable | Source | Sliding BPB | Delta vs mixed-int | Verdict |
|-----|----------|--------|-------------|-------------------|---------|
| Mixed-int | QUANT_ATTN_BITS=5 | Internal | 1.1500 | baseline | **POSITIVE** |
| A: Trigram | TRIGRAM=1 | vault code | — | — | **DEAD** (user pre-tested extensively) |
| B: mu-centering | Code change (subtract mean embed after optim step) | arXiv 2601.02031 | 1.1597 | +0.0097 | **DEAD** |
| C: Gated attention | GATED_ATTENTION=1 | Attention literature | 1.1497 | -0.0003 (noise) | **DEAD** |
| D: HEQ | Code change (entropy-maximizing quant scales) | Quantization literature | ~1.1540 | +0.0040 | **DEAD** |
| E: DDL | Code change (rank-1 delta residual) | Residual learning papers | 1.2005 | +0.0505 | **DEAD** |

## Detailed Arm Results

### Arm B: mu-centering (DEAD)
- **Paper:** arXiv 2601.02031
- **Change:** After optimizer step, subtract mean output embedding: `tok_emb.weight.sub_(tok_emb.weight.mean(dim=0, keepdim=True))`
- **Roundtrip BPB:** ~1.2000 (+0.0093 vs control)
- **Sliding BPB:** ~1.1597 (+0.0097 vs mixed-int)
- **Analysis:** Centering hurts — the embedding distribution is already well-shaped by the optimizer. Forcing zero-mean removes useful signal.

### Arm C: Gated attention (DEAD)
- **Change:** Per-head learned sigmoid gate on attention output. `attn_gate = nn.Linear(dim, num_heads)`, init bias=4.0.
- **Post-EMA val_bpb:** 1.1732 (-0.0004 vs mixed-int, noise)
- **Roundtrip BPB:** 1.1928 (+0.0021 vs mixed-int)
- **Sliding BPB:** 1.1497 (-0.0003 vs mixed-int, noise)
- **Steps:** 3205 (-87 vs mixed-int, ~1ms/step overhead)
- **Analysis:** No signal. Existing attn_scale + resid_mix params already provide sufficient per-layer gating.

### Arm D: HEQ entropy-maximizing quant scales (DEAD)
- **Change:** Replace `_find_best_row_scales` with entropy-maximizing search over 11 percentiles. Scores by entropy + MSE.
- **Step avg:** ~184ms vs 177ms baseline (+7ms overhead)
- **Steps completed:** fewer than control (overhead ate budget)
- **Sliding BPB:** ~1.1540 (+0.0040 vs mixed-int)
- **Analysis:** The entropy-maximizing search is too slow for export-time quantization. Even if scales were better, the overhead cost more steps than the quant quality gained.

### Arm E: DDL rank-1 delta residual (DEAD)
- **Paper:** Residual learning with learned direction vectors
- **Change:** Added `ddl_k` (direction vector) + `ddl_beta` (strength) to each Block. Residual update projects error onto learned direction.
- **Roundtrip BPB:** ~1.2407 (+0.0505 vs mixed-int)
- **Step avg:** ~227ms (+50ms, 27% slower)
- **VRAM:** Higher (extra params per block)
- **Analysis:** Catastrophic. The rank-1 update adds significant compute for negligible representational benefit at this model scale. The direction vectors don't learn useful structure in 3000 steps.

## Conclusions

1. **Rascal II is extremely well-optimized.** Bolt-on architectural tricks (gating, centering, novel residuals) don't move the needle.
2. **Mixed-int attn=5 is the only winner.** Ready for 8×GPU promotion.
3. **Speed > architecture at this point.** The base model can't be improved with small changes — getting more steps (megarascal) or better quantization is the remaining path.
4. **Research corpus value:** The 121K paper mining surfaced 5 candidates but none beat the baseline. The corpus was more useful for Ouroboros (Noisy QAT, contractive dt) where known architectural weaknesses exist.
