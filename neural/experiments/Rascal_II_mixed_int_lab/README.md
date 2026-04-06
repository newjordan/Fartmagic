# Rascal II Mixed-Int Lab

This folder is the active lab for buying back artifact budget and spending it on a better base model.

Seeded from:
`/home/frosty40/sota_nueral/experiments/Rascal_II_homebase`

Purpose:
- Keep `Rascal_II_homebase/` clean as the legal baseline.
- Test mixed-bit export policies as a first-class variable.
- Use recovered bytes to fund original capacity changes instead of defaulting to a bigger public-vocab trend.

What changed here:
- Export bitwidth is configurable per category:
  - `QUANT_ATTN_BITS`
  - `QUANT_MLP_BITS`
  - `QUANT_AUX_BITS`
  - `QUANT_EMBED_BITS`
  - `QUANT_OTHER_BITS`
- Supported values: `4`, `5`, `6`, `7`, `8`, `16`
- `16` means float passthrough storage.
- Artifact name is configurable with `QUANT_ARTIFACT_PATH` and defaults to `final_model.mixed.ptz`.

Current default policy:
- `attn=6`
- `mlp=6`
- `aux=6`
- `embed=6`
- `other=8`

First sweep:
```bash
SEED=300 QUANT_ATTN_BITS=5 QUANT_MLP_BITS=6 QUANT_EMBED_BITS=8 bash test_lab/Rascal_II_mixed_int_lab/run.sh
SEED=300 QUANT_ATTN_BITS=6 QUANT_MLP_BITS=5 QUANT_EMBED_BITS=8 bash test_lab/Rascal_II_mixed_int_lab/run.sh
SEED=300 QUANT_ATTN_BITS=5 QUANT_MLP_BITS=5 QUANT_EMBED_BITS=8 QUANT_AUX_BITS=8 bash test_lab/Rascal_II_mixed_int_lab/run.sh
```

## 4xGPU Ablation Results (2026-04-06, seed=300, 600s wallclock)

| Metric | Control (all int6) | Mixed (attn=5) | Delta |
|--------|-------------------|----------------|-------|
| Post-EMA val_bpb | 1.1763 | 1.1736 | **-0.0027** |
| Quant roundtrip BPB | 1.1902 | 1.1907 | +0.0005 (noise) |
| Sliding window BPB | 1.1528 | 1.1500 | **-0.0028** |
| Artifact size | 14.66 MB | 14.12 MB | **-0.54 MB saved** |
| Steps completed | 3214 | 3292 | +78 |

**Verdict:** attn=5 is a net positive. Better pre-quant BPB, saves 0.54 MB, quant roundtrip is flat.
Note: 4xGPU / ~3200 steps, not directly comparable to 8xGPU baseline (1.1099 BPB).

Rules:
- No SLOT.
- No evaluation-only adaptation tricks.
- Do not treat `4096` vocab as the default answer.
- If a mixed-bit policy wins size, spend that budget on a stronger base model and re-test honestly.
