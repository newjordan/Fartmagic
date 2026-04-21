# wh_kern_int Integration Plan — 2026-04-20

## Purpose

This branch is not for more kernel search. It is the branch where the best whale donor inherits the blessed whale kernel and becomes the carry-forward model path.

## Fixed Decisions

### Donor
- Use the strongest donor recipe from `2026-04-15_whale_l12_dim576_matrixbits5`.
- Treat that leg's tracked env as the donor contract:
  - `NUM_LAYERS=12`
  - `MODEL_DIM=576`
  - `EMBEDDING_DIM=576`
  - `MATRIX_BITS=5`
  - `XSA_LAST_N=11`
  - `PARALLEL_RESIDUAL_START=8`
  - `QK_GAIN_INIT=5.25`
  - `MATRIX_LR=0.022`
  - `MUON_WD=0.095`
  - `EMA_DECAY=0.9965`
  - `WARMDOWN_FRAC=0.72`
  - `TTT_ENABLED=1`

### Trainer Contract
- Do **not** revive `gemini_runs/pr1493_train_gpt.py`.
- The preserved PR1493-family contract lives in `pr1493_decompressed.py`.
- `scripts/train_gpt.py` is a different trainer surface and is not the donor splice target for this branch.

### Blessed Kernel
- Carry-forward `2x` winner:
  - family: `primary`
  - `DQ=128,64,8,5`
  - `DKDV=128,64,8,3`
  - `DKDV_MR=256`
  - `QMR=1984`
- Kernel anchor source: `legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/winner_anchor.json`

### Kernel Path
- Use the native whale path: `custom_whale_attn_fwd`
- Do **not** treat the older hybrid path `whale_fwd_fa3_bwd` as the blessed integration target.
- The integration goal is donor + blessed whale kernel, not donor + temporary hybrid fallback.

## Integration Matrix

| Layer | Source of truth | Carry-forward choice | Action | Risk if wrong |
|---|---|---|---|---|
| Donor recipe | `legs/2026-04-15_whale_l12_dim576_matrixbits5/tracked_env.sh` | matrixbits5 12L donor | preserve env exactly | accidental donor drift |
| Trainer contract | `pr1493_decompressed.py` | preserved PR1493-family surface | wrap it, do not resurrect purged file | wrong runner / invalid comparisons |
| Attention hook | preserved trainer `flash_attn_3_func` callsite | replace with `custom_whale_attn_fwd` | monkey-patch in wrapper | FA3 silently remains live |
| Kernel knobs | `winner_anchor.json` | primary `@1984` | export kernel env in tracked env | wrong kernel family deployed |
| Validation baseline | displaced challengers in winner docs | fallback `928`, hybrid A `2048`, historical Hybrid B `1664/1952/2192` | keep only for rollback A/B | reopening search by accident |
| Artifact path | donor leg convention | `artifacts/wh_kern_int_primary_donor/` | export renamed final artifacts | result confusion / overwrite |

## Diff Guard Policy

This branch now has two modes:

### Wrapper mode
- `gemini_runs/wh_kern_int_train_gpt.py` wraps `pr1493_decompressed.py`
- no leg-local `train_gpt.py` exists yet
- `leg_diff_guard` is not triggered by the wrapper alone

### Leg-trainer mode
If we create or edit `legs/*/train_gpt.py`, then `leg-diff-guard` becomes mandatory.

Required command before any run or commit:

```bash
python3 scripts/leg_diff_guard.py legs/2026-04-20_wh_kern_int_primary_donor
```

Rules:
- a FAIL is a blocker
- do not widen thresholds silently
- if we materialize a leg-local trainer, add the real `Parent:` line first

## Execution Order

1. Keep donor env fixed.
2. Keep kernel env fixed from the blessed `2x` anchor.
3. Verify wrapper imports and shell syntax.
4. Run a short donor integration smoke.
5. Check that logs print the blessed primary kernel settings.
6. Confirm artifact export still works.
7. Only then decide whether a leg-local trainer copy is necessary.

## Current Branch Artifacts

Implemented scaffold:
- `gemini_runs/wh_kern_int_train_gpt.py`
- `legs/2026-04-20_wh_kern_int_primary_donor/run.sh`
- `legs/2026-04-20_wh_kern_int_primary_donor/tracked_env.sh`
- `legs/2026-04-20_wh_kern_int_primary_donor/PLAN.md`
- `legs/2026-04-20_wh_kern_int_primary_donor/hypothesis.md`

Benchmark/fallback contract: `docs/wh_kern_int_benchmark_contract_20260420.md`

## Success Condition

The branch succeeds when:
- the donor runs on the preserved PR1493-family contract
- the blessed whale kernel is the actual default attention path
- artifacts export cleanly
- the branch can be evaluated without reopening kernel search
