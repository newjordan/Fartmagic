# Longworm RK Concepts (Staged)

This package stages a dedicated Longworm concept sweep with three transition-integrator variants:

- `41_longworm_k2_non_ngram_brotli`
- `42_longworm_k4_non_ngram_brotli`
- `43_longworm_k2_4_hybrid_non_ngram_brotli`

All three arms use the same Longworm capacity/schedule baseline and differ only by transition integrator:

- `rk2`
- `rk4`
- `hybrid_k2_k4` with `TRANSITION_HYBRID_LAST_N=4`

## Separation Guarantee

- Longworm now has its own trainer: `neural/experiments/Longworm/train_longworm.py`.
- The RK concept runner resolves `TRAIN_PY` to that Longworm trainer only.
- Rascal experiment files are not used by this RK concept sweep.

## Launch

```bash
cd /workspace/parameter-golf/neural
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
NPROC_PER_NODE=8 \
bash test_lab/PR_sidelane_8x_linked/run_longworm_rk.sh
```

Optional single-arm run:

```bash
ARM_ONLY=41_longworm_k2_non_ngram_brotli bash test_lab/PR_sidelane_8x_linked/run_longworm_rk.sh
```

## Status

Staged only. No run launched in this change.
