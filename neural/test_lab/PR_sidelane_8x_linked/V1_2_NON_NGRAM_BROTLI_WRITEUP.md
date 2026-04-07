# V1.2 Submission-Ready Test: Long-Context DenseHash (Non-Ngram + Brotli)

## Goal

Ship a submission-ready `v1.2` lane that:
- keeps the same legal non-ngram evaluation constraints,
- improves learning-per-step vs the initial `v1`,
- and enforces Brotli artifact compression.

Selected lane:
- `04_ssm_e2e_ttt_long_context / 22_v1_2_longctx_ttt_densehash_non_ngram_brotli`

## What Changed vs V1

Base architecture family remains long-context densehash, with `v1.2` overrides:
- `WARMUP_STEPS=128`
- `LATE_QAT_THRESHOLD=0`
- `MLP_MULT=3.5`
- `MATRIX_LR=0.03`
- `SCALAR_LR=0.03`
- quant bits:
  - `QUANT_ATTN_BITS=6`
  - `QUANT_MLP_BITS=7`
  - `QUANT_AUX_BITS=7`
  - `QUANT_EMBED_BITS=8`
  - `QUANT_OTHER_BITS=8`

These are intended to use available size headroom for better non-ngram quality while staying in the 10-minute build envelope.

## Legal Constraint Handling

N-gram evaluation is explicitly disabled in both the arm file and runner:
- `NGRAM_EVAL_ORDER=0`
- `NGRAM_EVAL_ALPHA=0.0`
- `NGRAM_EVAL_ADAPTIVE=0`
- `CUBRIC_CADENCE=0`

Runner also marks run status `fail_legal` if `final_sliding_window_ngram*` appears in logs.

## Compression Policy

`v1.2` enforces Brotli:
- checks for Python `brotli` module,
- auto-installs when missing (default `AUTO_INSTALL_BROTLI=1`),
- fails run summary as `fail_compressor` if log does not contain `Serialized model mixed+brotli:`.

## Files

- `test_lab/PR_sidelane_8x_linked/run_v1_2_non_ngram.sh`
- `test_lab/PR_sidelane_8x_linked/sidelane/scripts/run_v1_2_submission_non_ngram_brotli.sh`
- `test_lab/PR_sidelane_8x_linked/sidelane/04_ssm_e2e_ttt_long_context/arms/22_v1_2_longctx_ttt_densehash_non_ngram_brotli.env`

## Run Command

From `neural/`:

```bash
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
bash test_lab/PR_sidelane_8x_linked/run_v1_2_non_ngram.sh
```

Optional:

```bash
SEED=446 SUBMISSION_PROFILE=track_10min_16mb NPROC_PER_NODE=8 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
bash test_lab/PR_sidelane_8x_linked/run_v1_2_non_ngram.sh
```

Profiles:
- `track_10min_16mb` (default): `ITERATIONS=20000`, `MAX_WALLCLOCK_SECONDS=590`
- `longform`: `ITERATIONS=20000`, `MAX_WALLCLOCK_SECONDS=2400`

## Outputs

- log:
  - `test_lab/PR_sidelane_8x_linked/sidelane/runs/v1_2_submission_non_ngram_brotli/<timestamp>_s<seed>/v1_2_longctx_ttt_densehash_non_ngram_brotli.log`
- summary:
  - `test_lab/PR_sidelane_8x_linked/sidelane/runs/v1_2_submission_non_ngram_brotli/<timestamp>_s<seed>/summary.tsv`

`summary.tsv` fields:
- `status`
- `compressor`
- `model_params`
- `diag_bpb`
- `sw_bpb`
- `total_size_mixed_bytes`
- `total_size_int6_bytes`
- `step_avg_ms`
