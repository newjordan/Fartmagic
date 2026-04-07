# V1.3 Submission Sweep: Long-Context DenseHash (Non-Ngram + Brotli)

## Why V1.3 Exists

`v1.2` improved legal compliance and compression, but still showed weak learning-per-step near the 10-minute wallclock cap.  
This `v1.3` lane explicitly tests the two biggest hypotheses:

1. schedule mismatch under wallclock cap (too little useful late-phase optimization),
2. insufficient true model depth (`NUM_LAYERS`) for long-context quality.

## Sweep Arms

All arms keep legal non-ngram gates and Brotli artifact enforcement.

- `23_v1_3_schedule_control_non_ngram_brotli` (control)
  - schedule-aligned control with optimizer retune.
- `24_v1_3_depth12_non_ngram_brotli` (candidate)
  - increases true depth to 12 layers at dim 384.
- `25_v1_3_depth14_dim352_non_ngram_brotli` (candidate)
  - increases depth to 14 while trimming width for runtime balance.

## Common Constraints

- Non-ngram legal guard:
  - `NGRAM_EVAL_ORDER=0`
  - `NGRAM_EVAL_ALPHA=0.0`
  - `NGRAM_EVAL_ADAPTIVE=0`
  - `CUBRIC_CADENCE=0`
- Compression gate:
  - run fails with `fail_compressor` if `Serialized model mixed+brotli:` is not present.
- Track profile default:
  - `ITERATIONS=12000`
  - `MAX_WALLCLOCK_SECONDS=590`
- 4K quality tracking default:
  - `TRACK_4K_BPB=1`
  - `TRACK_EVAL_SEQ_LEN=4096`
  - `TRACK_EVAL_STRIDE=0` (keep arm/default stride unless explicitly overridden)

## Files

- `test_lab/PR_sidelane_8x_linked/run_v1_3_non_ngram.sh`
- `test_lab/PR_sidelane_8x_linked/sidelane/scripts/run_v1_3_submission_sweep_non_ngram_brotli.sh`
- `test_lab/PR_sidelane_8x_linked/sidelane/04_ssm_e2e_ttt_long_context/arms/23_v1_3_schedule_control_non_ngram_brotli.env`
- `test_lab/PR_sidelane_8x_linked/sidelane/04_ssm_e2e_ttt_long_context/arms/24_v1_3_depth12_non_ngram_brotli.env`
- `test_lab/PR_sidelane_8x_linked/sidelane/04_ssm_e2e_ttt_long_context/arms/25_v1_3_depth14_dim352_non_ngram_brotli.env`

## Run Command

From `neural/`:

```bash
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
bash test_lab/PR_sidelane_8x_linked/run_v1_3_non_ngram.sh
```

Run one arm only:

```bash
ARM_ONLY=24_v1_3_depth12_non_ngram_brotli \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
bash test_lab/PR_sidelane_8x_linked/run_v1_3_non_ngram.sh
```

## Outputs

- summary:
  - `test_lab/PR_sidelane_8x_linked/sidelane/runs/v1_3_submission_sweep_non_ngram_brotli/<timestamp>_s<seed>/summary.tsv`
- leaderboard:
  - `test_lab/PR_sidelane_8x_linked/sidelane/runs/v1_3_submission_sweep_non_ngram_brotli/<timestamp>_s<seed>/leaderboard.tsv`

`leaderboard.tsv` ranks by non-ngram `sw_bpb` (`final_sliding_window*_exact`).
When 4K tracking is enabled (default), ranking metric is `sw_bpb_4k` and the tracked eval context is recorded in `tracked_eval_seq_len`.
