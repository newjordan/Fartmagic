# V1 Submission-Ready Test: Long-Context DenseHash (Non-Ngram)

## Goal

Create a submission-ready test lane from the strongest practical candidate while explicitly excluding n-gram-assisted evaluation paths.

Selected base candidate:
- `04_ssm_e2e_ttt_long_context / 13_longctx_ttt_densehash`

V1 lane:
- `21_v1_longctx_ttt_densehash_non_ngram`

## Why This Candidate

From the 8xH100 linked sweep (`20260407_023314_s445`), the long-context candidate had:
- Strong non-ngram sliding-window quality under long-context settings.
- Stable throughput and artifact size behavior under 8-GPU execution.
- Clear architectural differentiation versus baseline (long context + dense hash pathway) without requiring external n-gram scoring.

Observed non-ngram metric from that sweep:
- `final_sliding_window_exact val_bpb: 1.61792653` for `13_longctx_ttt_densehash`.

## Legal Constraint Handling

This V1 lane is hardened to avoid n-gram-dependent ranking:

- Arm file sets:
  - `NGRAM_EVAL_ORDER=0`
  - `NGRAM_EVAL_ALPHA=0.0`
  - `NGRAM_EVAL_ADAPTIVE=0`
  - `CUBRIC_CADENCE=0`
- Runner script force-overrides those values again after sourcing env.
- Runner marks the run `fail_legal` if log contains `final_sliding_window_ngram*`.

This enforces a non-ngram metric path by construction.

## Files Added

- `test_lab/PR_sidelane_8x_linked/run_v1_non_ngram.sh`
- `test_lab/PR_sidelane_8x_linked/sidelane/scripts/run_v1_submission_non_ngram.sh`
- `test_lab/PR_sidelane_8x_linked/sidelane/04_ssm_e2e_ttt_long_context/arms/21_v1_longctx_ttt_densehash_non_ngram.env`

## Run Command

From `neural/`:

```bash
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
bash test_lab/PR_sidelane_8x_linked/run_v1_non_ngram.sh
```

Optional overrides:

```bash
SEED=446 ITERATIONS=800 NPROC_PER_NODE=8 MAX_WALLCLOCK_SECONDS=2400 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
bash test_lab/PR_sidelane_8x_linked/run_v1_non_ngram.sh
```

## Outputs

Runner writes:

- `test_lab/PR_sidelane_8x_linked/sidelane/runs/v1_submission_non_ngram/<timestamp>_s<seed>/v1_longctx_ttt_densehash_non_ngram.log`
- `test_lab/PR_sidelane_8x_linked/sidelane/runs/v1_submission_non_ngram/<timestamp>_s<seed>/summary.tsv`

`summary.tsv` includes:
- `status`
- `model_params`
- `diag_bpb`
- `sw_bpb` (non-ngram sliding window exact)
- `total_size_mixed_zlib_bytes`
- `total_size_int6_bytes` (if emitted)
- `step_avg_ms`

## Acceptance Criteria (V1 Gate)

1. `status=ok` in summary.
2. No `final_sliding_window_ngram*` lines in log.
3. `sw_bpb` present and finite.
4. Submission artifact size under 16MB (`total_size_* < 16777216`).
5. Run completes within wallclock budget for your lane target.

## Notes

- This is a single-candidate submission test lane, not a control-vs-candidate sweep.
- If you want strict reproducibility snapshots, archive:
  - `summary.tsv`
  - full log
  - exact commit SHA
  - environment package list.
