# Longworm V1.4 Power Sweep

## Goal

Increase model capacity and per-step learning pressure versus `v1.3`, targeting roughly `~80ms`-class step times on `8x H100` while keeping the same long-context training lane and Brotli artifact flow.

## Runner

- Launcher:
  - `test_lab/PR_sidelane_8x_linked/run_longworm.sh`
  - alias: `test_lab/PR_sidelane_8x_linked/run_v1_4_power.sh`
- Core script:
  - `test_lab/PR_sidelane_8x_linked/sidelane/scripts/run_v1_4_submission_sweep_brotli.sh`

## Sweep Arms

- `31_v1_4_power_l13_d408_non_ngram_brotli` (control)
- `32_v1_4_power_l14_d420_non_ngram_brotli` (candidate)
- `33_v1_4_power_l12_d456_non_ngram_brotli` (candidate)

All arms use:
- long-context training (`TRAIN_SEQ_LEN=2048`)
- elevated token budget per step (`TRAIN_BATCH_TOKENS=196608`)
- stronger optimization settings (higher `MATRIX_LR` / `SCALAR_LR`)
- Brotli compression path

## Defaults

- `PROJECT_CODENAME=longworm`
- `SUBMISSION_PROFILE=track_10min_16mb`
- `ITERATIONS=16000`
- `MAX_WALLCLOCK_SECONDS=590`
- 4K quality tracking defaults on:
  - `TRACK_4K_BPB=1`
  - `TRACK_EVAL_SEQ_LEN=4096`

## Run

From `neural/`:

```bash
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
NPROC_PER_NODE=8 \
bash test_lab/PR_sidelane_8x_linked/run_longworm.sh
```

Single-arm run:

```bash
ARM_ONLY=32_v1_4_power_l14_d420_non_ngram_brotli \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
NPROC_PER_NODE=8 \
bash test_lab/PR_sidelane_8x_linked/run_longworm.sh
```

## Outputs

- summary:
  - `test_lab/PR_sidelane_8x_linked/sidelane/runs/longworm_v1_4_power_sweep_brotli/<timestamp>_s<seed>/summary.tsv`
- leaderboard:
  - `test_lab/PR_sidelane_8x_linked/sidelane/runs/longworm_v1_4_power_sweep_brotli/<timestamp>_s<seed>/leaderboard.tsv`

Leaderboard ranking metric prefers `sw_bpb_4k` when present.
