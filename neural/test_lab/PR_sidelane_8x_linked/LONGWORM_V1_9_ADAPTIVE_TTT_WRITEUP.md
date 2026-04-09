# Longworm V1.9 Adaptive TTT

## Objective
Evaluate whether adaptive test-time compute improves Longworm under the 10-minute training budget without changing the core architecture family (`SSM + E2E TTT + long-context taps`).

## What Changed
- Kept Longworm in its own trainer path: `experiments/Longworm/train_longworm.py`.
- Added adaptive TTT controls:
  - `TTT_ADAPTIVE_ENABLED`
  - `TTT_ADAPTIVE_MIN_STEPS`, `TTT_ADAPTIVE_MAX_STEPS`
  - `TTT_ADAPTIVE_ERR_LOW`, `TTT_ADAPTIVE_ERR_HIGH`
  - `TTT_ADAPTIVE_EMA_DECAY`
  - optional drift gating: `TTT_DRIFT_ENABLED`, `TTT_DRIFT_COEFF`, `TTT_DRIFT_CLAMP`
  - stability clamps: `TTT_UPDATE_CLAMP`, `TTT_WEIGHT_CLAMP`
- Added dedicated v1.9 sweep runner and launcher:
  - `sidelane/scripts/run_v1_9_longworm_adaptive_ablation.sh`
  - `run_longworm_v1_9_adaptive.sh`

## Arm Matrix
- `39_v1_9_longworm_ssm_ttt_baseline_l8_d480_h12_kv4_non_ngram_brotli`
  - fixed TTT steps (`1`), no adaptive gating.
- `40_v1_9_longworm_loss_gated_ttt_l8_d480_h12_kv4_non_ngram_brotli`
  - loss-gated adaptive TTT steps (`1..3`), no drift term.
- `41_v1_9_longworm_loss_drift_gated_ttt_l8_d480_h12_kv4_non_ngram_brotli`
  - loss-gated + drift-gated adaptive TTT steps (`1..3`), slightly deeper SSM/TTT layer coverage.

## V1.9.1 Stability Tuning
- Reduced adaptive aggressiveness:
  - arm `40`: `TTT_STEPS=1`, `TTT_ADAPTIVE_MAX_STEPS=2`, `TTT_ADAPTIVE_ERR_HIGH=0.06`
  - arm `41`: reverted to `SSM_LAST_N=4`, `TTT_LAST_N=2`, plus drift-lite (`COEFF=0.20`, `CLAMP=0.10`)
- Added non-finite guards in TTT fast-weight updates:
  - sanitize gradients/updates with `nan_to_num`
  - clamp update magnitude and fast-weight state
  - preserve adaptive EMA only on finite error values

## Run
```bash
cd /home/frosty40/parameter-golf-lab/neural && TOKENIZER_PATH=/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp1024 NPROC_PER_NODE=1 bash test_lab/PR_sidelane_8x_linked/run_longworm_v1_9_adaptive.sh
```

Single-arm run:
```bash
cd /home/frosty40/parameter-golf-lab/neural && ARM_ONLY=40_v1_9_longworm_loss_gated_ttt_l8_d480_h12_kv4_non_ngram_brotli TOKENIZER_PATH=/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_1024_bpe.model DATA_PATH=/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp1024 NPROC_PER_NODE=1 bash test_lab/PR_sidelane_8x_linked/run_longworm_v1_9_adaptive.sh
```
