# Run Report: V1.3 Non-Ngram Brotli Sweep

Date: 2026-04-07 (America/Chicago)  
Project codename: Longworm  
Runner: `test_lab/PR_sidelane_8x_linked/run_v1_3_non_ngram.sh`  
Sweep output root:
`/workspace/parameter-golf/neural/test_lab/PR_sidelane_8x_linked/sidelane/runs/v1_3_submission_sweep_non_ngram_brotli/20260407_211357_s445`

## Environment Snapshot

- Hardware: `8x H100 80GB`
- `NPROC_PER_NODE=8`
- `submission_profile=track_10min_16mb`
- `iterations=12000`
- `max_wallclock_seconds=590`
- Dataset at run time: `train_shards=8`, `val_shards=1`

## Arm Results

### Arm 23 (control)

- Arm: `23_v1_3_schedule_control_non_ngram_brotli`
- `model_params=14695866`
- `final_sliding_window_exact val_bpb=1.26288672`
- `DIAGNOSTIC post_ema val_bpb=1.2868`
- `final_quant_roundtrip_exact val_bpb=1.28765825`
- Submission size: `10391678 bytes` (`~10.39 MB`)
- Quant roundtrip penalty was low and stable.

### Arm 24 (candidate, winner)

- Arm: `24_v1_3_depth12_non_ngram_brotli`
- `model_params=17355468`
- `final_sliding_window_exact val_bpb=1.23690539`  <-- best in this sweep
- `DIAGNOSTIC post_ema val_bpb=1.2613`
- `final_quant_roundtrip_exact val_bpb=1.26261795`
- Submission size: `12158810 bytes` (`~12.16 MB`)
- Stayed below 16 MB with improved quality over control.

### Arm 25 (candidate, failed)

- Arm: `25_v1_3_depth14_dim352_non_ngram_brotli`
- Failure:
  - `ValueError: model_dim must be divisible by num_heads`
- Secondary status:
  - `compression_check_failed` (expected after train failure)

## Sweep Outcome

- Summary file:
  - `/workspace/parameter-golf/neural/test_lab/PR_sidelane_8x_linked/sidelane/runs/v1_3_submission_sweep_non_ngram_brotli/20260407_211357_s445/summary.tsv`
- Leaderboard file:
  - `/workspace/parameter-golf/neural/test_lab/PR_sidelane_8x_linked/sidelane/runs/v1_3_submission_sweep_non_ngram_brotli/20260407_211357_s445/leaderboard.tsv`
- Best arm:
  - `24_v1_3_depth12_non_ngram_brotli`
  - `metric_bpb=1.23690539`

## Key Takeaways

1. Quantization is not the bottleneck. The quant penalty remained very small.
2. Quality improved with added depth (`arm 24`) while staying inside size constraints.
3. `arm 25` failed due to invalid dimensionality (`model_dim` / `num_heads` mismatch), not training instability.
4. With `train_shards=8`, data diversity is limited; stronger robustness runs should use more shards when disk allows.

## Immediate Next Steps

1. Keep quant policy (`6/7/7/8/8`) and Brotli path.
2. Re-run stronger power sweep with valid deeper configs.
3. Enforce submission gate for promotion:
   - non-ngram target `val_bpb <= 1.17`
   - size `< 16 MB`
   - pass full legal metric path.
