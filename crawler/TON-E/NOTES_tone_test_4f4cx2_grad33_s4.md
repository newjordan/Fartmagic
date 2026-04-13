# TON-E run capture: tone_test_4f4cx2_grad33_s4

Date: 2026-04-13
Branch: TON-E

## Configuration
- Topology: `4F+4Cx2` (effective depth 12)
- Schedule: `CRAWLER_GRADUATED=1`, `CRAWLER_START_FRAC=0.33`, `CRAWLER_RAMP_FRAC=0.33`
- World size: `8`
- Train budget: `600s` wallclock
- Quant mode: `int8_flat`

## Key metrics
- `raw_bpb`: `1.12784530`
- `quant_sw_bpb`: `1.11260324`
- `final_int8_flat_sliding_window_exact val_bpb`: `1.11260324`
- `val_loss`: `2.87395762`
- `steps`: `5348`
- `step_avg_ms`: `112.19`

## Size
- `code_bytes`: `135269`
- `total_submission_size`: `21201741`
- cap: `16000000`
- over cap: `5201741`
- `artifact_legal`: `no`

## Notes
- This run appears to be the strongest crawler-quality signal so far in TON-E tests.
- Primary blocker is artifact size, not model quality.
