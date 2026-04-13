# TON-E run capture: tone_test_4f3cx2_stage33_d448_s4

Date captured: 2026-04-13
Branch: TON-E
Source: user-pasted raw transcript (`logs/tone_test_4f3cx2_stage33_d448_s4.txt`)

## Configuration
- Topology: `4F+3Cx2` (effective depth 10)
- Schedule:
  - `CRAWLER_SAFE_WARMUP_STEPS=400`
  - `CRAWLER_GRADUATED=1`, `CRAWLER_START_FRAC=0.0`, `CRAWLER_RAMP_DELAY_FRAC=0.33`, `CRAWLER_RAMP_FRAC=0.33`
  - `CRAWLER_FORWARD_GRADUATED=1`, `CRAWLER_FORWARD_START_FRAC=0.0`, `CRAWLER_FORWARD_RAMP_DELAY_FRAC=0.33`, `CRAWLER_FORWARD_RAMP_FRAC=0.33`
- World size: `8` (`grad_accum_steps=1`)
- Tokenizer/data: `fineweb_8192_bpe.model`, `fineweb10B_sp8192` (`train_shards=80`, `val_tokens=40540160`)
- Model params: `18972284`
- Quant mode: `int8_flat`

## Schedule trace (selected)
- step `1`: `crawler_grad_mul=0.000`, `crawler_fwd_mul=0.000`
- step `500`: `crawler_grad_mul=0.000`, `crawler_fwd_mul=0.000`
- step `2000`: `crawler_grad_mul=0.000`, `crawler_fwd_mul=0.000`
- step `2500`: `crawler_grad_mul=0.216`, `crawler_fwd_mul=0.216`
- step `3000`: `crawler_grad_mul=0.460`, `crawler_fwd_mul=0.460`
- step `3500`: `crawler_grad_mul=0.705`, `crawler_fwd_mul=0.705`
- step `4000`: `crawler_grad_mul=0.950`, `crawler_fwd_mul=0.950`
- step `4500+`: `crawler_grad_mul=1.000`, `crawler_fwd_mul=1.000`

## Key metrics
- `raw_bpb`: `1.22377118`
- `quant_sw_bpb`: `1.20943232`
- `final_int8_flat_sliding_window_exact val_bpb`: `1.20943232`
- `val_loss`: `3.12407615`
- `step_avg_ms`: `96.66`
- `steps`: `6208`
- `train_time_s`: `600`
- `peak_memory_allocated_mib`: `20678`
- `peak_memory_reserved_mib`: `22170`

## Size
- `serialized_model_bytes`: `65868943`
- `serialized_model_int8_flat_bytes`: `12481012`
- `code_bytes`: `143625`
- `total_submission_size`: `12624637`
- cap: `16000000`
- under cap by: `3375363`
- `artifact_legal`: `yes`

## Notes
- This run is legal and materially faster than the previous `4F+4Cx2 grad33` test (`step_avg_ms 96.66` vs `112.19`).
- Quality is weaker than the oversized `4F+4Cx2 grad33` test (`raw_bpb 1.2238` vs `1.1278`), but with strong size margin.
