# TON-E run capture: tone_test_4f3cx2_stage33_d448_s4 (rerun)

Date captured: 2026-04-13
Branch context: TON-E
Source: user-pasted raw transcript in chat (run started around 19:24)

## Configuration (from transcript)
- Topology: `4F+3Cx2` (effective depth 10)
- `WORLD_SIZE=8`, `grad_accum_steps=1`
- Tokenizer/data: `fineweb_8192_bpe.model`, `fineweb10B_sp8192` (`train_shards=80`, `val_tokens=40540160`)
- Model params: `18972284`
- Crawler schedule:
  - `crawler_safe_warmup:steps:400`
  - grad/fwd staged: `start_frac=0.000`, `delay_frac=0.330`, `ramp_frac=0.330`
- Quant mode: `int8_flat`
- Compile: `COMPILE_ENABLED=1`, `COMPILE_FULLGRAPH=1`

## Key metrics
- `raw_bpb`: `1.23028190`
- `quant_sw_bpb`: `1.21630584`
- `final_int8_flat_sliding_window_exact val_bpb`: `1.21630584`
- `val_loss`: `3.14183110`
- `step_avg_ms`: `96.66`
- `steps`: `6208`
- `train_time_s`: `600`
- `peak_memory_allocated_mib`: `20676`
- `peak_memory_reserved_mib`: `22152`

## Timing samples from transcript
- step `1`: `step_avg=145.19ms` with `crawler_grad_mul=0.000` and `crawler_fwd_mul=0.000`
- step `500`: `step_avg=95.97ms` with `crawler_grad_mul=0.000` and `crawler_fwd_mul=0.000`
- step `2000`: `step_avg=96.34ms` with `crawler_grad_mul=0.000` and `crawler_fwd_mul=0.000`
- step `2500`: `step_avg=96.41ms` with `crawler_grad_mul=0.217` and `crawler_fwd_mul=0.217`
- step `6208`: `step_avg=96.66ms`

## Size
- `serialized_model_bytes`: `65868943`
- `serialized_model_int8_flat_bytes`: `12864796`
- `code_bytes`: `143625`
- `total_submission_size`: `13008421`
- cap: `16000000`
- under cap by: `2991579`
- `artifact_legal`: `yes`

## Notes
- Compared to the earlier d448 capture in this repo, this rerun has worse quality:
  - earlier note raw/quant_sw: `1.22377118` / `1.20943232`
  - this rerun raw/quant_sw: `1.23028190` / `1.21630584`
- Throughput is essentially unchanged (`step_avg_ms` remains `96.66`).
