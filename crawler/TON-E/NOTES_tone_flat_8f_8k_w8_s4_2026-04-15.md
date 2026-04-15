# TON-E run capture: flat 8F 8k 8xH100

Date captured: 2026-04-15
Branch: TON-E
Source: user-pasted raw transcript
Launcher: `crawler/TON-E/run_8f_flat_8k.py`
Commit anchor: `b345452`
Run ID: `flat_8f_8k_w8_s4_b345452`
Log path on pod: `crawler/TON-E/logs/flat_8f_8k_w8_s4_b345452.txt`

## Configuration
- Topology: `8F flat`
- Features: plain flat model, crawler off
- World size: `8` (`grad_accum_steps=1`)
- Tokenizer/data: `fineweb_8192_bpe.model`, `fineweb10B_sp8192`
- Model params: `24496708`
- Quant mode: `int6`
- Compile: `COMPILE_ENABLED=1`, `COMPILE_FULLGRAPH=1`
- MLP: `relu_sq`, `mlp_leaky_slope=0.5`
- Attention layout: `num_heads=8`, `num_kv_heads=4`

## Schedule trace
- step `500`: `train_loss=3.4036`, `step_avg_ms=69.36`
- step `1000`: `train_loss=3.1106`, `step_avg_ms=69.39`
- step `2000`: `train_loss=3.1749`, `step_avg_ms=69.37`
- step `3000`: `train_loss=3.0523`, `step_avg_ms=69.39`
- step `4000`: `val_loss=3.1468`, `val_bpb=1.2182`
- SWA start: step `7950`
- step `8000`: `val_loss=2.9794`, `val_bpb=1.1534`
- step `8635`: `val_loss=2.9361`, `val_bpb=1.1367`
- Wallclock stop: step `8635` at `600051ms`

## Key metrics
- `raw_bpb`: `1.13665402`
- `quant_sw_bpb`: `1.12816870`
- `val_loss`: `2.91416466`
- `step_avg_ms`: `69.49`
- `steps`: `8635`
- `train_time_s`: `600`
- `peak_memory_allocated_mib`: `18179`
- `peak_memory_reserved_mib`: `19658`
- `artifact_legal`: **yes**

## Size
- `serialized_model_bytes`: `87005846`
- `code_bytes`: `150315`
- `total_submission_size`: `15720893`
- cap: `16000000`
- under cap by: `279107`

## GPTQ
- Loop-aware 2-phase, `256` cal samples, `seq_len=2048`
- Calibrated `50` layers in `10.7s`
- Final exact roundtrip: `val_loss=2.95701557`, `val_bpb=1.14475288`
- Final sliding-window exact: `val_loss=2.91416466`, `val_bpb=1.12816870`

## Notes
- This is the strongest direct comparison against the current `6F+3C` 8k crawler run under the same 600s, 16MB regime.
- Throughput is almost exactly `2x` better than the crawler run (`69.49ms` vs `139.83ms`), yielding `8635` steps vs `4291`.
- Final legal int6 artifact quality is materially better than the crawler result (`1.12817` vs `1.20669` sliding-window exact bpb).
- Peak memory is much lower than the crawler run (`18.2GiB` vs `32.9GiB` allocated), which confirms the flat run still has substantial H100 headroom.

## Raw transcript tail
```text
step:3/20000 train_loss:9.9454 train_time:249ms step_avg:82.87ms
step:4/20000 train_loss:9.2262 train_time:317ms step_avg:79.25ms
step:5/20000 train_loss:8.5683 train_time:385ms step_avg:77.07ms
step:6/20000 train_loss:8.3624 train_time:453ms step_avg:75.56ms
step:7/20000 train_loss:8.0380 train_time:522ms step_avg:74.54ms
step:8/20000 train_loss:7.5631 train_time:590ms step_avg:73.81ms
step:9/20000 train_loss:7.5490 train_time:659ms step_avg:73.22ms
step:10/20000 train_loss:7.3676 train_time:727ms step_avg:72.73ms
step:500/20000 train_loss:3.4036 train_time:34678ms step_avg:69.36ms
step:1000/20000 train_loss:3.1106 train_time:69389ms step_avg:69.39ms
step:1500/20000 train_loss:3.3323 train_time:104038ms step_avg:69.36ms
step:2000/20000 train_loss:3.1749 train_time:138742ms step_avg:69.37ms
step:2500/20000 train_loss:3.3292 train_time:173467ms step_avg:69.39ms
step:3000/20000 train_loss:3.0523 train_time:208181ms step_avg:69.39ms
step:3500/20000 train_loss:3.2141 train_time:242894ms step_avg:69.40ms
step:4000/20000 train_loss:3.1563 train_time:277617ms step_avg:69.40ms
step:4000/20000 val_loss:3.1468 val_bpb:1.2182 train_time:277640ms step_avg:69.41ms
step:4500/20000 train_loss:3.2063 train_time:312361ms step_avg:69.41ms
step:5000/20000 train_loss:3.0611 train_time:347087ms step_avg:69.42ms
step:5500/20000 train_loss:3.1159 train_time:381824ms step_avg:69.42ms
step:6000/20000 train_loss:3.0959 train_time:416539ms step_avg:69.42ms
step:6500/20000 train_loss:3.0593 train_time:451276ms step_avg:69.43ms
step:7000/20000 train_loss:2.9916 train_time:485995ms step_avg:69.43ms
step:7500/20000 train_loss:3.0524 train_time:520711ms step_avg:69.43ms
swa:start step:7950
step:8000/20000 train_loss:3.0749 train_time:555541ms step_avg:69.44ms
step:8000/20000 val_loss:2.9794 val_bpb:1.1534 train_time:555594ms step_avg:69.45ms
step:8500/20000 train_loss:2.9149 train_time:590583ms step_avg:69.48ms
step:8635/20000 val_loss:2.9361 val_bpb:1.1367 train_time:600051ms step_avg:69.49ms
stopping_early: wallclock_cap train_time:600051ms step:8635/20000
peak memory allocated: 18179 MiB reserved: 19658 MiB
gptq:loop-aware 2-phase calibration samples=256 seq_len=2048...
gptq:loop-aware calibrated 50 layers in 10.7s
ema:SKIPPED (SKIP_EMA=1) — using live model weights
DIAGNOSTIC post_ema val_loss:2.9361 val_bpb:1.1367 eval_time:1439ms
Serialized model: 87005846 bytes
Code size: 150315 bytes
Serialized model int6+brotli: 15570578 bytes
Total submission size int6+brotli: 15720893 bytes
Total submission size int8+zlib: 15720893 bytes
final_int6_roundtrip val_loss:2.9570 val_bpb:1.1448 eval_time:11572ms
final_int6_roundtrip_exact val_loss:2.95701557 val_bpb:1.14475288
final_int6_sliding_window val_loss:2.9142 val_bpb:1.1282 stride:64 eval_time:63275ms
final_int6_sliding_window_exact val_loss:2.91416466 val_bpb:1.12816870
final_int8_zlib_roundtrip_exact val_loss:2.91416466 val_bpb:1.12816870

============================================
  RESULT — TON-E rhythm run seed=4
  model_params:  24496708
  raw_bpb:       1.13665402
  quant_mode:    int6
  quant_sw_bpb:  1.12816870
  val_loss:      2.91416466
  step_avg_ms:   69.49
  steps:         8635
  train_time_s:  600
  bytes_total:   15720893  (limit 16000000)
  bytes_code:    150315
  artifact_legal:yes
============================================
```
