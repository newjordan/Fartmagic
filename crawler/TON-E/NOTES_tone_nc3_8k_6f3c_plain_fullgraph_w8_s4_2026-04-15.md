# TON-E run capture: NC3 8k 6F+3C plain fullgraph 8xH100

Date captured: 2026-04-15
Branch: TON-E
Source: user-pasted raw transcript
Launcher: `crawler/TON-E/run_6f3c_8k.py`
Commit anchor: `fe3fe45`
Run ID: `launcher_nc3_8k_6f3c_w8_s4_fe3fe45_20260415_161708`

## Configuration
- Topology: `6F+3Cx3` (effective depth 15)
- Features: plain fullgraph crawler, no ramp
- World size: `8` (`grad_accum_steps=1`)
- Tokenizer/data: `fineweb_8192_bpe.model`, `fineweb10B_sp8192`
- Model params: `28495948`
- Quant mode: `int6`
- Compile: `COMPILE_ENABLED=1`, `COMPILE_FULLGRAPH=1`
- DDP: `find_unused_parameters=1`
- MLP: `relu_sq`, `mlp_leaky_slope=0.5`
- Attention layout: `num_heads=8`, `num_kv_heads=4`
- XSA: `last_11`

## Schedule trace
- Warmup: steps `1-20`
- step `0`: `val_loss=8.9991`, `val_bpb=3.4838`
- step `500`: `train_loss=3.4442`, `step_avg_ms=139.68`
- step `1000`: `train_loss=3.1125`, `step_avg_ms=139.67`
- step `2000`: `train_loss=3.1065`, `step_avg_ms=139.68`
- step `3000`: `train_loss=2.9332`, `step_avg_ms=139.69`
- SWA start: step `3600`
- step `4000`: `val_loss=2.9453`, `val_bpb=1.1402`
- step `4291`: `val_loss=2.9263`, `val_bpb=1.1328`
- Wallclock stop: step `4291` at `600031ms`

## Key metrics
- `raw_bpb`: `1.13284364`
- `quant_sw_bpb`: `1.20668569`
- `val_loss`: `3.11698135`
- `step_avg_ms`: `139.83`
- `steps`: `4291`
- `train_time_s`: `600`
- `peak_memory_allocated_mib`: `32879`
- `peak_memory_reserved_mib`: `32904`
- `artifact_legal`: **yes**

## Size
- `serialized_model_bytes`: `102876713`
- `code_bytes`: `150315`
- `total_submission_size`: `15712032`
- cap: `16000000`
- under cap by: `287968`
- selective_prune: `pre=16860471`, `post=15712032`, `pruned=3547672 values`

## GPTQ
- Loop-aware 2-phase, `256` cal samples, `seq_len=2048`
- Phase 1 patched `36` flat layers
- Phase 2 collected `22` crawler Hessians
- Merged `59` Hessians total
- Calibration time: `13.1s`
- Quantization: `54` GPTQ layers, `0` naive

## Notes
- Clean plain fullgraph run. No crawler staging or graduated ramp is active.
- Throughput stabilized at about `139.8ms/step`, slightly faster than the earlier sink+fused ramp run (`146.07ms/step`).
- Peak memory stayed low for 8xH100 at about `32.9GB`, leaving significant headroom.
- The raw model reached `1.13284` bpb before quantization and finished legal after int6 selective prune.
- Final exact roundtrip quality landed at `val_loss=3.11698135`, `val_bpb=1.20668569`.
