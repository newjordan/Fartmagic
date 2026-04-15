# TON-E run capture: litecrawler 8F 8k 8xH100

Date captured: 2026-04-15
Branch: TON-E
Source: user-pasted raw transcript
Launcher: `crawler/TON-E/run_litecrawler_8k.py`
Commit anchor: `94e7bdd`
Run ID: `litecrawler_8k_w8_s4_94e7bdd`
Log path on pod: `crawler/TON-E/logs/litecrawler_8k_w8_s4_94e7bdd.txt`

## Configuration
- Topology: `8F + LCx3`
- Features: latent slot crawler, no MoE
- World size: `8` (`grad_accum_steps=1`)
- Tokenizer/data: `fineweb_8192_bpe.model`, `fineweb10B_sp8192`
- Model params: `23817537`
- Quant mode: `int6`
- Compile: `COMPILE_ENABLED=1`, `COMPILE_FULLGRAPH=1`
- DDP: `find_unused_parameters=0`
- Litecrawler: `slots=64`, `slot_dim=128`, `chunk=128`, `slot_heads=4`, `slot_mlp_mult=2.0`

## Schedule trace
- Warmup: steps `1-20`
- step `500`: `train_loss=3.4545`, `step_avg_ms=88.01`
- step `1000`: `train_loss=3.1465`, `step_avg_ms=92.52`
- step `2000`: `train_loss=3.2101`, `step_avg_ms=92.82`
- step `3000`: `train_loss=6.3959`, `step_avg_ms=91.21`
- step `4000`: `val_loss=3.1652`, `val_bpb=1.2254`
- SWA start: step `5950`
- step `6632`: `val_loss=2.9676`, `val_bpb=1.1489`
- Wallclock stop: step `6632` at `600049ms`

## Key metrics
- `raw_bpb`: `1.14885062`
- `quant_sw_bpb`: `1.14119435`
- `val_loss`: `2.94781113`
- `step_avg_ms`: `90.48`
- `steps`: `6632`
- `train_time_s`: `600`
- `peak_memory_allocated_mib`: `19101`
- `peak_memory_reserved_mib`: `20882`
- `artifact_legal`: **yes**

## Size
- `serialized_model_bytes`: `86374051`
- `code_bytes`: `169071`
- `total_submission_size`: `15412452`
- cap: `16000000`
- under cap by: `587548`

## GPTQ
- Loop-aware 2-phase, `256` cal samples, `seq_len=2048`
- Calibrated `61` layers in `32.0s`
- Final exact roundtrip: `val_loss=2.99385412`, `val_bpb=1.15901423`
- Final sliding-window exact: `val_loss=2.94781113`, `val_bpb=1.14119435`

## Notes
- This is the first litecrawler baseline after fixing the slot OOM path. It runs cleanly at fullgraph on 8xH100.
- Compute bottleneck is materially improved versus the old full-token crawler. Compared with `6F+3C`, this run is `49.35ms` faster per step (`90.48` vs `139.83`) and gets `2341` more steps in the 600s window (`6632` vs `4291`).
- Final legal int6 artifact quality is dramatically better than the old full crawler (`1.14119` vs `1.20669` sliding-window exact bpb).
- It still trails the plain flat `8F` baseline, which finished at `1.12817` sliding-window exact bpb in `69.49ms` with `8635` steps.
- Peak memory is close to the flat baseline (`19.1GiB` vs `18.2GiB` allocated), which suggests the latent slot design removed the catastrophic recurrence overhead.
- Artifact size is comfortably under cap, leaving about `0.59MB` of headroom for more crawler capacity or MoE experiments.
