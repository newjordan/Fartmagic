# TON-E run capture: NC3 8k sink+fused ramp full 8xH100

Date captured: 2026-04-15
Branch: TON-E
Source: user-pasted raw transcript
Run ID: `tone_nc3_v8k_sink_fused_ramp_full_w8_s4`

## Configuration
- Topology: `7F+3Cx3` (effective depth 16)
- Features: **sink token + fused norm** + staged ramp
- Schedule:
  - `CRAWLER_COMPUTE_STAGED=1`
  - `CRAWLER_GRADUATED=1`, `CRAWLER_START_FRAC=0.0`, `CRAWLER_RAMP_DELAY_FRAC=0.33`, `CRAWLER_RAMP_FRAC=0.33`
  - `CRAWLER_FORWARD_GRADUATED=1`, `CRAWLER_FORWARD_START_FRAC=0.0`, `CRAWLER_FORWARD_RAMP_DELAY_FRAC=0.33`, `CRAWLER_FORWARD_RAMP_FRAC=0.33`
- World size: `8` (`grad_accum_steps=1`)
- Tokenizer/data: `fineweb_8192_bpe.model`, `fineweb10B_sp8192` (`train_shards=80`, `val_tokens=40540160`)
- Model params: `30857812`
- Quant mode: `int6`
- Compile: `COMPILE_ENABLED=1`, `COMPILE_FULLGRAPH=0`
- MLP: `relu_sq`, `mlp_leaky_slope=0.5`

## Schedule trace
- step `1`: `crawler_grad_mul=0.000`, `crawler_fwd_mul=0.000`, `crawler_active:0Lx0R`
- step `500–2500`: `crawler_grad_mul=0.000`, `crawler_fwd_mul=0.000`, `crawler_active:0Lx0R`
- step `3000`: `crawler_grad_mul=0.295`, `crawler_fwd_mul=0.295`, `crawler_active:1Lx1R`
- step `3500`: `crawler_grad_mul=0.995`, `crawler_fwd_mul=0.995`, `crawler_active:3Lx3R`
- step `4000+`: `crawler_grad_mul=1.000`, `crawler_fwd_mul=1.000`, `crawler_active:3Lx3R`

## Key metrics
- `raw_bpb`: `1.16892551`
- `quant_sw_bpb`: `1.18583327`
- `val_loss`: `3.06311761`
- `step_avg_ms`: `146.07`
- `steps`: `4109`
- `train_time_s`: `600`
- `peak_memory_allocated_mib`: `55199`
- `peak_memory_reserved_mib`: `59334`
- SWA start: step `3850`

## Size
- `serialized_model_bytes`: `112326655`
- `code_bytes`: `150315`
- `total_submission_size`: `16150617`
- cap: `16000000`
- over cap by: `150617`
- selective_prune: `pre=17394950`, `post=16150617`, `pruned=3747680 values`
- `artifact_legal`: **no**

## GPTQ
- Loop-aware 2-phase, 256 cal samples, seq_len 2048
- 42 flat layers patched phase1, 22 crawler Hessians phase2, 65 total merged
- 60 GPTQ layers, 0 naive

## Dynamo recompile warnings
- Hit `config.recompile_limit (8)` on all 8 ranks at step ~3000 (crawler loop count change from 0→1→2→3 triggers recompilation)

## Notes
- **ILLEGAL**: over 16MB cap by ~150KB. Selective prune couldn't shed enough.
- **Quality regression**: quant_sw_bpb `1.18583` is worse than 8k legal baseline (`1.15542`). Sink+fused did not help.
- 30.8M params is significantly larger than the legal baseline (18.5M). Params bloat from sink/fused features pushed past size cap.
- Throughput slow: 146ms/step, only 4109 steps in 600s (vs 8723 steps for legal baseline at 68.79ms/step).
- Crawler only active for final ~1100 steps (step 3000→4109). Very little effective crawler training time.
