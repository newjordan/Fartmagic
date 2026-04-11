# Ablation Log

## Screen (2xH100, 360s, seed=444) — component arms
See individual leg results in:
- legs/2026-04-10_control_direct/ablation.md (1.3181)
- legs/2026-04-10_matrix_lr_022_direct/ablation.md (1.3045)
- legs/2026-04-10_parallel_residual_direct/ablation.md (1.3085)
- legs/2026-04-10_muon_wd_095_direct/ablation.md (1.3160)
- legs/2026-04-10_depth_recurrence_direct/ablation.md (1.3201)
- legs/2026-04-10_stack_core_direct/ablation.md (1.3243)

## Full Run (8xH100, 600s, seed=444)
- Command: `SEED=444 bash legs/2026-04-10_midnight_iii/run.sh`
- Pod: 8xH100 SXM (vast.ai instance 34584159)
- final_sliding_window_exact val_bpb: **1.10616680**
- final_sliding_window_exact val_loss: 1.86771136
- post_ema val_bpb: 1.1301
- quant_roundtrip val_bpb: 1.55423085 (quant gap: ~0.45 BPB)
- step: 4297/20000
- train_time: 570058ms
- step_avg: 132.66ms
- peak memory: 36463 MiB allocated, 36510 MiB reserved
- artifact size: 12672492 bytes (mixed+brotli)
- layer_loop activated: step 1977 (frac=0.350)
- late_qat activated: step 4046 (scale=0.1498)
- swa started: step 3950
- Delta vs leader (1.10567949): **+0.00048731 BPB** — DOES NOT BEAT
- Date: 2026-04-11
- Verdict: VERY CLOSE. Within 0.0005 BPB of leader. Does not promote but indicates strong potential.

## Confirmation (8xH100, 600s, seed=300)
- final val_bpb:
- Verdict:
