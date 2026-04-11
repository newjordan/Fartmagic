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
- Command: `SEED=444 NPROC_PER_NODE=8 bash legs/2026-04-10_midnight_iii/run.sh`
- final val_bpb:
- Delta vs leader (1.10567949):
- Verdict:

## Confirmation (8xH100, 600s, seed=300)
- final val_bpb:
- Verdict:
