# Ablation Log

## Screen (2xH100, 360s, seed=444)
- Command: `SEED=444 NPROC_PER_NODE=2 bash legs/2026-04-10_stack_core_direct/run.sh`
- Seed: 444
- val_bpb: **1.3243**
- val_loss: 2.2360
- step: 701/20000
- train_time: 360075ms
- step_avg: 513.66ms
- peak memory: 36573 MiB allocated, 36648 MiB reserved
- artifact size: 8218971 bytes (mixed+brotli)
- Delta vs control: **+0.0062 BPB** (1.3243 vs 1.3181) — WORSE
- Date: 2026-04-11
- Pod: 2xH100 SXM (fresh pod)
- Notes: Full stack (qk_gain=5.25, matrix_lr=0.022, muon_wd=0.095, parallel_residual, depth_recurrence). NEGATIVE result on 6-min screen. Depth recurrence throughput penalty (513ms/step) dominated — lost ~96 steps vs control. All hyperparameter gains eaten by lost steps. Depth recurrence may need longer runs or 8xH100 parallelism to pay for itself.
