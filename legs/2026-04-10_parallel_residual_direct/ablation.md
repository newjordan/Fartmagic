# Ablation Log

## Screen (2xH100, 360s, seed=444)
- Command: `SEED=444 NPROC_PER_NODE=2 bash legs/2026-04-10_parallel_residual_direct/run.sh`
- Seed: 444
- val_bpb: **1.3085**
- val_loss: 2.2094
- step: 883/20000
- train_time: 360358ms
- step_avg: 408.11ms
- peak memory: 24687 MiB allocated, 24720 MiB reserved
- artifact size: 10473190 bytes (mixed+brotli)
- Delta vs control: **-0.0096 BPB** (1.3085 vs 1.3181)
- Date: 2026-04-11
- Pod: 2xH100 SXM (fresh pod)
- Notes: GPT-J parallel residual layers 7-11. Positive signal, slightly less memory than control.
