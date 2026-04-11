# Ablation Log

## Screen (2xH100, 360s, seed=444)
- Command: `SEED=444 NPROC_PER_NODE=2 bash legs/2026-04-10_matrix_lr_022_direct/run.sh`
- Seed: 444
- val_bpb: **1.3045**
- val_loss: 2.2027
- step: 896/20000
- train_time: 360105ms
- step_avg: 401.90ms
- peak memory: 24969 MiB allocated, 25280 MiB reserved
- artifact size: 10186402 bytes (mixed+brotli)
- Delta vs control: **-0.0136 BPB** (1.3045 vs 1.3181)
- Date: 2026-04-11
- Pod: 2xH100 SXM (fresh pod)
- Notes: MATRIX_LR=0.022 (baseline 0.025). Strong improvement. Best single-variable arm so far.
