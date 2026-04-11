# Ablation Log

## Screen (2xH100, 360s, seed=444)
- Command: `SEED=444 NPROC_PER_NODE=2 bash legs/2026-04-10_control_direct/run.sh`
- Seed: 444
- val_bpb: **1.3181**
- val_loss: 2.2256
- step: 797/20000
- train_time: 360424ms
- step_avg: 452.23ms
- peak memory: 24980 MiB allocated, 25294 MiB reserved
- artifact size: 10126175 bytes (mixed+brotli)
- Date: 2026-04-11
- Pod: 2xH100 SXM (vast.ai instance 34574274 → replaced by fresh pod)
- Notes: Clean control run. Baseline for all other screen arms.
