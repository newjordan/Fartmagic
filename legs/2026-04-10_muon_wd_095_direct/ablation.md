# Ablation Log

## Screen (2xH100, 360s, seed=444)
- Command: `SEED=444 NPROC_PER_NODE=2 bash legs/2026-04-10_muon_wd_095_direct/run.sh`
- Seed: 444
- val_bpb: **1.3160**
- val_loss: 2.2219
- step: 897/20000
- train_time: 360480ms
- step_avg: 401.87ms
- peak memory: 24969 MiB allocated, 25280 MiB reserved
- artifact size: 9038254 bytes (mixed+brotli)
- Delta vs control: **-0.0021 BPB** (1.3160 vs 1.3181)
- Date: 2026-04-11
- Pod: 2xH100 SXM (fresh pod)
- Notes: MUON_WD=0.095 (baseline 0.04). Slight improvement. Also faster step_avg (100 more steps in same 360s).
