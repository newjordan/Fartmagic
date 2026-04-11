# Ablation Log

## Screen (2xH100, 360s, seed=444)
- Command: `SEED=444 NPROC_PER_NODE=2 bash legs/2026-04-10_depth_recurrence_direct/run.sh`
- Seed: 444
- val_bpb: **1.3201**
- val_loss: 2.2290
- step: 706/20000
- train_time: 360218ms
- step_avg: 510.22ms
- peak memory: 36860 MiB allocated, 37138 MiB reserved
- artifact size: 9712747 bytes (mixed+brotli)
- Delta vs control: **+0.0020 BPB** (1.3201 vs 1.3181) — WORSE
- Date: 2026-04-11
- Pod: 2xH100 SXM (fresh pod)
- Notes: Depth recurrence layers 3-5, 2 loops. NEGATIVE result. Loop activation at step 314 (frac=0.351) cratered throughput from ~410ms to ~510ms/step. Memory spiked 25GB→37GB. Lost ~90 steps vs control. The per-step quality gain from recurrence did not compensate for lost steps under wallclock cap.
