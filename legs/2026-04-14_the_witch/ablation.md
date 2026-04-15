# Ablation Log

## Gate (1xGPU, 2000 steps)
- Command: `cd /workspace/parameter-golf && SEED=444 NPROC_PER_NODE=1 bash legs/2026-04-14_the_witch/gate.sh`
- Seed: 444
- Best proxy metric:
- Verdict:
- Notes:

## Full Run (8xH100, 600s, seed=444)
- Command: `cd /workspace/parameter-golf && SEED=444 NPROC_PER_NODE=8 bash legs/2026-04-14_the_witch/run.sh`
- Log: `legs/2026-04-14_the_witch/logs/full_seed444_20260415_032444.log`
- final_sliding_window_exact val_bpb: 1.14479486
- final_quant_roundtrip_exact val_bpb: 2.29939048
- post_ema val_bpb: 1.1685
- Total submission size: 9,117,626 bytes
- step_avg: 224.99ms (2× slower than 12L at 105ms)
- Steps reached: 2535/20000 (wallclock cap at 570s)
- Peak memory: 50117 MiB allocated / 50846 MiB reserved
- Verdict: FAIL — quant gap 1.131 BPB (threshold 0.05). 3 physical layers amplify quant error 4× through loop.

## Confirmation (8xH100, 600s, seed=300)
- Command: `cd /workspace/parameter-golf && SEED=300 NPROC_PER_NODE=8 bash legs/2026-04-14_the_witch/run.sh`
- final_sliding_window_exact val_bpb:
- final_quant_roundtrip_exact val_bpb:
- Verdict:
