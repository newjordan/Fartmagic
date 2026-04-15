# Ablation Log

## Gate (1xGPU, 2000 steps)
- Command: `cd /workspace/parameter-golf && SEED=444 NPROC_PER_NODE=1 bash legs/2026-04-14_the_witch/gate.sh`
- Seed: 444
- Best proxy metric:
- Verdict:
- Notes:

## Full Run (8xH100, 600s, seed=444)
- Command: `cd /workspace/parameter-golf && SEED=444 NPROC_PER_NODE=8 bash legs/2026-04-14_the_witch/run.sh`
- final_sliding_window_exact val_bpb:
- final_quant_roundtrip_exact val_bpb:
- Total submission size:
- step_avg:
- Verdict:

## Confirmation (8xH100, 600s, seed=300)
- Command: `cd /workspace/parameter-golf && SEED=300 NPROC_PER_NODE=8 bash legs/2026-04-14_the_witch/run.sh`
- final_sliding_window_exact val_bpb:
- final_quant_roundtrip_exact val_bpb:
- Verdict:
