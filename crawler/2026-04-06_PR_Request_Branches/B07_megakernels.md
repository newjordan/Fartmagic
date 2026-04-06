# B07: Megakernels

## Base

- Base ID: `BASE_RASCAL`
- Path: `records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py`
- Current reference score: `val_bpb=1.1099`

## TV0 (Runnable Kernel-First Variant)

```bash
cd /home/frosty40/parameter-golf-lab
SKIP_GPTQ=1 MLP_KERNEL_MODE=triton_act COMPILE_ENABLED=1 COMPILE_MODE=max-autotune \
SEED=42 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py
```

## Ablation Ladder

| Arm | Change | Goal |
|---|---|---|
| A0 | Rascal control | Baseline step time |
| A1 | Triton activation kernel only | Isolated kernel speedup |
| A2 | + compile mode autotune | Compound kernel/codegen gain |
| A3 | + kernel fusion for quant/dequant path (new) | End-to-end throughput |

## Code Touchpoints

- `train_gpt.py`: triton activation kernel paths, compile toggles
- Quant path: int6 projection + dequant hot loops

## References (Local Notes)

- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2603.06728_orion-characterizing-and-programming-apple-s-neural-engine-for-llm-training-and-inference_20260403_023002.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2603.02597_gputok-gpu-accelerated-byte-level-bpe-tokenization_20260401_051042.md`

