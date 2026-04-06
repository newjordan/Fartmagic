# B01: 1-bit Quantization

## Base

- Base ID: `BASE_BINARY`
- Path: `records/track_non_record_16mb/2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear/train_gpt_cuda_binary.py`
- Current reference score: `val_bpb=1.1239` (non-record)

## TV0 (Run Now)

```bash
cd /home/frosty40/parameter-golf-lab/records/track_non_record_16mb/2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear
bash run_cuda_binary.sh
```

## Ablation Ladder

| Arm | Change | Goal |
|---|---|---|
| A0 | Current binary recipe | Reproduce baseline log quality |
| A1 | `BITNET_GROUP_SIZE: 128 -> {64,256}` | Measure quant noise vs compression tradeoff |
| A2 | `SMEAR: 1 -> 0` and `ROPE_BASE` sweep | Isolate smear contribution under 1-bit |
| A3 | `MUON_BACKEND_STEPS: 3 -> {2,5}` | Stabilize 1-bit training curvature |
| A4 | `TRAIN_SEQ_LEN: 1024 -> 2048` with fixed tokens | Check long-range robustness in binary regime |

## Code Touchpoints

- `train_gpt_cuda_binary.py`: bit-linear kernels, fake quant path, compression pipeline

## References (Local Notes)

- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.22592_pquant-towards-effective-low-bit-language-models-via-decoupled-linear-quantization-aware-training_20260402_005555.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2603.27914_itq3-s-high-fidelity-3-bit-llm-inference-via-interleaved-ternary-quantization-with-rotation-domain-smoothing_20260401_130706.md`

