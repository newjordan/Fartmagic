# B02: Ternary Quantization

## Base

- Base ID: `BASE_TERNARY`
- Path: `records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/train_gpt_cuda_ternary.py`
- Current reference score: `val_bpb=1.1565`

## TV0 (Run Now)

```bash
cd /home/frosty40/parameter-golf-lab/records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon
bash run_cuda_ternary.sh
```

## Ablation Ladder

| Arm | Change | Goal |
|---|---|---|
| A0 | Current ternary recipe | Reproduce 3-seed behavior |
| A1 | `BITNET_GROUP_SIZE: 128 -> {64,256}` | Ternary grouping sensitivity |
| A2 | `SMEAR: 0 -> 1` | Test if ternary gains from smear gates like binary |
| A3 | `SLIDING_EVAL_STRIDE: 16 -> {32,64}` | Verify eval sensitivity and calibration |
| A4 | `MUON_MOMENTUM` + `WARMUP_STEPS` sweep | Improve optimizer stability for ternary plateaus |

## Code Touchpoints

- `train_gpt_cuda_ternary.py`: ternary projection + optimizer + quant artifact path

## References (Local Notes)

- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.22592_pquant-towards-effective-low-bit-language-models-via-decoupled-linear-quantization-aware-training_20260402_005555.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2603.27914_itq3-s-high-fidelity-3-bit-llm-inference-via-interleaved-ternary-quantization-with-rotation-domain-smoothing_20260401_130706.md`

