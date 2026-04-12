# Ablation Log — Midnight_Rider

## Setup
- Gate: 2xGPU or 4xGPU, 360s wallclock, seed=444
- Base: Midnight III vault source + progressive depth + GPTQ bank fix
- All arms: SKIP_GPTQ=0, COPRIME_MAX_LOADED_SHARDS=80, LATE_ACTIVATE_AT=0.7

## Results

| Arm | Late Layers | Late Quant | steps | step_avg (ms) | pre-quant BPB | quant_roundtrip BPB | artifact bytes | GPTQ/naive |
|-----|-------------|------------|-------|---------------|---------------|---------------------|----------------|------------|
| 2L_int8 | 2 | int8 | | | | | | |
| 2L_int7 | 2 | int7 | | | | | | |
| 3L_int8 | 3 | int8 | | | | | | |
| 3L_int7 | 3 | int7 | | | | | | |

## Notes
