# Crawler Concept Scoring Rubric

Updated: 2026-04-08  
Scope: rank new crawler add-ins before more ablations, using benefit/risk and gate practicality.

## Scoring Model

Score each concept on 0-5 (integer) for each dimension.

### Benefit (higher is better)

- `B1`: Expected int6_sw_bpb gain potential
- `B2`: Quality-per-wallclock potential (better quality at same or lower step_ms)
- `B3`: Artifact-size efficiency potential (helps stay under 16MB)
- `B4`: Stackability with BWX 9F baseline

### Risk (lower is better)

- `R1`: Implementation complexity risk
- `R2`: Training stability risk
- `R3`: Measurement ambiguity risk (hard to isolate causality)
- `R4`: Operational/submission risk (runtime, reproducibility, legal workflow)

### Feasibility (higher is better)

- `F1`: Time-to-first-2k-gate
- `F2`: GPU cost to decide go/no-go
- `F3`: Code surface area (smaller change gets higher score)

### Priority Formula

`Priority = 2*(B1+B2+B3+B4) + (F1+F2+F3) - (R1+R2+R3+R4)`

Interpretation:

- `>= 24`: priority A (run next)
- `16-23`: priority B (queue after A)
- `8-15`: priority C (needs tighter design first)
- `<= 7`: defer

## Ranked Candidate Queue

| Rank | Concept | B1 | B2 | B3 | B4 | R1 | R2 | R3 | R4 | F1 | F2 | F3 | Priority | Priority Band |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Continuous-QAT surrogate (SoftClamp + Sigmoid STE) | 4 | 3 | 3 | 4 | 2 | 2 | 2 | 1 | 4 | 4 | 3 | 31 | A |
| 2 | Sensitivity-based mixed-bit quant policy (MatGPTQ + LeanQuant style) | 4 | 3 | 5 | 4 | 3 | 2 | 2 | 2 | 3 | 3 | 3 | 30 | A |
| 3 | Dynamic loop halting (difficulty-aware compute) | 4 | 5 | 3 | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 2 | 28 | A |
| 4 | Adaptive level signals for recurrent blocks (Ringformer-style) | 4 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 2 | 24 | A |
| 5 | StableQAT ultra-low-bit surrogate | 3 | 3 | 4 | 3 | 3 | 2 | 3 | 2 | 3 | 3 | 2 | 22 | B |
| 6 | Persistent compact memory bridge | 4 | 3 | 2 | 3 | 4 | 3 | 3 | 2 | 2 | 2 | 2 | 19 | B |
| 7 | Single-pass PTQ via rotation transforms | 3 | 3 | 4 | 3 | 3 | 2 | 3 | 2 | 2 | 3 | 3 | 21 | B |
| 8 | Low-rank decomposed QAT (<1% trainable params) | 3 | 2 | 4 | 3 | 3 | 2 | 3 | 2 | 2 | 3 | 2 | 18 | B |
| 9 | Verifier-guided iterative refinement passes | 3 | 4 | 2 | 2 | 4 | 3 | 4 | 3 | 2 | 2 | 1 | 13 | C |
| 10 | Reconstruction-gated test-time training trigger | 3 | 3 | 2 | 2 | 4 | 4 | 4 | 3 | 2 | 2 | 1 | 9 | C |
| 11 | Difficulty-aware bandit controller for loop budget | 3 | 4 | 2 | 2 | 4 | 4 | 4 | 3 | 2 | 2 | 1 | 11 | C |
| 12 | Continual online TTA with drift-control retention anchors | 2 | 3 | 2 | 1 | 5 | 5 | 4 | 3 | 1 | 1 | 1 | 0 | defer |

## Recommended Next 5 Gates (Execution-Ready Order)

Each gate is scoped to one variable on top of BWX 9F or explicit chosen parent.

1. `QAT_SURROGATE=softclamp_sigmoidste`
2. `QUANT_POLICY=sensitivity_map_v1`
3. `DYNAMIC_LOOP_HALTING=1` (eval-only)
4. `LEVEL_SIGNAL_DIM=8`
5. `STABLE_QAT=1`

## Anti-Patterns To Avoid

- Uniform bitwidth across all blocks under a hard 16MB cap.
- Static min-max PTQ without layer sensitivity or loss/error awareness.
- Stacking multiple new mechanisms before individual 2k gate validation.

## Source Anchors

- `/home/frosty40/ml-research-analysis/ml_research_analysis_2024/2410.10849_continuous-approximations-for-improving-quantization-aware-training-of-llms_20260212_082344.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.03537_matgptq-accurate-and-efficient-post-training-matryoshka-quantization_20260402_005940.md`
- `/home/frosty40/ml-research-analysis/archive/_archived_research_analysis_v1/2407.10032_leanquant-accurate-and-scalable-large-language-model-quantization-with-loss-error-aware-grid_20260201_150754.md`
- `/home/frosty40/ml-research-analysis/archive/_archived_research_analysis_v1/2511.22316_singlequant-efficient-quantization-of-large-language-models-in-a-single-pass_20260203_185150.md`
- `/home/frosty40/ml-research-analysis/archive/_archived_research_analysis_v1/2601.19320_stableqat-stable-quantization-aware-training-at-ultra-low-bitwidths_20260203_202937.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.08864_understanding-dynamic-compute-allocation-in-recurrent-transformers_20260403_111041.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2025/2502.13181_ringformer-rethinking-recurrent-transformer-with-adaptive-level-signals_20260210_143925.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2025/2505.00929_compact-recurrent-transformer-with-persistent-memory_20260210_143543.md`
- `/home/frosty40/ml-research-analysis/spot_analyses/technique_extraction/test_time_compute_scaling.md`
- `/home/frosty40/ml-research-analysis/spot_analyses/technique_extraction/test_time_adaptation.md`
- `/home/frosty40/ml-research-analysis/spot_analyses/technique_extraction/continual_online_tta.md`
