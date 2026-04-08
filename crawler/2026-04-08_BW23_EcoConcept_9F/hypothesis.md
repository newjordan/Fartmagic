# Hypothesis: BW23_EcoConcept_9F

Date: 2026-04-08  
Track: crawler  
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Objective

Set up an economical concept-validation lane with two mirrored executions:

1. full 2k gate matrix (primary signal), and  
2. DGX Spark smoke matrix (cheap sanity pass with identical arms).

Both run the same arm set so smoke can quickly reject bad ideas before larger runs.

## Concepts under test

1. **Continuous QAT surrogate variants** (training-side)
   - `QAT_SURROGATE=legacy` (STE baseline)
   - `QAT_SURROGATE=softclamp`
   - `QAT_SURROGATE=sigmoidste`

2. **Sensitivity-style mixed-bit export policy** (post-window quant-only)
   - `INT6_CATS` controls which parameter categories remain int6 (`mlp,attn,aux` baseline).

## One-variable discipline

- Window stage isolates QAT behavior:
  - QAT on/off baseline check
  - surrogate variant sweeps against QAT legacy
- Quant stage isolates export policy only:
  - fixed checkpoint (`SKIP_TRAIN=1`, `INIT_MODEL_PATH=<best window ckpt>`)
  - only `INT6_CATS` changes.

## Execution

- Full gate matrix:
  - `SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-08_BW23_EcoConcept_9F/gate.sh`
- DGX Spark smoke matrix:
  - `SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-08_BW23_EcoConcept_9F/gate_dgx_spark_smoke.sh`

## Gate target

- Window stage: any treatment with `delta_vs_ctrl <= -0.003` int6_sw_bpb and acceptable step_ms.
- Quant stage: better quality-size tradeoff vs `INT6_CATS=mlp,attn,aux`.
