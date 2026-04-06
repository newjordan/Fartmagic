# Megarascal: Kernel Fusion Speed Ablation

**Parent:** Rascal II (1.1099 BPB, vault/1.110_15.5mb_baseline.py)
**Goal:** More steps in 10-min wallclock via kernel fusion = lower BPB at same wall time.
**Metric:** ms/step (primary), BPB at 2000 steps (secondary — confirms no regression).

## Arms

| Arm | Variable | Mechanism |
|-----|----------|-----------|
| Control | Baseline | COMPILE_MODE="" (default) |
| A | COMPILE_MODE=reduce-overhead | CUDA graphs via torch.compile — eliminates kernel launch overhead |
| B | MLP_KERNEL_MODE=triton_act | Use existing Triton LeakyReLU² kernel (already in code but not default) |
| C | Fused RMSNorm+MLP kernel | Custom Triton: fuse norm → up_proj → activation into one kernel, skip HBM roundtrip |

## Why these arms

- Arm A is zero code change, just env var. CUDA graphs batch kernel launches. Known 10-30% speedup on small models.
- Arm B activates code already written but not enabled by default. Tests whether the Triton activation kernel beats torch.compile's version.
- Arm C is the real megakernel test — fuses RMSNorm + up_proj + LeakyReLU² to avoid materializing the 1536-wide intermediate in HBM.

## Gate

2000 steps, 1×GPU, seed=300. Measure avg ms/step. Any arm that's ≥5% faster proceeds to 4×GPU validation.
