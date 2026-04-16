# whale_bwd_ablations — hypothesis

## Goal

Beat FA3's backward (pure-Triton, no FA3 dep) on H100 at the headline shape
`B=4, T=2048, H=8, KV=4, D=64` causal bf16. Baseline numbers (from
`legs/2026-04-16_whale_pod_autoresearch/evidence/final_sweep.json`):

| backend    | fwd ms | fwd+bwd ms | bwd-only (derived) |
|:-----------|-------:|-----------:|-------------------:|
| whale_fast | 0.080  | 0.449      | 0.369              |
| fa3        | 0.092  | 0.352      | 0.260              |

Gap on backward wall: **~109 µs**.

Kernel-time breakdown of current pure-Triton backward:

| kernel      | µs (approx) |
|:------------|------------:|
| preprocess  |   ~20       |
| dkdv        |  ~130       |
| dq          |   ~76       |
| sum         |  ~226       |

Wall is 369 µs ⇒ ~140 µs is non-kernel overhead (3 bwd launches + 4 empty_like
+ autograd glue).

## Hypotheses, in priority order

1. **H1 (fuse Δ into dq):** computing `Δᵢ = <oᵢ, doᵢ>` inside the dq kernel
   kills a whole kernel launch and 1 HBM roundtrip on the Δ tensor. Expected
   save: 30–50 µs wall.
2. **H2 (TMA loads):** `tl.make_tensor_descriptor` enables hardware TMA on
   K/V/dO in dkdv and Q/dO in dq. Lower instruction count, better prefetch.
   Expected save: 20–40 µs.
3. **H3 (maxnreg tuning):** explicit `maxnreg` on bwd kernels may improve
   occupancy vs compiler default. Expected save: 10–20 µs if it helps.
4. **H4 (wider autotune for dkdv):** current winner is (64,64,4,3). Try
   (128,64,4,3), (128,128,8,3), (64,128,4,3) with explicit tuning.
5. **H5 (single fused persistent bwd):** one kernel that visits each KV
   block, computes dK/dV and accumulates dQ via `tl.atomic_add` to a global
   buffer. Eliminates 2 of 3 launches. Risky because atomic_add had
   correctness issues on this stack earlier, but dq (dense float32) is the
   right shape for it.

## Success criterion

Pure-Triton `whale` backward wall ≤ FA3 backward wall (≤ 0.260 ms on the
headline) while keeping correctness within bf16 tolerance vs SDPA.

## Variables

Each ablation edits `vault/whale_kernel_triton.py` directly. Benchmarks run
via `bench_stable.py` (copied from the autoresearch leg). All configs are
tracked in `tracked_env.sh` + direct edits.
