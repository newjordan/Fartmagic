# whale_bwd_tma_dkdv — hypothesis (H2)

## Goal

Test whether the already-scaffolded TMA dkdv variant
(`WHALE_BWD_VARIANT=fused_delta_tma`, kernel
`_attn_bwd_dkdv_inline_delta_tma_kernel` in `vault/whale_kernel_triton.py`)
is faster than the current winner `fused_delta` on the headline shape.

The TMA path uses `tl.make_tensor_descriptor` for K/V/Q/O/DO/DK/DV loads
in the dkdv kernel; the dq path stays as the existing
`_attn_bwd_dq_inline_delta_kernel`. Lower instruction count + hardware
prefetch should reduce dkdv kernel time below 194 µs (current per
`whale_bwd_ablations` profiler snapshot).

`whale_bwd_ablations/RESULTS.md` notes TMA as "high risk, unclear
payoff, did not attempt." Evidence shows the kernel is already wired
up (see `vault/whale_kernel_triton.py:_attn_bwd_dkdv_inline_delta_tma_kernel`,
launch site at line ~1255 under `use_tma_dkdv=True`). All that remains
is to benchmark.

## Variants and constraints

- TMA requires `BLOCK_D == D`, which holds when D is a power of 2 (D∈{32,64,128}).
- TMA is dispatched per-shape via:
    `WHALE_BWD_VARIANT=fused_delta_tma` → `use_tma_dkdv=True`.
  If the shape doesn't satisfy `BLOCK_D == D`, the kernel falls back to
  non-TMA `fused_delta` automatically (line 1201).
- Config list: `_bwd_kv_inline_tma_configs()` — same as inline non-TMA but
  inherits `maxnreg=224`. Per Triton-gotcha findings,
  `maxnreg=224` only corrupts atomic_add, not plain stores. Since this
  kernel does plain `tl.store` on dk/dv (not atomic_add), `maxnreg=224`
  should remain safe here.

## Success criterion

Pure-Triton `whale fused_delta_tma` wall < `fused_delta` wall on the
headline (current 0.359 ms whale_fast). Stretch: beat FA3 (0.353 ms).
Correctness within bf16 tol on every shape.

## What this does NOT test

- TMA on dq kernel (would need a separate scaffold).
- Persistent fused dkdv+dq (covered and ruled out by H5).
- Multi-GPU (out of scope until 1×H100 phase concludes).
