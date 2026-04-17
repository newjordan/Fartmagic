# H10 — Lever D: T-keyed autotune config dispatch (long-T configs)

## Hypothesis
**fact**: The kernels in `vault/whale_kernel_triton.py` already include `T_MAX`
in their `@triton.autotune(... key=[...])` lists, so Triton autotune *will*
re-tune (pick a different `triton.Config`) for each distinct `T_MAX` value.

**fact**: All eight config functions (`_fwd_configs`, `_fwd_tma_configs`,
`_bwd_kv_configs`, `_bwd_kv_inline_configs`, `_bwd_kv_inline_tma_configs`,
`_bwd_q_configs`, `_bwd_kv_split_h_configs`, `_bwd_fused_configs`) currently
return the **same** Cartesian product list regardless of any T-context. The
T-keyed re-tuning only re-selects from a fixed pool that is biased toward
short-T tile sizes (BLOCK_M/BLOCK_N at 64 dominate the candidate count).

**fact**: H6 long-T profile shows whale loses 1.5–1.7x to FA3 at T>=4096
across both fwd and bwd at shapes (4,4096,8,4,64), (2,8192,8,4,64),
(4,2048,16,16,64), (2,2048,8,4,128).

**fact**: FA3 hopper bwd uses BLOCK_M=BLOCK_N=128, num_stages=2, num_warps=8
(2 MMA WGs) for hdim=64 long T.

**inference**: Whale autotune is exploring too many small-tile candidates and
likely picking a small-tile winner that is *locally* fast but globally
suboptimal vs FA3's larger-tile, fewer-stage configs at long T. Forcing the
long-T autotune pool to be {(128,128), (128,256), (256,128)} x {warps=8} x
{stages=2,3} should let the right config win.

## Proposal
**proposal**: Add an opt-in `WHALE_LONGT_CONFIGS=1` env that swaps the
**entire** autotune list each function returns to a long-T-biased list. No
kernel rewrites; no per-call branching inside `@triton.autotune` (Triton
autotune keys are read at compile-time, not by the config-list builder).

This leg is doc-only; the patch lives in `vault_patch.md`. Implementation
into `vault/whale_kernel_triton.py` is gated on this leg's bench evidence.

## Success criterion
Any whale variant (`auto`, `baseline`, `fused_delta`) gains **>=10%** at
T=8192 (shape `2,8192,8,4,64`) vs the same variant with the existing
(short-T-biased) config list, measured by `bench_stable.py` mean
fwd-or-fwd+bwd ms.

## Risk surface
- **Correctness**: zero. Still autotuned over a list of valid `triton.Config`
  values. Same kernels, same call sites.
- **maxnreg=224 atomic_add corruption**: the long-T list for
  `_bwd_fused_configs` MUST keep the existing `mr < 224` guard.
  See triton_gotchas memory note + `legs/2026-04-16_whale_bwd_persistent_atomic`.
- **Compile time**: long-T list is *smaller* than current (4 BMxBN x 1-2
  warp x 2-3 stage = ~12 configs vs 24-48 today). Autotune cost should
  drop, not rise.

## Out of scope for this leg
- Per-shape branching inside the kernel.
- TMA on/off toggling (orthogonal lever).
- Any change to FA3 hybrid path.
