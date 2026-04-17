# ready.md — Lever B (fused_bwd_tma_dq) apply checklist

Status: **ready to apply as-is.** Line numbers in `vault_patch.md` have been
re-verified against the post-Lever-A vault (1533 lines,
`vault/whale_kernel_triton.py`). All Triton 3.6.0 API constraints have been
confirmed from the installed source under
`/home/frosty40/miniconda3/lib/python3.13/site-packages/triton/`.

## Exact line numbers to use when applying

| Target site (post-Lever-A vault)                                 | Line(s)   |
|------------------------------------------------------------------|-----------|
| `_ensure_tma_allocator` helper (host-side)                       | 37–47     |
| `_bwd_fused_configs()` (ends)                                    | 174–206   |
| **Insert `_bwd_fused_tma_dq_configs()` after this line**         | **206**   |
| `_attn_bwd_fused_kernel` decorator                               | 753       |
| `_attn_bwd_fused_kernel` def                                     | 759       |
| `_attn_bwd_fused_kernel` body — dq loop BEFORE block              | 807–847   |
| `_attn_bwd_fused_kernel` body ends                               | 854       |
| **Insert `_attn_bwd_fused_tma_dq_kernel` after this line**       | **854**   |
| `_attn_bwd_dq_cast_kernel` decorator (don't cross this)           | 857       |
| Existing descriptor-in-kernel precedent (Q/O/DO per-hg)          | 696–707   |
| Dispatch `variant = os.environ.get(...)`                          | 1231      |
| Dispatch `use_kv_warpspec` (Lever A — preserve!)                  | 1233      |
| Dispatch — BEFORE flag block to replace                          | 1231–1253 |
| Dispatch — `if use_fused_bwd:` begins                            | 1258      |
| Dispatch — `dq_f32 = torch.zeros(...)`                            | 1259      |
| Dispatch — fused-kernel launch (args)                            | 1261–1274 |
| Dispatch — cast-kernel launch + `return`                         | 1275–1285 |

## Config count

`_bwd_fused_tma_dq_configs()` generates **17 configs total** (matches
`_bwd_fused_configs()` cardinality): 16 configs × {BM, BN}∈{64,128}² ×
num_warps∈{4,8} × num_stages∈{3,4} + 1 fallback (BM=BN=64, w=4, s=2), all
with `maxnreg=192`.

## Key Triton 3.6 constraints confirmed

- `block_shape` every dim must be a power of 2
  (`triton/_utils.py:48–58`). BLOCK_M ∈ {64, 128} and D (pow2, gated) OK.
- Last-dim block ≥ 16 bytes (`language/semantic.py:1938–1941`). D*4 ≥ 16.
- `strides[-1] == 1` (`language/semantic.py:1943–1945`). Held by
  `torch.zeros(B,T,H,D)`.
- Base pointer 16-byte aligned (docstring `language/core.py:2277–2280`).
  Held by PyTorch caching allocator (min block align 512 B).
- fp32 is a supported dtype for `descriptor.atomic_add`
  (`language/semantic.py:1115–1120`); **no `_has_native_tma` gate
  needed** for fp32 add (the gate only triggers for fp16/bf16 min/max).
- `descriptor.atomic_add` has no `mask=` kwarg
  (`language/core.py:1410`) — tile is pre-zeroed via
  `tl.where(q_mask, dq_local, 0.0)`.
- Per-program descriptor creation inside `@triton.jit` is already in
  production at vault lines 696–707 (`_attn_bwd_dkdv_inline_delta_tma_kernel`
  builds Q/O/DO descriptors per `hg` iteration).
- `_ensure_tma_allocator()` must be called **host-side** before launch
  (it calls `triton.set_allocator(...)`); the patch does this correctly.

## Remaining unknowns (flag for first bench)

1. TMA atomic-reduce correctness at `num_stages >= 4` on this pod's
   Triton 3.6/cu130 build — not independently verified. **Mitigation:
   first correctness bench should pin num_stages=3, num_warps=4,
   BLOCK_M=BLOCK_N=64, maxnreg=192** via `WHALE_BWD_FUSED_CONFIG` or
   pinned autotune list before enabling the full sweep.
2. `maxnreg` interaction with TMA reduce: the 224 scalar-atomic bug is
   a different code path (`semantic.atomic_add` vs
   `create_descriptor_reduce`). 192 is a conservative cap with no
   evidence either way for TMA. Fallback order: 192 → 128 → pin a
   single safe config.
3. Numerical parity target: dq/dk/dv max-abs vs `WHALE_BWD_VARIANT=
   fused_bwd` must match to 1e-4 on bf16 inputs before any perf claim.

## Apply order (when GPU frees)

1. Insert `_bwd_fused_tma_dq_configs()` at line 207 (right after 206).
2. Insert `_attn_bwd_fused_tma_dq_kernel` at line 855 (right after 854,
   before the blank line preceding `_attn_bwd_dq_cast_kernel` at 857).
3. Patch the dispatch block 1231–1253 and the `if use_fused_bwd:` block
   starting at 1258 per Change 3. Preserve Lever A's
   `use_kv_warpspec = ...` line at 1233.
4. Run `python3 scripts/leg_diff_guard.py legs/2026-04-16_whale_fused_bwd_tma_dq`
   on the vault-modified tree (if the guard covers the vault).
5. First bench: `bash legs/2026-04-16_whale_fused_bwd_tma_dq/run.sh`
   with a single pinned config via `WHALE_BWD_FUSED_CONFIG` to verify
   numerics; then full autotune.
