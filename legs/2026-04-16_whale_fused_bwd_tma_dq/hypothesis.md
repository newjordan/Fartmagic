# whale_fused_bwd_tma_dq — H8

## Lever B
Replace the scalar `tl.atomic_add(DQ_F32 + offsets, dq_local, mask=q_mask)`
inside `_attn_bwd_fused_kernel` (vault/whale_kernel_triton.py:843, post-Lever-A) with
`descriptor.atomic_add(...)` on a `tl.make_tensor_descriptor` over DQ_F32.
This lowers to the same `cp.reduce.async.bulk.add.f32` (TMA
`SM90_BULK_REDUCE_ADD`) primitive FA3 hopper bwd uses for dQ accumulation
(`mainloop_bwd_sm90_tma_gmma_ws.hpp:644`).

## Why
H5 found the monolithic fused-bwd kernel is gated by dQ atomic-add
bandwidth, not math:

| variant      | shape (4,2048,8,4,64) fb_mean ms |
|:-------------|---------------------------------:|
| fused_delta  | 0.359 (baseline / current best)  |
| fused_bwd    | 0.709  (2.0× slower)             |

The dkdv math is identical between the two; the only difference is that
fused_bwd does `BLOCK_M*BLOCK_N` per-thread fp32 atomics into HBM for dQ
on every M-block iteration, while fused_delta writes dQ once via a
separate dq_inline kernel without atomics. Replacing the per-thread
atomicAdd with TMA bulk reduce-add should collapse that gap.

## Success criteria
- **Primary:** TMA-dq fused_bwd ≤ 0.359 ms fb on (4,2048,8,4,64), i.e.
  beats or matches fused_delta on the headline.
- **Stretch:** ≤ 0.30 ms fb on the headline (clear win) and ≤ FA3
  (`whale_hybrid` fb) on long-T shapes (4,4096,8,4,64), (2,8192,8,4,64).
- **Numerics:** dQ/dK/dV match the baseline within 1e-2 max abs (bf16
  attention bwd tolerance).

## Constraints / risks
- TMA descriptors require BLOCK_D == D, the descriptor base must be
  16-byte aligned, and the inner stride must be 1. DQ_F32 is allocated
  contiguous as `(B, T, H, D)` so `stride_dqfd == 1` and the per-(b,h)
  base `DQ_F32 + b*stride_dqfb + h*stride_dqfh` is fp32-aligned (fp32
  itself is 4 bytes; the base is 16-byte aligned because `T*D*4 % 16 == 0`
  for D∈{64,128} which are typical here).
- `descriptor_atomic_add` accepts `{uint32, int32, uint64, float32,
  float16, bfloat16}` (semantic.py:1117); fp32 is the path we want.
- `maxnreg≥224` corrupted scalar `tl.atomic_add` on this Triton 3.6 stack
  (`_bwd_fused_configs` docstring at vault/whale_kernel_triton.py:174–197).
  TMA atomic_add is a different lowering path (`cp.reduce.async.bulk.add`
  vs `red.global.add`), but until proven safe we cap at maxnreg≤192 in
  the new config list.
- The descriptor must be re-built per (b, h) because the base pointer
  shifts with stride_dqfb / stride_dqfh.
