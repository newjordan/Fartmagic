# vault_patch.md — DO NOT auto-apply

This leg uses the **existing** `_attn_bwd_dkdv_split_h_kernel` and
`WHALE_BWD_SPLIT_H=1` dispatch already in `vault/whale_kernel_triton.py`.

**No vault edit is required to run the bench.** This file documents what
the existing code does so an implementer can verify the lever is being
exercised end-to-end, plus the patch that *would* be needed if we
decided to follow up with a fresh GQA-to-grid kernel that targets the
inline-Δ path (which the existing split-H kernel does NOT cover).

Target file: `vault/whale_kernel_triton.py`
Existing kernel: `_attn_bwd_dkdv_split_h_kernel` (declared L918)
Existing dispatch: `_whale_attn_bwd_impl` `if use_split_h:` branch
(L1372-L1394), gated by `WHALE_BWD_SPLIT_H=1`.

## Section 1 — what already exists in vault (no change required)

### Kernel signature (vault L918-L944, exact)
```python
@triton.autotune(configs=_bwd_kv_split_h_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_bwd_dkdv_split_h_kernel(
    Q, K, V, DO, DK_F32, DV_F32, LSE, DELTA,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_dob, stride_dot, stride_doh, stride_dod,
    stride_dkb, stride_dkt, stride_dkh, stride_dkd,
    stride_dvb, stride_dvt, stride_dvh, stride_dvd,
    stride_lb, stride_lh, stride_lt,
    stride_db, stride_dh, stride_dt,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)        # <-- B * NUM_HEADS, not B * NUM_KV_HEADS
    b = bh // NUM_HEADS
    h = bh % NUM_HEADS
    kv_h = h // (NUM_HEADS // NUM_KV_HEADS)
```
The Q-head is decoded from `program_id(1)` directly; **no inner `for hg`
loop**. Each program owns one (b, h, n_block) tuple and writes its
fp32 partial dk, dv into `DK_F32`, `DV_F32` at slot `h` of the H-axis.

### Grid and dispatch (vault L1372-L1394, exact)
```python
if use_split_h:
    group = H // KV
    # Per-Q-head partial workspace laid out as [B, T, H, D] to match q strides.
    dk_part = torch.empty((B, T, H, D), device=q.device, dtype=torch.float32)
    dv_part = torch.empty((B, T, H, D), device=q.device, dtype=torch.float32)
    grid_kv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * H)   # <-- B*H not B*KV
    _attn_bwd_dkdv_split_h_kernel[grid_kv](
        q, k, v, do_c, dk_part, dv_part, lse, delta,
        ...
    )
    # Reduce per-head partials over each KV group: [B, T, KV, group, D] -> [B, T, KV, D]
    dk.copy_(dk_part.view(B, T, KV, group, D).sum(dim=3).to(dk.dtype))
    dv.copy_(dv_part.view(B, T, KV, group, D).sum(dim=3).to(dv.dtype))
```

### Activation
```bash
WHALE_BWD_SPLIT_H=1
```
Set in `tracked_env.sh`. The dispatch hits the `use_split_h` branch
(vault L1372) and skips the `for hg` kernel.

### Caveats (read before launching)
1. The split-H path runs the **baseline** dkdv kernel (`use_fused_delta=False`),
   so `WHALE_BWD_VARIANT` must NOT be `fused_delta` / `fused_delta_tma` /
   `fused_bwd`. The default `auto` picks `use_fused_delta=True` for
   `T <= WHALE_FUSED_DELTA_T_MAX` (default 3072). For T=8192 (primary)
   and T=4096 (secondary), `auto` already disables fused-delta, so split-H
   activates correctly. For T=2048 / T=1024 (the two launch-bound
   secondaries), `auto` picks fused-delta and **split-H is a no-op**. To
   force split-H at those shapes set `WHALE_BWD_VARIANT=baseline`.
2. The split-H autotune cache key is the same as the non-split kernel
   (`["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"]`). Flushing
   the cache once before the first AFTER pass is recommended; tracked
   via `WHALE_FLUSH_TRITON_CACHE=1` in the leg env.

## Section 2 — optional: GQA-to-grid for the inline-Δ kernel

Only attempt if the Section 1 bench shows a meaningful BEFORE/AFTER
delta on at least one production shape. Otherwise this is dead code.

The existing `_attn_bwd_dkdv_split_h_kernel` does NOT have an
inline-Δ variant; the inline-Δ kernel (`_attn_bwd_dkdv_inline_delta_kernel`,
vault L559) still runs the inner `for hg` loop. To split-H the inline-Δ
path, a near-clone of `_attn_bwd_dkdv_split_h_kernel` with the inline
`o_tile` load + `delta = sum(o*do, axis=1)` would be needed.

### BEFORE (vault L606-L645, current inline-Δ kernel, unchanged for this leg)
```python
for hg in range(group):
    h = kv_h * group + hg
    for m_block in range(m_start_block, m_end_block):
        ...
        o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)
        delta = tl.sum(o_tile.to(tl.float32) * do.to(tl.float32), axis=1)
        ...
```

### AFTER — proposed `_attn_bwd_dkdv_inline_delta_split_h_kernel` (NEW kernel)
- Replace the `bkv = pid(1)` decomposition with `bh = pid(1); b = bh // H;
  h = bh % H; kv_h = h // group`.
- Drop the `for hg in range(group)` loop entirely.
- Write per-Q-head fp32 partials into a workspace `[B, T, H, D]`,
  identical to `_attn_bwd_dkdv_split_h_kernel`.
- Add a new dispatch branch in `_whale_attn_bwd_impl` for
  `use_fused_delta and use_split_h`, with the same post-kernel
  `view(...).sum(dim=3)` reduce.
- New `_bwd_kv_inline_split_h_configs()` mirroring `_bwd_kv_inline_configs`
  with `maxnreg=224` (no atomics → safe).

This is intentionally not implemented in this leg. It is the **next
follow-up patch** if and only if Section 1's bench shows the lever has
legs at the production shape range.

## Section 3 — atomic_add alternative (REJECTED)

The "atomic_add into final dK/dV instead of post-kernel reduce" path is
**HIGH RISK** and explicitly rejected for this leg:

- `vault/whale_kernel_triton.py` L177-L180:
  > maxnreg=224 breaks atomic_add correctness on this Triton 3.6 stack:
  > `tl.atomic_add` of a 2-D fp32 tile amplified values ~19005x when
  > maxnreg=224 was forced (row*1 + col*1000 probe, observed on H100 SXM,
  > see legs/2026-04-16_whale_bwd_persistent_atomic/hypothesis.md).
- `legs/2026-04-16_whale_pod_autoresearch/RESULTS.md` L78-L80 confirms
  the `tl.atomic_add` split-H attempt produced ~24000x value inflation.
- Bandwidth: atomic combine on dK/dV serializes group programs at the
  N-block tile level; with group=2 and BLOCK_N=128 that is `128 * D * 4B
  = 32-64KB` of atomic traffic per block per pair, on top of the
  existing dK/dV write.

If a future leg wants to revisit atomics, it MUST first rerun the
maxnreg=192 sweep on the current Triton 3.6 stack and verify the
atomic_add correctness probe in the existing
`legs/2026-04-16_whale_bwd_persistent_atomic/` test. That is out of
scope here.

## Numerics tolerance to enforce in `bench_numerics.py`
- max |Δ| (custom vs sdpa) for fwd output: <= 5e-3 in bf16
- max |Δ| for dQ, dK, dV: <= 1e-2 in bf16
- These are the existing tolerances in
  `legs/2026-04-16_whale_fast_kernels/bench_numerics.py`; this leg
  reuses that script verbatim.

## Notes for the implementer
- No vault edit needed to run Section 1.
- Section 2 (new inline-Δ split-H kernel) is gated on Section 1 showing
  a positive delta on a production shape. Implement only then.
- Section 3 is rejected; do not implement without an updated atomic_add
  correctness probe.
