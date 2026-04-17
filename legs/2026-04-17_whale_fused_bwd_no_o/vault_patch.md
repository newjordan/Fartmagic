# vault_patch — fused_bwd without O reload

**Status:** PROPOSAL. No vault edits have been made. This document is the
BEFORE/AFTER the user or a subsequent leg would apply to `vault/whale_kernel_triton.py`.

## Summary of patch

1. Add a new kernel `_attn_bwd_fused_no_o_kernel` (clone of
   `_attn_bwd_fused_kernel` at L810) that takes `DELTA [B, H, T]` fp32
   as an argument instead of `O [B, T, H, D]`, and loads a row vector
   of `delta` per M-iter instead of an `o_tile`.
2. Add a dispatch branch for `WHALE_BWD_VARIANT=fused_bwd_no_o` in
   `whale_attn_bwd` (wrapper around L1388). The branch launches
   `_attn_bwd_preprocess_kernel` (already exists, L421) to build
   `delta`, then launches the new fused kernel, then the existing
   `_attn_bwd_dq_cast_kernel` (L1015) to cast dQ fp32 → bf16.
3. Autotune config list `_bwd_fused_no_o_configs` cloned from
   `_bwd_fused_configs` (no config changes; only the kernel body
   differs).

The current fused kernels (`_attn_bwd_fused_kernel`,
`_attn_bwd_fused_tma_dq_kernel`) are left untouched so existing
`fused_bwd` / `fused_bwd_tma_dq` variants keep working for A/B.

## BEFORE — `_attn_bwd_fused_kernel` (vault/whale_kernel_triton.py L810-905)

Critical excerpt of the inner per-M-block loop (L860-889), where the
O-reload + delta recompute lives:

```python
for hg in range(group):
    h = kv_h * group + hg
    for m_block in range(m_start_block, m_end_block):
        start_m = m_block * BLOCK_M
        offs_m_cur = start_m + offs_m
        row_mask = offs_m_cur < T_MAX
        q_mask = row_mask[:, None] & (offs_d[None, :] < D)

        q_ptrs  = Q  + b*stride_qb  + h*stride_qh  + offs_m_cur[:,None]*stride_qt  + offs_d[None,:]*stride_qd
        do_ptrs = DO + b*stride_dob + h*stride_doh + offs_m_cur[:,None]*stride_dot + offs_d[None,:]*stride_dod
        o_ptrs  = O  + b*stride_ob  + h*stride_oh  + offs_m_cur[:,None]*stride_ot  + offs_d[None,:]*stride_od
        q  = tl.load(q_ptrs,  mask=q_mask, other=0.0)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)
        o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)                  # <-- KILL
        delta  = tl.sum(o_tile.to(tl.float32) * do.to(tl.float32), axis=1) # <-- KILL

        lse_ptrs = LSE + b*stride_lb + h*stride_lh + offs_m_cur*stride_lt
        lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
        p = tl.exp2(s - lse[:,None] * LOG2E)
        ...
        ds = p * (dp - delta[:,None])
```

Signature (L810-830):

```python
def _attn_bwd_fused_kernel(
    Q, K, V, O, DO, DK, DV, DQ_F32, LSE,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,           # <-- drop O strides
    stride_dob, stride_dot, stride_doh, stride_dod,
    stride_dkb, stride_dkt, stride_dkh, stride_dkd,
    stride_dvb, stride_dvt, stride_dvh, stride_dvd,
    stride_dqfb, stride_dqft, stride_dqfh, stride_dqfd,
    stride_lb, stride_lh, stride_lt,
    T_MAX, NUM_HEADS, NUM_KV_HEADS, SCALE,
    BLOCK_M, BLOCK_N, BLOCK_D, D, IS_CAUSAL,
):
```

## AFTER — `_attn_bwd_fused_no_o_kernel` (NEW, add alongside existing kernel)

Signature change: `O` replaced by `DELTA`; O strides replaced by
DELTA strides `(stride_deb, stride_deh, stride_det)`:

```python
@triton.autotune(
    configs=_bwd_fused_no_o_configs(),   # new list, cloned from _bwd_fused_configs
    key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"],
    reset_to_zero=["DQ_F32"],
)
@triton.jit
def _attn_bwd_fused_no_o_kernel(
    Q, K, V, DELTA, DO, DK, DV, DQ_F32, LSE,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_deb, stride_deh, stride_det,                   # <-- was stride_o*
    stride_dob, stride_dot, stride_doh, stride_dod,
    stride_dkb, stride_dkt, stride_dkh, stride_dkd,
    stride_dvb, stride_dvt, stride_dvh, stride_dvd,
    stride_dqfb, stride_dqft, stride_dqfh, stride_dqfd,
    stride_lb, stride_lh, stride_lt,
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
    bkv = tl.program_id(1)
    b = bkv // NUM_KV_HEADS
    kv_h = bkv % NUM_KV_HEADS
    group = NUM_HEADS // NUM_KV_HEADS

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    k_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
    k_ptrs = K + b*stride_kb + kv_h*stride_kh + offs_n[:,None]*stride_kt + offs_d[None,:]*stride_kd
    v_ptrs = V + b*stride_vb + kv_h*stride_vh + offs_n[:,None]*stride_vt + offs_d[None,:]*stride_vd
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
    v = tl.load(v_ptrs, mask=k_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    qk_scale_log2 = SCALE * LOG2E

    if IS_CAUSAL:
        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
    else:
        m_start_block = 0
    m_end_block = tl.cdiv(T_MAX, BLOCK_M)

    for hg in range(group):
        h = kv_h * group + hg
        for m_block in range(m_start_block, m_end_block):
            start_m = m_block * BLOCK_M
            offs_m_cur = start_m + offs_m
            row_mask = offs_m_cur < T_MAX
            q_mask = row_mask[:, None] & (offs_d[None, :] < D)

            q_ptrs  = Q  + b*stride_qb  + h*stride_qh  + offs_m_cur[:,None]*stride_qt  + offs_d[None,:]*stride_qd
            do_ptrs = DO + b*stride_dob + h*stride_doh + offs_m_cur[:,None]*stride_dot + offs_d[None,:]*stride_dod
            q  = tl.load(q_ptrs,  mask=q_mask, other=0.0)
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)

            # NEW: load precomputed delta row vector (BLOCK_M floats)
            # instead of O tile (BLOCK_M * D bf16).
            delta_ptrs = DELTA + b*stride_deb + h*stride_deh + offs_m_cur*stride_det
            delta = tl.load(delta_ptrs, mask=row_mask, other=0.0)

            lse_ptrs = LSE + b*stride_lb + h*stride_lh + offs_m_cur*stride_lt
            lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)

            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
            p = tl.exp2(s - lse[:,None] * LOG2E)

            if IS_CAUSAL:
                p_mask = row_mask[:,None] & (offs_n[None,:] < T_MAX) & (offs_m_cur[:,None] >= offs_n[None,:])
            else:
                p_mask = row_mask[:,None] & (offs_n[None,:] < T_MAX)
            p = tl.where(p_mask, p, 0.0)

            dv = tl.dot(tl.trans(p).to(q.dtype), do, acc=dv, out_dtype=tl.float32)
            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            ds = p * (dp - delta[:,None])
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk, out_dtype=tl.float32)

            dq_local = tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32) * SCALE
            dq_ptrs = (DQ_F32 + b*stride_dqfb + h*stride_dqfh
                       + offs_m_cur[:,None]*stride_dqft + offs_d[None,:]*stride_dqfd)
            tl.atomic_add(dq_ptrs, dq_local, mask=q_mask)

    dk = dk * SCALE
    dk_ptrs = DK + b*stride_dkb + kv_h*stride_dkh + offs_n[:,None]*stride_dkt + offs_d[None,:]*stride_dkd
    dv_ptrs = DV + b*stride_dvb + kv_h*stride_dvh + offs_n[:,None]*stride_vt + offs_d[None,:]*stride_vd
    store_mask = (offs_n[:,None] < T_MAX) & (offs_d[None,:] < D)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=store_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=store_mask)
```

Diff against the existing `_attn_bwd_fused_kernel`:

- Remove: `O` arg, `stride_ob, stride_ot, stride_oh, stride_od` args,
  `o_ptrs = ...`, `o_tile = tl.load(o_ptrs, ...)`,
  `delta = tl.sum(o_tile.to(fp32) * do.to(fp32), axis=1)`.
- Add: `DELTA` arg, `stride_deb, stride_deh, stride_det` args,
  `delta_ptrs = DELTA + b*stride_deb + h*stride_deh + offs_m_cur*stride_det`,
  `delta = tl.load(delta_ptrs, mask=row_mask, other=0.0)`.
- Everything else (LSE load, s/p recompute, dv/dp/ds/dk math, dq
  atomic_add) is unchanged byte-for-byte.

## Dispatch change in `whale_attn_bwd` (vault/whale_kernel_triton.py L1388-1457)

BEFORE (L1400-1414) — current branch set:

```python
elif variant == "fused_bwd":
    use_fused_delta = False
    use_tma_dkdv = False
    use_fused_bwd = True
    use_fused_bwd_tma_dq = False
elif variant == "fused_bwd_tma_dq":
    use_fused_delta = False
    use_tma_dkdv = False
    use_fused_bwd = True
    use_fused_bwd_tma_dq = True
```

AFTER — add `fused_bwd_no_o`:

```python
elif variant == "fused_bwd":
    use_fused_delta = False
    use_tma_dkdv = False
    use_fused_bwd = True
    use_fused_bwd_tma_dq = False
    use_fused_bwd_no_o = False
elif variant == "fused_bwd_tma_dq":
    use_fused_delta = False
    use_tma_dkdv = False
    use_fused_bwd = True
    use_fused_bwd_tma_dq = True
    use_fused_bwd_no_o = False
elif variant == "fused_bwd_no_o":
    use_fused_delta = False
    use_tma_dkdv = False
    use_fused_bwd = True
    use_fused_bwd_tma_dq = False
    use_fused_bwd_no_o = True
```

And inside `if use_fused_bwd:` (L1425), add the no-O branch before the
existing two. The branch:

1. Allocates `delta = torch.empty(B, H, T, dtype=fp32, device=cuda)`.
2. Launches `_attn_bwd_preprocess_kernel` on `(o, do_c, delta)` —
   exact same call signature as L1464-1472.
3. Allocates `dq_f32` as today (L1431).
4. Launches `_attn_bwd_fused_no_o_kernel[grid_kv]` with `(q, k, v,
   delta, do_c, dk, dv, dq_f32, lse)` and `delta.stride(0/1/2)`
   instead of `o.stride(0/1/2/3)`.
5. Launches `_attn_bwd_dq_cast_kernel` as today (L1449-1455).

## HBM bytes saved (per bwd call, headline shape)

Let `B=4, T=2048, H=8, KV=4, D=64, group=H/KV=2`. Assume BLOCK_M=128,
BLOCK_N=64 (existing autotune winners).

- Program grid: `(T/BLOCK_N=32, B*KV=16) = 512` programs.
- Per program, inner loop executes `group * (T/BLOCK_M) = 2 * 16 = 32`
  iterations (non-causal). Causal halves that on average: ~16.
- Per iter, O reload = `BLOCK_M * D * 2 = 128 * 64 * 2 = 16384 bytes`.
- Per program per call, O reload = `16 * 16384 = 262144 B = 256 KB`.
- Total O reload removed = `512 * 256 KB = 128 MB` (causal avg).
- Total O reload removed (non-causal worst case) = **256 MB**.

Added:

- Preprocess reads O, dO (each `B*H*T*D*2 = 8 MB`) and writes delta
  (`B*H*T*4 = 256 KB`).
- Fused kernel reads delta (`B*H*T*4 = 256 KB` — each row read 16
  times across programs worst case, total ≤ 4 MB).

Net HBM saved at headline (causal): ~128 MB – 20 MB ≈ **108 MB per
bwd call**. At H100's ~3 TB/s effective HBM bandwidth that bounds
savings at ~36 µs by BW alone; the real win should also include
register-pressure relief (one fewer 128×64 bf16 tile live in the
inner loop).

## Correctness argument

Math is algebraically identical. `_attn_bwd_preprocess_kernel`
computes exactly `delta[b,h,i] = sum_d o[b,i,h,d] * do[b,i,h,d]` in
fp32, which is the same value the existing fused kernel computes via
`tl.sum(o_tile.to(fp32) * do.to(fp32), axis=1)`. The only numerical
difference is reduction order across the D axis, which bf16→fp32
accumulation already tolerates (the 2-kernel `baseline` variant uses
exactly this pattern and passes the same numerics test). No change
to dS, dK, dV, dQ math.
