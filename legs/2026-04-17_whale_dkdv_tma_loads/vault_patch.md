# vault_patch.md — DO NOT auto-apply

This file describes the patch to `vault/whale_kernel_triton.py`. The patch adds
a NEW kernel `_attn_bwd_dkdv_tma_loads_kernel` next to the existing
`_attn_bwd_dkdv_kernel` (L455-L572). It also adds a `use_tma_kv_loads`
dispatch branch in `custom_whale_attn_bwd`. The original non-TMA kernel is
LEFT UNCHANGED so the `WHALE_BWD_KV_TMA_LOADS=0` fallback is byte-for-byte
what's in-tree today.

Vault is frozen by CLAUDE.md rules — the user applies this patch, not an
agent. Verify against `legs/2026-04-16_whale_fast_kernels/bench_numerics.py`
before any training run consumes the new kernel.

Target file: `vault/whale_kernel_triton.py`
Anchor: right after `_attn_bwd_dkdv_kernel` (current L572), before
`# Backward dK / dV kernel with inline Delta` header (L575).

## BEFORE — `_attn_bwd_dkdv_kernel` (L455-L572, excerpted around load sites)

```python
@triton.autotune(configs=_bwd_kv_inline_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_bwd_dkdv_kernel(
    Q, K, V, DO, DK, DV, LSE, DELTA,
    ...
):
    pid_n = tl.program_id(0)
    bkv = tl.program_id(1)
    b = bkv // NUM_KV_HEADS
    kv_h = bkv % NUM_KV_HEADS
    group = NUM_HEADS // NUM_KV_HEADS

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # ---- plain masked loads for K, V ----
    k_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
    k_ptrs = K + b * stride_kb + kv_h * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
    v_ptrs = V + b * stride_vb + kv_h * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
    v = tl.load(v_ptrs, mask=k_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    qk_scale_log2 = SCALE * LOG2E

    if IS_CAUSAL:
        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
        m_masking_max = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)
    else:
        m_start_block = 0
        m_masking_max = 0
    m_end_block = tl.cdiv(T_MAX, BLOCK_M)

    for hg in range(group):
        h = kv_h * group + hg

        if IS_CAUSAL:
            m_mask_end = tl.minimum(m_masking_max, m_end_block)
            for m_block in range(m_start_block, m_mask_end):
                start_m = m_block * BLOCK_M
                offs_m_cur = start_m + offs_m
                row_mask = offs_m_cur < T_MAX
                q_mask = row_mask[:, None] & (offs_d[None, :] < D)

                # ---- plain masked loads for Q, DO ----
                q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
                do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
                q = tl.load(q_ptrs, mask=q_mask, other=0.0)
                do = tl.load(do_ptrs, mask=q_mask, other=0.0)

                lse_ptrs  = LSE   + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
                delta_ptrs = DELTA + b * stride_db + h * stride_dh + offs_m_cur * stride_dt
                lse   = tl.load(lse_ptrs,   mask=row_mask, other=0.0)
                delta = tl.load(delta_ptrs, mask=row_mask, other=0.0)

                s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
                p = tl.exp2(s - lse[:, None] * LOG2E)

                p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX) & (offs_m_cur[:, None] >= offs_n[None, :])
                p = tl.where(p_mask, p, 0.0)

                dv = tl.dot(tl.trans(p).to(q.dtype), do, acc=dv, out_dtype=tl.float32)
                dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
                ds = p * (dp - delta[:, None])
                dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk, out_dtype=tl.float32)
            unmasked_start = m_mask_end
        else:
            unmasked_start = m_start_block

        for m_block in range(unmasked_start, m_end_block):
            start_m = m_block * BLOCK_M
            offs_m_cur = start_m + offs_m
            row_mask = offs_m_cur < T_MAX
            q_mask = row_mask[:, None] & (offs_d[None, :] < D)

            # ---- plain masked loads for Q, DO ----
            q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
            do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)

            lse_ptrs  = LSE   + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
            delta_ptrs = DELTA + b * stride_db + h * stride_dh + offs_m_cur * stride_dt
            lse   = tl.load(lse_ptrs,   mask=row_mask, other=0.0)
            delta = tl.load(delta_ptrs, mask=row_mask, other=0.0)

            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
            p = tl.exp2(s - lse[:, None] * LOG2E)

            p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX)
            p = tl.where(p_mask, p, 0.0)

            dv = tl.dot(tl.trans(p).to(q.dtype), do, acc=dv, out_dtype=tl.float32)
            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            ds = p * (dp - delta[:, None])
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk, out_dtype=tl.float32)

    dk = dk * SCALE
    dk_ptrs = DK + b * stride_dkb + kv_h * stride_dkh + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + b * stride_dvb + kv_h * stride_dvh + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
    store_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=store_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=store_mask)
```

## AFTER — ADD a new sibling kernel (leave the kernel above untouched)

Insert the following block immediately after the existing
`_attn_bwd_dkdv_kernel` body, before the `# Backward dK / dV kernel with
inline Delta` comment header.

```python
# ---------------------------------------------------------------------------
# Backward dK / dV kernel with early-exit + TMA loads for K,V,Q,DO (DELTA pre-
# computed). Opt-in via WHALE_BWD_KV_TMA_LOADS=1 through custom_whale_attn_bwd.
# Math is identical to _attn_bwd_dkdv_kernel; only the bf16 bulk loads change.
# Requires BLOCK_D == D (TMA last-dim must equal descriptor last-dim).
# Stores for DK/DV stay on plain tl.store to keep the diff narrow — a TMA-
# store follow-up can be a separate child leg once this one shows a win.
# ---------------------------------------------------------------------------


@triton.autotune(configs=_bwd_kv_inline_tma_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_bwd_dkdv_tma_loads_kernel(
    Q, K, V, DO, DK, DV, LSE, DELTA,
    stride_qb, stride_qt, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_dob, stride_dot, stride_doh,
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

    # K, V descriptors built once per program; reused across all Q heads / M blocks.
    K_desc = tl.make_tensor_descriptor(
        K + b * stride_kb + kv_h * stride_kh,
        shape=[T_MAX, D], strides=[stride_kt, 1], block_shape=[BLOCK_N, D],
    )
    V_desc = tl.make_tensor_descriptor(
        V + b * stride_vb + kv_h * stride_vh,
        shape=[T_MAX, D], strides=[stride_vt, 1], block_shape=[BLOCK_N, D],
    )
    k = K_desc.load([pid_n * BLOCK_N, 0])
    v = V_desc.load([pid_n * BLOCK_N, 0])

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    qk_scale_log2 = SCALE * LOG2E

    if IS_CAUSAL:
        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
        m_masking_max = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)
    else:
        m_start_block = 0
        m_masking_max = 0
    m_end_block = tl.cdiv(T_MAX, BLOCK_M)

    for hg in range(group):
        h = kv_h * group + hg

        # Q / DO descriptors built once per Q-head, reused across ~m_end_block - m_start_block blocks.
        Q_desc = tl.make_tensor_descriptor(
            Q + b * stride_qb + h * stride_qh,
            shape=[T_MAX, D], strides=[stride_qt, 1], block_shape=[BLOCK_M, D],
        )
        DO_desc = tl.make_tensor_descriptor(
            DO + b * stride_dob + h * stride_doh,
            shape=[T_MAX, D], strides=[stride_dot, 1], block_shape=[BLOCK_M, D],
        )

        # --- masking phase (kept exactly as early-exit kernel) ---
        if IS_CAUSAL:
            m_mask_end = tl.minimum(m_masking_max, m_end_block)
            for m_block in range(m_start_block, m_mask_end):
                start_m = m_block * BLOCK_M
                offs_m_cur = start_m + offs_m
                row_mask = offs_m_cur < T_MAX

                q  = Q_desc.load([start_m, 0])
                do = DO_desc.load([start_m, 0])

                lse_ptrs  = LSE   + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
                delta_ptrs = DELTA + b * stride_db + h * stride_dh + offs_m_cur * stride_dt
                lse   = tl.load(lse_ptrs,   mask=row_mask, other=0.0)
                delta = tl.load(delta_ptrs, mask=row_mask, other=0.0)

                s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
                p = tl.exp2(s - lse[:, None] * LOG2E)

                p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX) & (offs_m_cur[:, None] >= offs_n[None, :])
                p = tl.where(p_mask, p, 0.0)

                dv = tl.dot(tl.trans(p).to(q.dtype), do, acc=dv, out_dtype=tl.float32)
                dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
                ds = p * (dp - delta[:, None])
                dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk, out_dtype=tl.float32)
            unmasked_start = m_mask_end
        else:
            unmasked_start = m_start_block

        # --- unmasked phase ---
        for m_block in range(unmasked_start, m_end_block):
            start_m = m_block * BLOCK_M
            offs_m_cur = start_m + offs_m
            row_mask = offs_m_cur < T_MAX

            q  = Q_desc.load([start_m, 0])
            do = DO_desc.load([start_m, 0])

            lse_ptrs  = LSE   + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
            delta_ptrs = DELTA + b * stride_db + h * stride_dh + offs_m_cur * stride_dt
            lse   = tl.load(lse_ptrs,   mask=row_mask, other=0.0)
            delta = tl.load(delta_ptrs, mask=row_mask, other=0.0)

            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
            p = tl.exp2(s - lse[:, None] * LOG2E)

            p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX)
            p = tl.where(p_mask, p, 0.0)

            dv = tl.dot(tl.trans(p).to(q.dtype), do, acc=dv, out_dtype=tl.float32)
            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            ds = p * (dp - delta[:, None])
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk, out_dtype=tl.float32)

    dk = dk * SCALE

    # Stores stay on plain masked tl.store — tile shape is [BLOCK_N, D].
    offs_d = tl.arange(0, D)
    dk_ptrs = DK + b * stride_dkb + kv_h * stride_dkh + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + b * stride_dvb + kv_h * stride_dvh + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
    store_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=store_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=store_mask)
```

## Python-level dispatch — `custom_whale_attn_bwd`

In the non-fused, non-split-h branch at current L1545-L1560
(`grid_kv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * KV); _attn_bwd_dkdv_kernel[grid_kv](...)`),
wrap the call site like so:

```python
    else:
        grid_kv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * KV)
        use_tma_kv_loads = os.environ.get("WHALE_BWD_KV_TMA_LOADS", "0") == "1"
        if use_tma_kv_loads:
            # Precondition: D in {64, 128}, bf16, contiguous [B, T, H, D].
            # 16-byte alignment of base + b*stride_b + kv_h*stride_h follows
            # from a contiguous layout with D * element_size % 16 == 0.
            assert D in (64, 128), "TMA loads kernel requires D in {64,128}"
            _attn_bwd_dkdv_tma_loads_kernel[grid_kv](
                q, k, v, do_c, dk, dv, lse, delta,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                do_c.stride(0), do_c.stride(1), do_c.stride(2),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
                D=D,
                IS_CAUSAL=causal,
            )
        else:
            _attn_bwd_dkdv_kernel[grid_kv](
                q, k, v, do_c, dk, dv, lse, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
                BLOCK_D=block_d, D=D,
                IS_CAUSAL=causal,
            )
```

Note: the TMA kernel drops the BLOCK_D kwarg because the descriptor forces
`BLOCK_D == D`; the autotune table `_bwd_kv_inline_tma_configs` already
reflects that (no BLOCK_D key in the configs).

## early-exit constexpr question

Q: Does `tl.minimum(m_masking_max, m_end_block)` fold at compile time in the
new TMA kernel?

A: No — and that matches the non-TMA early-exit kernel. `m_masking_max =
tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)` depends on `pid_n`, which is a
runtime program ID, so `m_masking_max` is runtime. `m_end_block =
tl.cdiv(T_MAX, BLOCK_M)` is constexpr (T_MAX and BLOCK_M are both
constexpr), but the `tl.minimum` of a runtime scalar with a constexpr
scalar stays runtime. TMA does not change this: the descriptor is
constructed from constexpr shape/strides, and the tile *offset*
`[pid_n * BLOCK_N, 0]` is runtime — identical semantics to the non-TMA
load. Conclusion: no new compile-time folding opportunity, no regression
vs the non-TMA early-exit kernel.

If we wanted the masking-phase bound to be fully constexpr we would need
`m_masking_max` to be BLOCK_M-aligned at compile time, which requires
`BLOCK_N % BLOCK_M == 0` plus a grid specialization on pid_n parity. Not
in scope for this leg.

## implementer checklist
1. Insert the new kernel into `vault/whale_kernel_triton.py` after
   `_attn_bwd_dkdv_kernel` (current L572). Do NOT remove or modify
   `_attn_bwd_dkdv_kernel`.
2. Add the `use_tma_kv_loads` branch in `custom_whale_attn_bwd` at the
   non-fused, non-split-h call site.
3. Run `bash legs/2026-04-17_whale_dkdv_tma_loads/run.sh before` first
   (env var unset, routes to original kernel — sanity check numerics
   unchanged).
4. Run `bash legs/2026-04-17_whale_dkdv_tma_loads/run.sh after` second
   (env var set to 1 inside `tracked_env.sh`, routes to TMA kernel).
5. Flush the triton autotune cache once before the first `after` run
   (`WHALE_FLUSH_TRITON_CACHE=1`, already default in tracked_env.sh).
6. Compare the before/after JSON outputs in `evidence/` per-shape.

## numerics tolerance to enforce
Reuse the tolerances in `legs/2026-04-16_whale_fast_kernels/bench_numerics.py`:
- fwd output max abs delta: <= 5e-3
- dQ / dK / dV max abs delta: <= 1e-2
