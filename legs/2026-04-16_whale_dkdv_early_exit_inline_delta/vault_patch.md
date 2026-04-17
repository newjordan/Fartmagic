# vault_patch.md — DO NOT auto-apply

This file describes the patch to `vault/whale_kernel_triton.py` for the
inline-delta dkdv variant. It is for review only. The actual edit must be
made by the user (vault is frozen by CLAUDE.md rules) and verified against
`bench_numerics.py` before any training run consumes the new kernel.

Sequencing: do NOT apply until the sibling leg
`legs/2026-04-16_whale_dkdv_early_exit/` shows a positive bench delta on its
base-kernel patch. If the base patch is a no-op or regression, this patch
also should not land.

Target file: `vault/whale_kernel_triton.py`
Target function: `_attn_bwd_dkdv_inline_delta_kernel` (declared L559)

## BEFORE (L600-L637, exact)
```python
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

            q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
            do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
            o_ptrs = O + b * stride_ob + h * stride_oh + offs_m_cur[:, None] * stride_ot + offs_d[None, :] * stride_od
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)
            o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)
            delta = tl.sum(o_tile.to(tl.float32) * do.to(tl.float32), axis=1)

            lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
            lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)

            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
            p = tl.exp2(s - lse[:, None] * LOG2E)

            if IS_CAUSAL:
                p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX) & (offs_m_cur[:, None] >= offs_n[None, :])
            else:
                p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX)
            p = tl.where(p_mask, p, 0.0)

            dv = tl.dot(tl.trans(p).to(q.dtype), do, acc=dv, out_dtype=tl.float32)
            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            ds = p * (dp - delta[:, None])
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk, out_dtype=tl.float32)
```

## AFTER — split into masking + unmasked phases
```python
    if IS_CAUSAL:
        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
        # Largest M-block (exclusive) where the (m_block, n_block) tile
        # straddles the causal diagonal. Beyond this, every entry of the
        # tile satisfies offs_m >= offs_n, so the causal mask is trivially
        # true. FA3 reference: hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp L1008.
        m_masking_max = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)
    else:
        m_start_block = 0
        m_masking_max = 0
    m_end_block = tl.cdiv(T_MAX, BLOCK_M)

    # NOTE: use plain `range(...)`, not `tl.range(...)`. Triton 3.6's
    # NVGPUWarpSpecialization pass crashes on this kernel; the existing
    # M-loop already uses bare `range` for the same reason.
    for hg in range(group):
        h = kv_h * group + hg

        # --- masking phase: m_start_block .. min(m_masking_max, m_end_block)
        if IS_CAUSAL:
            m_mask_end = tl.minimum(m_masking_max, m_end_block)
            for m_block in range(m_start_block, m_mask_end):
                start_m = m_block * BLOCK_M
                offs_m_cur = start_m + offs_m
                row_mask = offs_m_cur < T_MAX
                q_mask = row_mask[:, None] & (offs_d[None, :] < D)

                q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
                do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
                o_ptrs = O + b * stride_ob + h * stride_oh + offs_m_cur[:, None] * stride_ot + offs_d[None, :] * stride_od
                q = tl.load(q_ptrs, mask=q_mask, other=0.0)
                do = tl.load(do_ptrs, mask=q_mask, other=0.0)
                o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)
                delta = tl.sum(o_tile.to(tl.float32) * do.to(tl.float32), axis=1)

                lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
                lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)

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

        # --- unmasked phase: rest of the M-loop, no causal predicate / where
        for m_block in range(unmasked_start, m_end_block):
            start_m = m_block * BLOCK_M
            offs_m_cur = start_m + offs_m
            row_mask = offs_m_cur < T_MAX
            q_mask = row_mask[:, None] & (offs_d[None, :] < D)

            q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
            do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
            o_ptrs = O + b * stride_ob + h * stride_oh + offs_m_cur[:, None] * stride_ot + offs_d[None, :] * stride_od
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)
            o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)
            delta = tl.sum(o_tile.to(tl.float32) * do.to(tl.float32), axis=1)

            lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
            lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)

            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
            p = tl.exp2(s - lse[:, None] * LOG2E)

            # Sequence-bounds only. For T_MAX divisible by BLOCK_M & BLOCK_N
            # this whole `tl.where` collapses to identity at compile time.
            p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX)
            p = tl.where(p_mask, p, 0.0)

            dv = tl.dot(tl.trans(p).to(q.dtype), do, acc=dv, out_dtype=tl.float32)
            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            ds = p * (dp - delta[:, None])
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk, out_dtype=tl.float32)
```

## Exact Edit calls (do NOT apply yet)

### Edit 1 — split the M-start computation to also produce `m_masking_max`

`old_string` (matches L600-L604 in `vault/whale_kernel_triton.py`):
```python
    if IS_CAUSAL:
        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
    else:
        m_start_block = 0
    m_end_block = tl.cdiv(T_MAX, BLOCK_M)
```

`new_string`:
```python
    if IS_CAUSAL:
        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
        # Largest M-block (exclusive) where the (m_block, n_block) tile
        # straddles the causal diagonal. Beyond this, every entry of the
        # tile satisfies offs_m >= offs_n, so the causal mask is trivially
        # true. FA3 reference: hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp L1008.
        m_masking_max = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)
    else:
        m_start_block = 0
        m_masking_max = 0
    m_end_block = tl.cdiv(T_MAX, BLOCK_M)
```

### Edit 2 — split the M-loop body into masking and unmasked phases

`old_string` (matches L606-L637 in `vault/whale_kernel_triton.py`):
```python
    for hg in range(group):
        h = kv_h * group + hg
        for m_block in range(m_start_block, m_end_block):
            start_m = m_block * BLOCK_M
            offs_m_cur = start_m + offs_m
            row_mask = offs_m_cur < T_MAX
            q_mask = row_mask[:, None] & (offs_d[None, :] < D)

            q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
            do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
            o_ptrs = O + b * stride_ob + h * stride_oh + offs_m_cur[:, None] * stride_ot + offs_d[None, :] * stride_od
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)
            o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)
            delta = tl.sum(o_tile.to(tl.float32) * do.to(tl.float32), axis=1)

            lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
            lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)

            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
            p = tl.exp2(s - lse[:, None] * LOG2E)

            if IS_CAUSAL:
                p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX) & (offs_m_cur[:, None] >= offs_n[None, :])
            else:
                p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX)
            p = tl.where(p_mask, p, 0.0)

            dv = tl.dot(tl.trans(p).to(q.dtype), do, acc=dv, out_dtype=tl.float32)
            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            ds = p * (dp - delta[:, None])
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk, out_dtype=tl.float32)
```

`new_string`:
```python
    for hg in range(group):
        h = kv_h * group + hg

        # --- masking phase: m_start_block .. min(m_masking_max, m_end_block)
        if IS_CAUSAL:
            m_mask_end = tl.minimum(m_masking_max, m_end_block)
            for m_block in range(m_start_block, m_mask_end):
                start_m = m_block * BLOCK_M
                offs_m_cur = start_m + offs_m
                row_mask = offs_m_cur < T_MAX
                q_mask = row_mask[:, None] & (offs_d[None, :] < D)

                q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
                do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
                o_ptrs = O + b * stride_ob + h * stride_oh + offs_m_cur[:, None] * stride_ot + offs_d[None, :] * stride_od
                q = tl.load(q_ptrs, mask=q_mask, other=0.0)
                do = tl.load(do_ptrs, mask=q_mask, other=0.0)
                o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)
                delta = tl.sum(o_tile.to(tl.float32) * do.to(tl.float32), axis=1)

                lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
                lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)

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

        # --- unmasked phase: rest of the M-loop, no causal predicate / where
        for m_block in range(unmasked_start, m_end_block):
            start_m = m_block * BLOCK_M
            offs_m_cur = start_m + offs_m
            row_mask = offs_m_cur < T_MAX
            q_mask = row_mask[:, None] & (offs_d[None, :] < D)

            q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
            do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
            o_ptrs = O + b * stride_ob + h * stride_oh + offs_m_cur[:, None] * stride_ot + offs_d[None, :] * stride_od
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)
            o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)
            delta = tl.sum(o_tile.to(tl.float32) * do.to(tl.float32), axis=1)

            lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
            lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)

            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
            p = tl.exp2(s - lse[:, None] * LOG2E)

            # Sequence-bounds only. For T_MAX divisible by BLOCK_M & BLOCK_N
            # this whole `tl.where` collapses to identity at compile time.
            p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX)
            p = tl.where(p_mask, p, 0.0)

            dv = tl.dot(tl.trans(p).to(q.dtype), do, acc=dv, out_dtype=tl.float32)
            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
            ds = p * (dp - delta[:, None])
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk, out_dtype=tl.float32)
```

## Notes for the implementer
- The base-kernel patch (sibling leg) uses `tl.range(..., num_stages=NUM_STAGES_WS,
  warp_specialize=WARPSPEC)`. The inline-delta kernel does **not** thread
  `NUM_STAGES_WS` / `WARPSPEC` constexprs, and the original loop uses bare
  `range(...)`. We keep `range(...)` here because Triton 3.6's
  `NVGPUWarpSpecialization` pass crashes on this kernel (existing comment is
  implicit in the kernel using `range` rather than `tl.range`).
- `tl.minimum(m_masking_max, m_end_block)` mirrors the base patch idiom. If
  Triton 3.6 fails to fold the constexpr-vs-runtime `tl.minimum`, fall back
  to a Python-level `min(...)` if both bounds are constexpr at trace time
  (they are not in this kernel — `pid_n` is a runtime int — so `tl.minimum`
  is required).
- Autotune key is unchanged (`D, IS_CAUSAL, NUM_HEADS, NUM_KV_HEADS, T_MAX`).
  Flush the Triton cache once via `WHALE_FLUSH_TRITON_CACHE=1` before the
  first benched run to avoid stale picks for the new IR.
- Do NOT touch `_attn_bwd_dkdv_inline_delta_tma_kernel` (L658+) in this
  leg — it has a different layout (TMA tensor descriptors) and is a child
  leg.

## Numerics tolerance to enforce in `bench_numerics.py`
- max |Δ| (custom vs sdpa) for fwd output: <= 5e-3 in bf16
- max |Δ| for dQ, dK, dV: <= 1e-2 in bf16
- These are the existing tolerances in `legs/2026-04-16_whale_fast_kernels/bench_numerics.py`;
  this leg reuses that script verbatim through run.sh.
