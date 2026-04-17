# vault_patch.md — DO NOT auto-apply

This file describes the patch to `vault/whale_kernel_triton.py`. It is for
review only. The actual edit must be made by the user (vault is frozen by
CLAUDE.md rules) and verified against `bench_numerics.py` before any
training run consumes the new kernel.

Target file: `vault/whale_kernel_triton.py`
Target function: `_attn_bwd_dkdv_kernel` (declared L427)

## BEFORE (L470-L509, exact)
```python
    if IS_CAUSAL:
        # Only Q rows with index >= pid_n * BLOCK_N can attend to this KV block.
        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
    else:
        m_start_block = 0
    m_end_block = tl.cdiv(T_MAX, BLOCK_M)

    for hg in range(group):
        h = kv_h * group + hg
        for m_block in tl.range(m_start_block, m_end_block, 1,
                                num_stages=NUM_STAGES_WS,
                                warp_specialize=WARPSPEC):
            start_m = m_block * BLOCK_M
            offs_m_cur = start_m + offs_m
            row_mask = offs_m_cur < T_MAX
            q_mask = row_mask[:, None] & (offs_d[None, :] < D)

            q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
            do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)

            lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
            delta_ptrs = DELTA + b * stride_db + h * stride_dh + offs_m_cur * stride_dt
            lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)
            delta = tl.load(delta_ptrs, mask=row_mask, other=0.0)

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
        # Only Q rows with index >= pid_n * BLOCK_N can attend to this KV block.
        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
        # Largest M-block (exclusive) where the (m_block, n_block) tile
        # straddles the causal diagonal. Beyond this, every entry of the tile
        # satisfies offs_m >= offs_n, so the causal mask is trivially true.
        # FA3 reference: hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp L1008.
        m_masking_max = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)
    else:
        m_start_block = 0
        m_masking_max = 0
    m_end_block = tl.cdiv(T_MAX, BLOCK_M)

    # Single inlined body, parameterized by APPLY_CAUSAL_MASK so the Triton
    # compiler specializes each loop. We rely on `m_block < m_masking_max`
    # being a runtime int comparison; both call-sites pass a constexpr bool
    # for APPLY_CAUSAL_MASK so the inner branch is fully eliminated.
    for hg in range(group):
        h = kv_h * group + hg

        # --- masking phase: m_start_block .. min(m_masking_max, m_end_block)
        if IS_CAUSAL:
            m_mask_end = tl.minimum(m_masking_max, m_end_block)
            for m_block in tl.range(m_start_block, m_mask_end, 1,
                                    num_stages=NUM_STAGES_WS,
                                    warp_specialize=WARPSPEC):
                start_m = m_block * BLOCK_M
                offs_m_cur = start_m + offs_m
                row_mask = offs_m_cur < T_MAX
                q_mask = row_mask[:, None] & (offs_d[None, :] < D)

                q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
                do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
                q = tl.load(q_ptrs, mask=q_mask, other=0.0)
                do = tl.load(do_ptrs, mask=q_mask, other=0.0)

                lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
                delta_ptrs = DELTA + b * stride_db + h * stride_dh + offs_m_cur * stride_dt
                lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)
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

        # --- unmasked phase: rest of the M-loop, no causal predicate / where
        for m_block in tl.range(unmasked_start, m_end_block, 1,
                                num_stages=NUM_STAGES_WS,
                                warp_specialize=WARPSPEC):
            start_m = m_block * BLOCK_M
            offs_m_cur = start_m + offs_m
            row_mask = offs_m_cur < T_MAX
            q_mask = row_mask[:, None] & (offs_d[None, :] < D)

            q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
            do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)

            lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
            delta_ptrs = DELTA + b * stride_db + h * stride_dh + offs_m_cur * stride_dt
            lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)
            delta = tl.load(delta_ptrs, mask=row_mask, other=0.0)

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
- The same change should also be evaluated for `_attn_bwd_dkdv_inline_delta_kernel`
  (the fused-Δ variant introduced after L520). The mask predicate appears in
  the same shape there. **Do that as a follow-up patch in this same leg only
  after the base patch shows a positive delta.**
- Autotune key is unchanged (`D, IS_CAUSAL, NUM_HEADS, NUM_KV_HEADS, T_MAX`).
  No need to invalidate cached configs, but flush the autotune cache once
  before the first benched run to avoid stale picks for the new IR.
- If `tl.minimum(constexpr, runtime)` causes Triton 3.6 to fall off the
  fast path, replace with a Python-level constexpr comparison or reorder
  to keep both bounds runtime ints. (Triton 3.6 ceiling at 72us per
  fwd was already documented in commit `2e07aea`; do not let constexpr
  promotion regress that.)

## Numerics tolerance to enforce in `bench_numerics.py`
- max |Δ| (custom vs sdpa) for fwd output: <= 5e-3 in bf16
- max |Δ| for dQ, dK, dV: <= 1e-2 in bf16
- These are the existing tolerances in `legs/2026-04-16_whale_fast_kernels/bench_numerics.py`;
  this leg reuses that script verbatim through gate.sh.
