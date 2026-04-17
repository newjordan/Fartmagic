# Coordinator Edit Sequence — Lever B (TMA atomic_add dQ)

Verified against `vault/whale_kernel_triton.py` @ 1554 lines, 2026-04-16.
Lever A is REVERTED; early-exit refactor only touches `_attn_bwd_dkdv_kernel`
(L427) — outside Lever B's surgery zone.

Apply 3 Edit calls **in order** (each old_string is unique in the file):

---

## Edit 1 — insert `_bwd_fused_tma_dq_configs()` after `_bwd_fused_configs()`

`file_path`: `/home/frosty40/SOTA_FINAL/vault/whale_kernel_triton.py`

`old_string`:
```
    configs.append(triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},
                                 num_warps=4, num_stages=2, **extra))
    return configs


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
```

`new_string`:
```
    configs.append(triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},
                                 num_warps=4, num_stages=2, **extra))
    return configs


def _bwd_fused_tma_dq_configs():
    """Config list for _attn_bwd_fused_tma_dq_kernel. Same shape as
    _bwd_fused_configs but pins maxnreg<=192 for safety: scalar
    tl.atomic_add corrupted at maxnreg>=224 on Triton 3.6 cu130, and
    while TMA atomic_add lowers via cp.reduce.async.bulk.add (a
    different code path) we don't lift the cap until proven safe."""
    forced = _env_force("BWD_FUSED_TMA_DQ")
    if forced:
        return forced
    maxnreg_env = os.environ.get("WHALE_BWD_FUSED_MAXNREG", "").strip()
    extra = {}
    if maxnreg_env:
        mr = int(maxnreg_env)
        if mr > 0:
            if mr > 192:
                raise ValueError(
                    f"WHALE_BWD_FUSED_MAXNREG={mr} exceeds the 192 cap on the"
                    " TMA-dq fused bwd until safety is verified. Refusing."
                )
            extra["maxnreg"] = mr
    else:
        extra["maxnreg"] = 192
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128)]:
        for w in (4, 8):
            for s in (3, 4):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, **extra))
    configs.append(triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},
                                 num_warps=4, num_stages=2, **extra))
    return configs


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
```

---

## Edit 2 — insert `_attn_bwd_fused_tma_dq_kernel` between fused_kernel and dq_cast_kernel

`file_path`: `/home/frosty40/SOTA_FINAL/vault/whale_kernel_triton.py`

`old_string`:
```
    dk = dk * SCALE
    dk_ptrs = DK + b * stride_dkb + kv_h * stride_dkh + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + b * stride_dvb + kv_h * stride_dvh + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
    store_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=store_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=store_mask)


@triton.jit
def _attn_bwd_dq_cast_kernel(
```

`new_string`:
```
    dk = dk * SCALE
    dk_ptrs = DK + b * stride_dkb + kv_h * stride_dkh + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + b * stride_dvb + kv_h * stride_dvh + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
    store_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=store_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=store_mask)


@triton.autotune(
    configs=_bwd_fused_tma_dq_configs(),
    key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"],
    reset_to_zero=["DQ_F32"],
)
@triton.jit
def _attn_bwd_fused_tma_dq_kernel(
    Q, K, V, O, DO, DK, DV, DQ_F32, LSE,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
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
    tl.static_assert(BLOCK_D == D, "TMA-dq fused bwd requires BLOCK_D == D")
    pid_n = tl.program_id(0)
    bkv = tl.program_id(1)
    b = bkv // NUM_KV_HEADS
    kv_h = bkv % NUM_KV_HEADS
    group = NUM_HEADS // NUM_KV_HEADS

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

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
    else:
        m_start_block = 0
    m_end_block = tl.cdiv(T_MAX, BLOCK_M)

    for hg in range(group):
        h = kv_h * group + hg
        # Per-(b,h) DQ descriptor. Last dim must be contiguous (stride_dqfd==1
        # by torch.zeros B,T,H,D). Base = DQ_F32 + b*stride_dqfb + h*stride_dqfh
        # is 16-byte aligned: caching allocator gives >=512B alignment, and
        # h*D fp32 = h*D*4 bytes is a mult of 16 for D pow2 >= 4.
        DQ_desc = tl.make_tensor_descriptor(
            DQ_F32 + b * stride_dqfb + h * stride_dqfh,
            shape=[T_MAX, D],
            strides=[stride_dqft, 1],
            block_shape=[BLOCK_M, D],
        )
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

            # dq via TMA descriptor.atomic_add: lowers to
            # cp.reduce.async.bulk.add.f32 (FA3 hopper bwd's primitive).
            # No mask= kwarg on descriptor.atomic_add (core.py:1410); zero
            # OOB rows so tail tiles contribute exact zero.
            dq_local = tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32) * SCALE
            dq_local = tl.where(q_mask, dq_local, 0.0)
            DQ_desc.atomic_add([start_m, 0], dq_local)

    dk = dk * SCALE
    dk_ptrs = DK + b * stride_dkb + kv_h * stride_dkh + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + b * stride_dvb + kv_h * stride_dvh + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
    store_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=store_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=store_mask)


@triton.jit
def _attn_bwd_dq_cast_kernel(
```

---

## Edit 3 — extend dispatch site (variant + use_fused_bwd block)

`file_path`: `/home/frosty40/SOTA_FINAL/vault/whale_kernel_triton.py`

`old_string`:
```
    variant = os.environ.get("WHALE_BWD_VARIANT", "auto")
    fused_delta_t_max = int(os.environ.get("WHALE_FUSED_DELTA_T_MAX", "3072"))
    if variant == "auto":
        use_fused_delta = T <= fused_delta_t_max
        use_tma_dkdv = False
        use_fused_bwd = False
    elif variant == "fused_delta_tma":
        use_fused_delta = True
        use_tma_dkdv = True
        use_fused_bwd = False
    elif variant == "fused_bwd":
        use_fused_delta = False
        use_tma_dkdv = False
        use_fused_bwd = True
    else:
        use_fused_delta = variant == "fused_delta"
        use_tma_dkdv = False
        use_fused_bwd = False
    # TMA requires BLOCK_D == D, which holds iff D is a power of 2.
    if use_tma_dkdv and block_d != D:
        use_tma_dkdv = False
    use_split_h = os.environ.get("WHALE_BWD_SPLIT_H", "0") == "1"

    if use_tma_dkdv:
        _ensure_tma_allocator()

    if use_fused_bwd:
        dq_f32 = torch.zeros(B, T, H, D, device=q.device, dtype=torch.float32)
        grid_kv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * KV)
        _attn_bwd_fused_kernel[grid_kv](
            q, k, v, o, do_c, dk, dv, dq_f32, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            dq_f32.stride(0), dq_f32.stride(1), dq_f32.stride(2), dq_f32.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
            BLOCK_D=block_d, D=D, IS_CAUSAL=causal,
        )
```

`new_string`:
```
    variant = os.environ.get("WHALE_BWD_VARIANT", "auto")
    fused_delta_t_max = int(os.environ.get("WHALE_FUSED_DELTA_T_MAX", "3072"))
    if variant == "auto":
        use_fused_delta = T <= fused_delta_t_max
        use_tma_dkdv = False
        use_fused_bwd = False
        use_fused_bwd_tma_dq = False
    elif variant == "fused_delta_tma":
        use_fused_delta = True
        use_tma_dkdv = True
        use_fused_bwd = False
        use_fused_bwd_tma_dq = False
    elif variant == "fused_bwd":
        use_fused_delta = False
        use_tma_dkdv = False
        use_fused_bwd = True
        use_fused_bwd_tma_dq = False
    elif variant == "fused_bwd_tma_dq":
        use_fused_delta = False
        use_tma_dkdv = False
        use_fused_bwd = True            # share dq_f32 scratch + cast path
        use_fused_bwd_tma_dq = True
    else:
        use_fused_delta = variant == "fused_delta"
        use_tma_dkdv = False
        use_fused_bwd = False
        use_fused_bwd_tma_dq = False
    # TMA requires BLOCK_D == D, which holds iff D is a power of 2.
    if use_tma_dkdv and block_d != D:
        use_tma_dkdv = False
    # The TMA-dq variant *also* needs BLOCK_D == D; fall back to the
    # scalar fused_bwd kernel if D is not a power of 2.
    if use_fused_bwd_tma_dq and block_d != D:
        use_fused_bwd_tma_dq = False
    use_split_h = os.environ.get("WHALE_BWD_SPLIT_H", "0") == "1"

    if use_tma_dkdv:
        _ensure_tma_allocator()

    if use_fused_bwd:
        if use_fused_bwd_tma_dq:
            _ensure_tma_allocator()
            fused_kernel = _attn_bwd_fused_tma_dq_kernel
        else:
            fused_kernel = _attn_bwd_fused_kernel
        dq_f32 = torch.zeros(B, T, H, D, device=q.device, dtype=torch.float32)
        grid_kv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * KV)
        fused_kernel[grid_kv](
            q, k, v, o, do_c, dk, dv, dq_f32, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            dq_f32.stride(0), dq_f32.stride(1), dq_f32.stride(2), dq_f32.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
            BLOCK_D=block_d, D=D, IS_CAUSAL=causal,
        )
```
