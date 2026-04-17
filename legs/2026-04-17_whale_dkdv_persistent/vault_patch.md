# Vault patch — `_attn_bwd_dkdv_persistent_kernel`

**DO NOT AUTO-APPLY.** This leg authors the patch; user approves before
the vault edit lands. `vault/` edits are gated by CLAUDE.md.

Target file: `vault/whale_kernel_triton.py`.

There are two parts:

1. **Add a new `@triton.jit` kernel** next to `_attn_bwd_dkdv_kernel`
   (insertion after line 573, before the inline_delta kernel banner at
   line 575). The new kernel body is essentially the existing kernel
   wrapped in an outer persistent loop; the inner body is verbatim.
2. **Switch the launcher** at line 1546-1560 to branch on
   `WHALE_BWD_KV_PERSISTENT=1`.

## Part 1 — New kernel (insert after line 573)

### BEFORE (`vault/whale_kernel_triton.py:452-573`, unchanged)

```python
@triton.autotune(configs=_bwd_kv_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_bwd_dkdv_kernel(
    Q, K, V, DO, DK, DV, LSE, DELTA,
    stride_qb, stride_qt, stride_qh, stride_qd,
    ...
    IS_CAUSAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    bkv = tl.program_id(1)
    b = bkv // NUM_KV_HEADS
    kv_h = bkv % NUM_KV_HEADS
    group = NUM_HEADS // NUM_KV_HEADS
    # ... (rest of body, unchanged — includes the causal early-exit
    #      m_start_block = (pid_n * BLOCK_N) // BLOCK_M at line 497)
```

### AFTER — add this kernel immediately after line 573, BEFORE the
`# Backward dK / dV kernel with inline Δ` banner at line 575

```python
# ---------------------------------------------------------------------------
# Persistent variant of the dK / dV kernel (H6).
# Grid is (NUM_SMS,). Each program iterates over assigned
# (b, kv_head, n_block) tiles in N-major order via
#   pid += tl.num_programs(0)
# Keeps K/V resident in L2 across adjacent N-blocks of the same (b, kv_h)
# on the same SM, and load-balances the causal early-exit imbalance.
# Identical math to _attn_bwd_dkdv_kernel — no atomic_add is introduced.
# ---------------------------------------------------------------------------


@triton.autotune(configs=_bwd_kv_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_bwd_dkdv_persistent_kernel(
    Q, K, V, DO, DK, DV, LSE, DELTA,
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
    B: tl.constexpr,
):
    # Total number of (b, kv_head, n_block) tiles.
    n_blocks = tl.cdiv(T_MAX, BLOCK_N)
    num_tiles = B * NUM_KV_HEADS * n_blocks

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    group = NUM_HEADS // NUM_KV_HEADS
    qk_scale_log2 = SCALE * LOG2E

    while pid < num_tiles:
        # Decompose linear tile id into (pid_n, bkv) so adjacent pids
        # walk adjacent N-blocks of the same (b, kv_h) — N-major order.
        pid_n = pid % n_blocks
        bkv   = pid // n_blocks
        b     = bkv // NUM_KV_HEADS
        kv_h  = bkv %  NUM_KV_HEADS

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
        k_ptrs = K + b * stride_kb + kv_h * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + kv_h * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        if IS_CAUSAL:
            m_start_block  = (pid_n * BLOCK_N) // BLOCK_M
            m_masking_max  = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)
        else:
            m_start_block  = 0
            m_masking_max  = 0
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

                    q_ptrs  = Q  + b * stride_qb  + h * stride_qh  + offs_m_cur[:, None] * stride_qt  + offs_d[None, :] * stride_qd
                    do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
                    q  = tl.load(q_ptrs,  mask=q_mask, other=0.0)
                    do = tl.load(do_ptrs, mask=q_mask, other=0.0)

                    lse_ptrs   = LSE   + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
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

                q_ptrs  = Q  + b * stride_qb  + h * stride_qh  + offs_m_cur[:, None] * stride_qt  + offs_d[None, :] * stride_qd
                do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
                q  = tl.load(q_ptrs,  mask=q_mask, other=0.0)
                do = tl.load(do_ptrs, mask=q_mask, other=0.0)

                lse_ptrs   = LSE   + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
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

        pid += n_progs
```

Notes on the kernel body vs `_attn_bwd_dkdv_kernel`:

- Math and masking are **identical**. Every `tl.load`, `tl.dot`,
  `tl.where`, `tl.store`, and the causal early-exit
  `m_start_block = (pid_n * BLOCK_N) // BLOCK_M` are verbatim.
- Only difference: outer `while pid < num_tiles: ... pid += n_progs`
  wraps the per-tile work, and `pid_n`/`bkv` are derived from `pid`
  instead of `program_id(0)` / `program_id(1)`.
- `B` must be passed as a `tl.constexpr` because the total-tile count
  `B * NUM_KV_HEADS * cdiv(T_MAX, BLOCK_N)` must be a constexpr at
  compile time for the `while` bound to lower cleanly. The existing
  non-persistent kernel does **not** take `B` as a constexpr; the
  launcher change below includes it.
- No `atomic_add`. dK/dV go to unique `(b, kv_h, n_block)` slots as
  before.

## Part 2 — Launcher change

### BEFORE (`vault/whale_kernel_triton.py:1545-1560`)

```python
    else:
        grid_kv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * KV)
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

### AFTER

```python
    else:
        use_persistent = os.environ.get("WHALE_BWD_KV_PERSISTENT", "0") == "1"
        if use_persistent:
            num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
            grid_kv = (num_sms,)
            _attn_bwd_dkdv_persistent_kernel[grid_kv](
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
                B=B,
            )
        else:
            grid_kv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * KV)
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

## Rollout

1. User approves.
2. Apply Part 1 (insert new kernel) and Part 2 (launcher branch).
3. `bash legs/2026-04-17_whale_dkdv_persistent/run.sh` for benchmarks.
4. If negative, revert both changes (the env gate keeps the default path
   unchanged, so leaving the kernel in place is harmless, but a clean
   revert is preferred to keep the vault minimal).

## Risk notes

- **Autotune + persistent**: `_bwd_kv_configs()` emits multiple
  (BLOCK_M, BLOCK_N, num_warps, num_stages) configs. Persistent grid
  size (NUM_SMS) is fixed and independent of BLOCK_N, so autotune should
  work without `reset_to_zero` (no accumulator). The triton_gotchas
  memory item about `reset_to_zero=[name]` applies only to kernels that
  write via `atomic_add`; this kernel uses plain `tl.store`.
- **maxnreg**: `_bwd_kv_configs()` (used by the non-inline `dkdv` kernel
  path, not `_bwd_kv_inline_configs`) does **not** set `maxnreg=224`;
  the `maxnreg=224` atomic_add gotcha does not apply regardless.
- **num_stages**: On persistent kernels, high `num_stages` can spill
  registers when the loop body is replayed many times. If autotune
  picks a spilling config, the `_env_force("BWD_KV")` lever lets us pin
  a single config for the persistent run.
