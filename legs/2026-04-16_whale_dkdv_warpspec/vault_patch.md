# vault_patch.md — Lever A (warp_specialize on dkdv M-loop)

**STATUS: documentation only. Do NOT apply automatically. User will
review and apply manually to `vault/whale_kernel_triton.py`.**

Target file: `/home/frosty40/SOTA_FINAL/vault/whale_kernel_triton.py`
(current length: 1523 lines, verified via `wc -l`).

This patch makes three coordinated edits:

1. Add a new `tl.constexpr` parameter `WARPSPEC: tl.constexpr` to
   `_attn_bwd_dkdv_kernel` (signature at L427-446).
2. Replace the inner Python `for m_block in range(...)` M-loop
   (currently L477) with a `tl.range(...)` call that conditionally
   passes `warp_specialize=True` and `num_stages=NUM_STAGES_WS` when
   `WARPSPEC=True`. The body is unchanged.
3. At the dispatch site for the baseline path
   (`_attn_bwd_dkdv_kernel[grid_kv](...)`, currently L1366-1379), read
   the new env knob `WHALE_BWD_KV_WARPSPEC` near the top of
   `_whale_attn_bwd_impl` (~L1223 area) and pass `WARPSPEC=` as a
   constexpr kwarg.

The behavior when `WHALE_BWD_KV_WARPSPEC=0` (default unset → 0) is
**byte-identical** to the current kernel: `WARPSPEC=False` selects the
plain `for m_block in range(...)` branch.

---

## Edit 1 — kernel signature (`_attn_bwd_dkdv_kernel`, L427-446)

### BEFORE (L427-446)

```python
def _attn_bwd_dkdv_kernel(
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
):
```

### AFTER

```python
def _attn_bwd_dkdv_kernel(
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
    WARPSPEC: tl.constexpr = False,
    NUM_STAGES_WS: tl.constexpr = 3,
):
```

Notes:
- Defaults preserve current behavior — old call sites that do not pass
  `WARPSPEC` still see `False`.
- `NUM_STAGES_WS` is a separate compile-time knob so the autotuner-chosen
  `num_stages` (which controls software pipelining at the Triton-IR
  level) is decoupled from the warp-spec stage count. Defaulting to 3
  matches the FA3 producer/consumer depth on H100 SXM.

---

## Edit 2 — M-loop body (`_attn_bwd_dkdv_kernel`, L475-506)

### BEFORE (L475-477)

```python
    for hg in range(group):
        h = kv_h * group + hg
        for m_block in range(m_start_block, m_end_block):
```

### AFTER

```python
    for hg in range(group):
        h = kv_h * group + hg
        if WARPSPEC:
            m_iter = tl.range(m_start_block, m_end_block, 1,
                              num_stages=NUM_STAGES_WS,
                              warp_specialize=True)
        else:
            m_iter = range(m_start_block, m_end_block)
        for m_block in m_iter:
```

Body (L478-506) is unchanged. The Triton compiler treats both branches
as compile-time-resolved (since `WARPSPEC` is `tl.constexpr`), so only
one loop form is emitted per autotune key.

Notes:
- `tl.range(start, stop, step, num_stages=N, warp_specialize=True)` is
  the verified Triton 3.6 cu130 signature. We pass `step=1` explicitly
  to match the documented form.
- `async_task` and `num_consumer_groups` are NOT exposed in this Triton
  build (verified before staging this lever); the compiler picks the
  producer/consumer warp split automatically.

---

## Edit 3 — dispatch site (`_whale_attn_bwd_impl`, L1223 + L1366-1379)

### BEFORE — env read (L1223-1244 region)

```python
    variant = os.environ.get("WHALE_BWD_VARIANT", "auto")
    fused_delta_t_max = int(os.environ.get("WHALE_FUSED_DELTA_T_MAX", "3072"))
```

### AFTER — env read (insert one line after the `fused_delta_t_max` read)

```python
    variant = os.environ.get("WHALE_BWD_VARIANT", "auto")
    fused_delta_t_max = int(os.environ.get("WHALE_FUSED_DELTA_T_MAX", "3072"))
    use_kv_warpspec = os.environ.get("WHALE_BWD_KV_WARPSPEC", "0") == "1"
```

### BEFORE — call site (L1365-1379)

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

### AFTER — call site (only the trailing kwargs gain `WARPSPEC=`)

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
            WARPSPEC=use_kv_warpspec,
        )
```

Behavior with `WHALE_BWD_KV_WARPSPEC` unset → `use_kv_warpspec=False`
→ identical to today (kernel default `WARPSPEC: tl.constexpr = False`
also defends).

---

## Autotune configs (`_bwd_kv_configs()`, L97-106)

Current list (kept intact when `WARPSPEC=False`):

```python
for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128)]:
    for w in (4, 8):
        for s in (2, 3, 4, 5):
            configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                         num_warps=w, num_stages=s))
```

`warp_specialize=True` requires:
- `num_stages >= 2` — the `s=2,3,4,5` spread already satisfies this.
- `num_warps in {4, 8}` — already covered.

**Recommendation:** keep the existing config list as-is for the first
run. Reasons:

1. The current sweep gives the autotuner enough room to find a
   warpspec-friendly winner without dropping any candidate.
2. Compile time at `WARPSPEC=True` will be 2–3× higher per config
   because each loop is lowered twice (producer + consumer); the
   `(128, 256)` and `(256, 128)` shapes will be slow to compile but
   may unlock the best dkdv throughput at T=8192. We do NOT want to
   drop them blind.
3. If autotune time is unacceptable on the pod (>10 min for a single
   shape), trim to **`(s in {3, 4})`** and **`(bm, bn) in {(128,128), (128,256), (256,128)}`**
   for a follow-up, and pin via `WHALE_BWD_KV_CONFIG=BM,BN,W,S` once a
   winner is identified.

**No new configs are required** for this lever — `warp_specialize=True`
piggybacks on existing `num_stages`/`num_warps` candidates.

---

## maxnreg note (atomic_add gotcha)

`_attn_bwd_dkdv_kernel` writes dK/dV via plain `tl.store` (L512-513).
There is **no `tl.atomic_add` in this kernel**, so the
`maxnreg >= 224` corruption gotcha (memory: triton_gotchas_atomic_add,
evidence: `legs/2026-04-16_whale_bwd_persistent_atomic/hypothesis.md`)
does NOT gate this lever. We are not adding `maxnreg=` to
`_bwd_kv_configs()` here, so the kernel runs at the Triton default
register budget and the atomic_add bug cannot be triggered through
this path.

If a future patch raises maxnreg on this kernel to coax warp-spec
performance, that is still safe **for this kernel only** — but the
constraint still applies to any kernel using `tl.atomic_add`
(`_attn_bwd_fused_kernel`, etc.).
