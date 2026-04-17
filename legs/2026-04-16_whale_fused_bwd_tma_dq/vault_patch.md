# vault_patch.md — H8 / Lever B (TMA dQ in fused bwd)

**Status: doc-only. Do NOT edit `vault/whale_kernel_triton.py` automatically;
the user applies this patch manually.**

Three changes to `vault/whale_kernel_triton.py` (**line numbers re-verified
2026-04-16 (third pass) against the CURRENT vault, 1554 lines; Lever A has
been REVERTED (no `WHALE_BWD_KV_WARPSPEC`, no `use_kv_warpspec`, no
WARPSPEC/NUM_STAGES_WS constexprs in `_attn_bwd_dkdv_inline_delta_kernel`).
The early-exit-masking refactor was applied to `_attn_bwd_dkdv_kernel`
(L427) only — entirely outside Lever B's surgery zone. Anchors:**):

1. Add a new autotune config builder `_bwd_fused_tma_dq_configs()` capped at
   `maxnreg <= 192` (next to `_bwd_fused_configs()` at line 174, which now
   ends at line 206).
2. Add a new kernel `_attn_bwd_fused_tma_dq_kernel` that mirrors
   `_attn_bwd_fused_kernel` (decorator at line 776, `def` at line 782, body
   ends at line 877) but accumulates dQ into a per-(b,h)
   `tl.make_tensor_descriptor` via `descriptor.atomic_add`.
3. Extend the dispatch site (`variant = os.environ.get(...)` at line 1254)
   to recognize `WHALE_BWD_VARIANT=fused_bwd_tma_dq` and route to the new
   kernel, calling `_ensure_tma_allocator()` (helper at line 37) first.

---

## Triton API confirmation

From the local Triton 3.6.0 install (`/home/frosty40/miniconda3/lib/python3.13/
site-packages/triton/language/`):

- `core.py:2267` — `tl.make_tensor_descriptor(base, shape, strides,
  block_shape, padding_option="zero")`. Docs: base must be 16-byte aligned,
  leading strides multiples of 16-byte, **last dim contiguous**. 2-5D supported.
- `core.py:1410` — `descriptor.atomic_add(self, offsets, value)`. The offsets
  list length must equal `len(block_shape)` and value.shape must equal
  `block_shape` (`semantic.py:1102–1106`).
- `semantic.py:1115–1120` — `descriptor_atomic_add` lowers to
  `create_descriptor_reduce(kind=ADD, ...)`. **Allowed dtypes:
  `{uint32, int32, uint64, float32, float16, bfloat16}`** — fp32 is what we
  need (DQ_F32 is fp32). No native-TMA branch is required for fp32 (only
  fp16/bf16 min/max gates on `_has_native_tma`).
- The pod is sm90 H100, so `_has_native_tma` returns True regardless; this
  reduce-add lowers to `cp.reduce.async.bulk.add.f32` — same primitive as
  FA3 hopper bwd's `SM90_BULK_REDUCE_ADD` for dQ.

---

## Change 1 — new config list

Insert after `_bwd_fused_configs()` (currently ends at line 206):

```python
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
```

---

## Change 2 — new kernel `_attn_bwd_fused_tma_dq_kernel`

Insert immediately after `_attn_bwd_fused_kernel` ends (line 877, before the
`@triton.jit` for `_attn_bwd_dq_cast_kernel` at line 880). The kernel is a
verbatim copy of `_attn_bwd_fused_kernel` with two edits:

- A `DQ_desc` is built per (b, h) via `tl.make_tensor_descriptor`.
- The dQ accumulation goes through `DQ_desc.atomic_add(...)` instead of
  scalar `tl.atomic_add` on raw pointers.

### BEFORE (lines 830–870 of `_attn_bwd_fused_kernel`, current vault @ 1554 lines)

```python
830    for hg in range(group):
831        h = kv_h * group + hg
832        for m_block in range(m_start_block, m_end_block):
833            start_m = m_block * BLOCK_M
834            offs_m_cur = start_m + offs_m
835            row_mask = offs_m_cur < T_MAX
836            q_mask = row_mask[:, None] & (offs_d[None, :] < D)
837
838            q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_cur[:, None] * stride_qt + offs_d[None, :] * stride_qd
839            do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_cur[:, None] * stride_dot + offs_d[None, :] * stride_dod
840            o_ptrs = O + b * stride_ob + h * stride_oh + offs_m_cur[:, None] * stride_ot + offs_d[None, :] * stride_od
841            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
842            do = tl.load(do_ptrs, mask=q_mask, other=0.0)
843            o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)
844            delta = tl.sum(o_tile.to(tl.float32) * do.to(tl.float32), axis=1)
845
846            lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m_cur * stride_lt
847            lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)
848
849            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
850            p = tl.exp2(s - lse[:, None] * LOG2E)
851
852            if IS_CAUSAL:
853                p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX) & (offs_m_cur[:, None] >= offs_n[None, :])
854            else:
855                p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX)
856            p = tl.where(p_mask, p, 0.0)
857
858            dv = tl.dot(tl.trans(p).to(q.dtype), do, acc=dv, out_dtype=tl.float32)
859            dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
860            ds = p * (dp - delta[:, None])
861            dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk, out_dtype=tl.float32)
862
863            # dq contribution from this KV block: ds @ K * SCALE -> BLOCK_M x BLOCK_D,
864            # atomic-added into the fp32 scratch. The final SCALE is applied here so
865            # each per-block contribution is already in the output's scale; the cast
866            # kernel only downshifts fp32 -> bf16.
867            dq_local = tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32) * SCALE
868            dq_ptrs = (DQ_F32 + b * stride_dqfb + h * stride_dqfh
869                       + offs_m_cur[:, None] * stride_dqft + offs_d[None, :] * stride_dqfd)
870            tl.atomic_add(dq_ptrs, dq_local, mask=q_mask)
```

### AFTER (the corresponding block in `_attn_bwd_fused_tma_dq_kernel`)

`tl.static_assert(BLOCK_D == D)` is added once near the top of the kernel
(after the `offs_d` line), since TMA descriptors require the block to
cover the full last-dim. The DQ descriptor is built per-(b, h) just like
the existing `_attn_bwd_dkdv_inline_delta_tma_kernel` does for Q_desc /
O_desc / DO_desc inside the `for hg in range(group):` loop at lines
**692–703** (with K/V/DK/DV descriptors at lines 659–674). The dq
accumulation becomes `DQ_desc.atomic_add([start_m, 0], dq_local)`.

```python
    tl.static_assert(BLOCK_D == D, "TMA-dq fused bwd requires BLOCK_D == D")

    # ... unchanged setup through the dk/dv accumulators and m_start_block ...

    for hg in range(group):
        h = kv_h * group + hg
        # Per-(b,h) descriptor over the fp32 dQ scratch. Last dim must be
        # contiguous (stride_dqfd == 1, satisfied by torch.zeros B,T,H,D).
        # Base = DQ_F32 + b*stride_dqfb + h*stride_dqfh; this offset is
        # 16-byte aligned because stride_dqfh = D fp32s = D*4 bytes (D
        # is a power of 2 >= 4) and stride_dqfb = T*H*D*4 bytes.
        DQ_desc = tl.make_tensor_descriptor(
            DQ_F32 + b * stride_dqfb + h * stride_dqfh,
            shape=[T_MAX, D],
            strides=[stride_dqft, 1],
            block_shape=[BLOCK_M, D],
        )

        for m_block in range(m_start_block, m_end_block):
            start_m    = m_block * BLOCK_M
            offs_m_cur = start_m + offs_m
            row_mask   = offs_m_cur < T_MAX
            q_mask     = row_mask[:, None] & (offs_d[None, :] < D)

            # --- unchanged Q/DO/O loads, delta, lse, s, p, dv, dp, ds, dk ---
            #     (lines 838–861 of _attn_bwd_fused_kernel, copied verbatim)

            # dq_local: same math, same SCALE; only the store changes.
            dq_local = tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32) * SCALE
            # Boundary tile masking: TMA does its own OOB clipping
            # (padding_option="zero" by default), but we still want to
            # avoid reducing into rows that are past T_MAX. For the
            # T_MAX-aligned case (T % BLOCK_M == 0) this is a no-op; for
            # the tail block we mask dq_local before the reduce so the
            # OOB rows contribute exact zero.
            dq_local = tl.where(q_mask, dq_local, 0.0)
            DQ_desc.atomic_add([start_m, 0], dq_local)
```

Notes:
- `descriptor.atomic_add` does not accept a `mask=` kwarg
  (`core.py:1410`). Tail handling has to be done by zeroing the tile
  before the reduce; TMA's padding_option only protects loads/stores
  from OOB, not the reduce-add value. With `BLOCK_M ∈ {64,128}` and
  benchmark T values 2048/4096/8192 the tail is fully aligned, so this
  branch is just a safety net.
- The kernel signature is **identical** to `_attn_bwd_fused_kernel` —
  same args, same stride list, same constexprs. The dispatch at the
  Python side (Change 3) calls it with the exact same argument tuple
  used at lines 1283–1296.
- Use `@triton.autotune(configs=_bwd_fused_tma_dq_configs(), key=[...same...],
  reset_to_zero=["DQ_F32"])` on the new kernel (mirrors the decorator at
  lines 776–780 of `_attn_bwd_fused_kernel`).

---

## Change 3 — dispatch wire-up at line 1254

### BEFORE (lines 1254–1275 of the current vault @ 1554 lines)

```python
1254    variant = os.environ.get("WHALE_BWD_VARIANT", "auto")
1255    fused_delta_t_max = int(os.environ.get("WHALE_FUSED_DELTA_T_MAX", "3072"))
1256    if variant == "auto":
1257        use_fused_delta = T <= fused_delta_t_max
1258        use_tma_dkdv = False
1259        use_fused_bwd = False
1260    elif variant == "fused_delta_tma":
1261        use_fused_delta = True
1262        use_tma_dkdv = True
1263        use_fused_bwd = False
1264    elif variant == "fused_bwd":
1265        use_fused_delta = False
1266        use_tma_dkdv = False
1267        use_fused_bwd = True
1268    else:
1269        use_fused_delta = variant == "fused_delta"
1270        use_tma_dkdv = False
1271        use_fused_bwd = False
1272    # TMA requires BLOCK_D == D, which holds iff D is a power of 2.
1273    if use_tma_dkdv and block_d != D:
1274        use_tma_dkdv = False
1275    use_split_h = os.environ.get("WHALE_BWD_SPLIT_H", "0") == "1"
```

Note: Lever A has been REVERTED — `WHALE_BWD_KV_WARPSPEC` is NOT in the
current vault. Lever B's dispatch BEFORE block reflects the actual
current state. The early-exit refactor only touched
`_attn_bwd_dkdv_kernel` (line 427), which is in a separate kernel from
`_attn_bwd_fused_kernel` and from the dispatch site. No conflict.

### AFTER

Add a new boolean `use_fused_bwd_tma_dq` and wire it up. The existing
`use_fused_bwd` block (lines 1280–1307 of the current vault — body runs
from `dq_f32 = torch.zeros(...)` at 1281 through the final
`return dq, dk, dv` at 1307) becomes the union of `fused_bwd` and
`fused_bwd_tma_dq`; only the kernel handle differs.

```python
    variant = os.environ.get("WHALE_BWD_VARIANT", "auto")
    fused_delta_t_max = int(os.environ.get("WHALE_FUSED_DELTA_T_MAX", "3072"))
    if variant == "auto":
        use_fused_delta     = T <= fused_delta_t_max
        use_tma_dkdv        = False
        use_fused_bwd       = False
        use_fused_bwd_tma_dq = False
    elif variant == "fused_delta_tma":
        use_fused_delta     = True
        use_tma_dkdv        = True
        use_fused_bwd       = False
        use_fused_bwd_tma_dq = False
    elif variant == "fused_bwd":
        use_fused_delta     = False
        use_tma_dkdv        = False
        use_fused_bwd       = True
        use_fused_bwd_tma_dq = False
    elif variant == "fused_bwd_tma_dq":
        use_fused_delta     = False
        use_tma_dkdv        = False
        use_fused_bwd       = True   # share the dq_f32 scratch + cast path
        use_fused_bwd_tma_dq = True
    else:
        use_fused_delta     = variant == "fused_delta"
        use_tma_dkdv        = False
        use_fused_bwd       = False
        use_fused_bwd_tma_dq = False
    if use_tma_dkdv and block_d != D:
        use_tma_dkdv = False
    # The TMA-dq variant *also* needs BLOCK_D == D; fall back to the
    # scalar fused_bwd kernel if D is not a power of 2.
    if use_fused_bwd_tma_dq and block_d != D:
        use_fused_bwd_tma_dq = False
    use_split_h = os.environ.get("WHALE_BWD_SPLIT_H", "0") == "1"
```

Then in the `if use_fused_bwd:` block (line 1280), pick the kernel by
the new flag and `_ensure_tma_allocator()` when needed:

```python
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
            q.stride(0),  q.stride(1),  q.stride(2),  q.stride(3),
            k.stride(0),  k.stride(1),  k.stride(2),  k.stride(3),
            v.stride(0),  v.stride(1),  v.stride(2),  v.stride(3),
            o.stride(0),  o.stride(1),  o.stride(2),  o.stride(3),
            do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            dq_f32.stride(0), dq_f32.stride(1), dq_f32.stride(2), dq_f32.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
            BLOCK_D=block_d, D=D, IS_CAUSAL=causal,
        )
        # ... rest of the cast block stays unchanged (lines 1271–1281) ...
```

---

## Autotune config to use

`@triton.autotune(configs=_bwd_fused_tma_dq_configs(), key=["D", "IS_CAUSAL",
"NUM_HEADS", "NUM_KV_HEADS", "T_MAX"], reset_to_zero=["DQ_F32"])` — the same
autotune key + `reset_to_zero` list as `_attn_bwd_fused_kernel` (lines 749–753);
only the configs callable changes. Generated configs (with maxnreg=192):

```
BLOCK_M, BLOCK_N, num_warps, num_stages
64,  64,  4, 3 / 4 3 / 4   (with maxnreg=192)
64, 128,  4, 3 / 4, 4 / 8, 3 / 8, 4
128, 64,  4, 3 / 4, 4 / 8, 3 / 8, 4
128,128,  4, 3 / 4, 4 / 8, 3 / 8, 4
64,  64,  4, 2                            (fallback)
```

(17 configs total — same cardinality as `_bwd_fused_configs()`.)

---

## Safety checks (re-verified 2026-04-16 against local Triton 3.6.0 source)

### Triton 3.6 TMA constraints — facts from the source

- **`block_shape` every dim must be a power of 2.** `_utils.py:48–58`
  (`validate_block_shape`) raises `ValueError` if any element is not a
  power of 2. The current BLOCK_M set {64, 128} is compliant. D must be
  a power of 2 — the patch already gates on `block_d == D`.
- **Last-dim block byte size ≥ 16.** `semantic.py:1938–1941` requires
  `contig_dim_size * elem_size >= 16`. D * 4 ≥ 16 → D ≥ 4; whale has
  D ∈ {80, 96} → OK for power-of-2 D only (80 is not pow2; whale will
  fall back to non-TMA if D=80).
- **Last-dim stride must be 1.** `semantic.py:1943–1945` enforces
  `strides[-1] == 1`. We pass literal `1`, and the underlying
  `torch.zeros(B,T,H,D)` has `stride_dqfd == 1` ✓.
- **Base pointer 16-byte aligned, leading strides multiples of 16 bytes.**
  `core.py:2277–2280` (docstring — the runtime check is in the TMA
  driver, not in Triton's Python layer). For fp32 this is `stride * 4`
  bytes; `stride_dqft = H*D` elements, so `H*D*4` bytes must be a
  multiple of 16 → H*D must be a multiple of 4.
- **fp32 is a verified-supported dtype for `descriptor.atomic_add`.**
  `semantic.py:1117` asserts
  `desc.dtype in {uint32, int32, uint64, float32, float16, bfloat16}`.
  **fp32 does not require the `_has_native_tma` gate** (that gate is
  only tripped by `descriptor_atomic_min/max` with fp16/bf16 at
  `semantic.py:1128–1129`). On H100 sm90 `_has_native_tma()` is True
  anyway (`semantic.py:1122–1124`).
- **No `mask=` on `descriptor.atomic_add`.** `core.py:1410` signature is
  `atomic_add(self, offsets, value, _semantic=None)` — tiles must be
  pre-zeroed (the patch does this via `tl.where(q_mask, dq_local, 0.0)`).
- **Per-program descriptor creation inside a `@triton.jit` kernel is
  fine and already in production.** `vault/whale_kernel_triton.py` does
  this in several places:
  - `_attn_fwd_kernel` — `tl.make_tensor_descriptor` at lines 329–344.
  - `_attn_bwd_dkdv_inline_delta_tma_kernel` — K/V/DK/DV descriptors
    built once at 659–674, then Q/O/DO built **per `hg` iteration
    inside the `for hg in range(group)` loop** at 692–703. This is the
    exact pattern the new kernel will use for `DQ_desc`.

### DQ_F32 alignment — safe as-is

`dq_f32 = torch.zeros(B, T, H, D, device=q.device, dtype=torch.float32)`
at dispatch line 1255 gives a freshly allocated contiguous tensor. The
PyTorch CUDA caching allocator returns blocks aligned to at least
**512 bytes** (see `c10/cuda/CUDACachingAllocator.cpp`,
`kMinBlockSize = 512`), which easily satisfies the 16-byte base
requirement. Per-(b, h) offsets shift by `b*(T*H*D) + h*D` fp32 elements
= `(b*T*H + h)*D*4` bytes; with D a power of 2 ≥ 16 this is a multiple
of 16 B ✓.

No padding is needed; no alignment assertion is required in the
dispatch code. A cheap runtime assertion could be added in Python for
defensiveness:

```python
assert dq_f32.data_ptr() % 16 == 0, "dq_f32 base not 16-byte aligned"
assert dq_f32.stride(-1) == 1, "dq_f32 last dim not contiguous"
```

These are optional — both hold by construction.

### `_ensure_tma_allocator` inside `@triton.jit`? No — it's a host call.

`_ensure_tma_allocator` at vault line 37 is a **plain Python function**
that calls `triton.set_allocator(...)`. It must be called from host
code (the dispatch site), not from within the kernel. The patch already
does this correctly in Change 3: `if use_fused_bwd_tma_dq:
_ensure_tma_allocator()` runs on the host before the `fused_kernel[...]`
launch.

### maxnreg cap — 192 is the right cap; no further drop needed a priori

The `tl.atomic_add` corruption at `maxnreg >= 224`
(`legs/2026-04-16_whale_bwd_persistent_atomic/hypothesis.md`) was a
**scalar-atomic-pointer** regression, not a TMA-atomic-reduce
regression — the code paths are entirely disjoint:
`tl.atomic_add(ptr, val, mask=...)` lowers through
`semantic.atomic_add` (pointer-based atomics), whereas
`desc.atomic_add(offsets, val)` lowers through
`create_descriptor_reduce` → `cp.reduce.async.bulk.add.f32`
(a TMA bulk-reduce SASS instruction). The 192 cap is a conservative
belt-and-suspenders in case a shared register-pressure effect exists;
we do **not** have evidence of a second regression at 192 on the TMA
path. Proposed autotune sweep (post-bench): lift to 224 if 192 is
stable, then drop the cap entirely if 224 is stable. Until that sweep
lands, 192 stays.

Fallback plan if TMA atomic_add is unsupported or wrong at 192:
1. Try `extra["maxnreg"] = 128`.
2. If still broken, pin a single config (BLOCK_M=64, BLOCK_N=64,
   num_warps=4, num_stages=2, maxnreg=128) via
   `WHALE_BWD_FUSED_MAXNREG=128` plus a `_env_force` override.
3. If broken regardless of maxnreg, the variant is not viable on this
   Triton build — user falls back to `WHALE_BWD_VARIANT=fused_bwd`.

### Other uncertainties tracked

- TMA atomic-reduce correctness under compile-time `num_stages >= 4`
  has not been independently verified on this pod. First bench should
  run with a reduced config list (`num_stages=3`, `num_warps=4`,
  `BLOCK_M=BLOCK_N=64`) to isolate correctness before sweeping.
- Tail handling (T % BLOCK_M != 0) relies on `tl.where(q_mask, ...)`
  zeroing OOB rows before the reduce. TMA itself drops OOB loads/stores
  but does not drop OOB reduce values — any non-zero fp32 in those
  rows would be atomically added at a clipped coordinate. Benchmark T
  values (2048, 4096, 8192) are all multiples of 128 so the zeroing
  path is a safety net only; pathological T values would exercise it.
