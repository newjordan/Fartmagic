# vault_patch.md — H9b only (doc-only; do NOT apply without user approval)

This document describes the kernel patch that Phase C of `run.sh` would
exercise. It MUST NOT be applied to `vault/whale_kernel_triton.py` without
explicit user approval. Phase A and Phase B in `run.sh` require zero vault
edits.

## Target file
`/home/frosty40/SOTA_FINAL/vault/whale_kernel_triton.py`

## Why a patch
The K/V loop in `_attn_fwd_tma_kernel` (lines 359-376) currently iterates as
a single warpgroup with no producer/consumer split. FA3 Sm90 fwd uses a
warp-specialized split where one warp issues TMA loads and another runs
WGMMA. Triton 3.6 cu130 exposes the same effect via
`tl.range(..., warp_specialize=True)` — this is the only knob available in
this stack (`tl.async_task` is NOT exposed).

## Cited locations (re-verified 2026-04-16 post-Lever-A edits)
Vault is now 1533 lines. Lever A added `WARPSPEC: tl.constexpr` + `NUM_STAGES_WS`
to `_attn_bwd_dkdv_kernel` (lines 427-448; iterator switch at 479-484). That
edit did not touch the fwd path. Lever C line numbers shifted as follows:

- TMA forward kernel definition: **lines 304-384** (was 302-384; +2).
- K/V loop to be patched: **line 359** (unchanged; `for start_n in range(0, hi, BLOCK_N):`).
- Autotune config list: `_fwd_tma_configs()` at **lines 81-94** (unchanged).
- Config-decoder helper used by `WHALE_FWD_TMA_CONFIG`: `_env_force` at **lines 50-64** (unchanged).
- Dispatch site that picks the TMA kernel: **lines 1179-1192** in `_whale_attn_fwd_impl` (was 1171-1184; +8).
- `_whale_attn_fwd_impl` def: **line 1164** (was 1156; +8).

## Patch — kernel signature (line 304)

BEFORE (lines 304-319):
```python
@triton.autotune(configs=_fwd_tma_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_fwd_tma_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qt, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_ob, stride_ot, stride_oh,
    stride_lb, stride_lh, stride_lt,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
```

AFTER:
```python
@triton.autotune(configs=_fwd_tma_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_fwd_tma_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qt, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_ob, stride_ot, stride_oh,
    stride_lb, stride_lh, stride_lt,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WARPSPEC: tl.constexpr = False,
    NUM_STAGES_WS: tl.constexpr = 3,
):
```

Note: align with Lever A's naming convention in `_attn_bwd_dkdv_kernel`
(lines 446-447: `WARPSPEC: tl.constexpr = False, NUM_STAGES_WS: tl.constexpr = 3`).
Using `NUM_STAGES_WS` (not `NUM_STAGES`) keeps Lever A and Lever C consistent.

## Patch — K/V loop (line 359)

BEFORE (lines 359-376):
```python
    for start_n in range(0, hi, BLOCK_N):
        offs_n_cur = start_n + offs_n
        k = K_desc.load([start_n, 0])
        v = V_desc.load([start_n, 0])

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
        if IS_CAUSAL:
            valid = (offs_n_cur[None, :] < T_MAX) & (offs_m[:, None] >= offs_n_cur[None, :])
        else:
            valid = offs_n_cur[None, :] < T_MAX
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.where(m_new > float("-inf"), tl.exp2(m_i - m_new), 1.0)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(q.dtype), v, acc=acc * alpha[:, None], out_dtype=tl.float32)
        m_i = m_new
```

AFTER (only the loop header line changes; body identical):
```python
    for start_n in tl.range(0, hi, BLOCK_N, num_stages=NUM_STAGES_WS, warp_specialize=WARPSPEC):
        offs_n_cur = start_n + offs_n
        k = K_desc.load([start_n, 0])
        v = V_desc.load([start_n, 0])

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
        if IS_CAUSAL:
            valid = (offs_n_cur[None, :] < T_MAX) & (offs_m[:, None] >= offs_n_cur[None, :])
        else:
            valid = offs_n_cur[None, :] < T_MAX
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.where(m_new > float("-inf"), tl.exp2(m_i - m_new), 1.0)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(q.dtype), v, acc=acc * alpha[:, None], out_dtype=tl.float32)
        m_i = m_new
```

## Patch — dispatch site (lines 1179-1192)

The wrapper must read `WHALE_FWD_TMA_WARPSPEC` and pass it through, plus a
matching `NUM_STAGES_WS` so the loop's `num_stages` matches the kernel's
`num_stages` (Triton 3.6 requires consistency when warp_specialize=True).

BEFORE (lines 1179-1192):
```python
    use_tma_fwd = os.environ.get("WHALE_FWD_VARIANT", "default") == "tma" and block_d == D
    if use_tma_fwd:
        _ensure_tma_allocator()
        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _attn_fwd_tma_kernel[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            lse.stride(0), lse.stride(1), lse.stride(2),
            T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
            D=D, IS_CAUSAL=causal,
        )
```

AFTER:
```python
    use_tma_fwd = os.environ.get("WHALE_FWD_VARIANT", "default") == "tma" and block_d == D
    if use_tma_fwd:
        _ensure_tma_allocator()
        warpspec = os.environ.get("WHALE_FWD_TMA_WARPSPEC", "0") == "1"
        num_stages_ws = int(os.environ.get("WHALE_FWD_TMA_NUM_STAGES", "3"))
        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _attn_fwd_tma_kernel[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            lse.stride(0), lse.stride(1), lse.stride(2),
            T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
            D=D, IS_CAUSAL=causal,
            WARPSPEC=warpspec, NUM_STAGES_WS=num_stages_ws,
        )
```

## Autotune config list constraints

`tl.range(..., warp_specialize=True)` on Triton 3.6 cu130 requires:
- `num_stages >= 2` (the loop must be pipelineable; `s=2` is the minimum and
  `s in {2, 3}` are the safe values for fwd-attention K/V loops on H100).
- `num_warps in {4, 8}` — both are exposed today; `num_warps=8` is needed for
  the producer/consumer split to actually parcel out warps to TMA vs WGMMA.
- `BLOCK_M`, `BLOCK_N` powers of 2 — already enforced by `_fwd_tma_configs`
  at lines 81-94 (current spread: `(64,64), (64,128), (128,64), (128,128)`).
- `BLOCK_D == D` — already enforced (TMA descriptor uses `block_shape=[BLOCK_*, D]`
  at lines 333, 337, 341, 345, so non-power-of-2 D would fail to launch).

## TMA + warp_specialize interaction (Triton 3.6 backend evidence)

Checked `/home/frosty40/miniconda3/lib/python3.13/site-packages/triton/` on this
host. H100 (cap 9) is the Hopper path; `capability // 10 in [8, 9]` branch at
`triton/backends/nvidia/compiler.py:268-277` routes warp_specialize to
`nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)`
(line 274). That Hopper pass is the same mechanism FA3 uses to split TMA loads
from WGMMA on Sm90. The `tl.range` plumbing (`triton/language/core.py:3286-3303`
`range.__init__` accepts `warp_specialize=bool`; IR attribute set at
`triton/compiler/code_generator.py:1224-1225` as `tt.warp_specialize`) is
agnostic to whether the loaded tensors came from `tl.make_tensor_descriptor`
or plain pointers — the attribute tags the for-op and the Hopper pass partitions
the loop body regardless of load source.

Stale-doc warning: the `tl.range` docstring (`core.py:3281-3283`) says
"warp specialization is only supported on Blackwell GPUs." That is inaccurate
against this build — the Hopper warpspec pass (line 274 of the nvidia compiler)
is present and active when `capability // 10 == 9`. Treat the docstring as
under-documented, not as a disallow.

No extra constraint was found for TMA descriptors inside a `warp_specialize=True`
loop beyond the general ones already listed (num_stages>=2, num_warps in {4,8}).
Lever A already exercised WARPSPEC on a non-TMA bwd kernel at
`vault/whale_kernel_triton.py:479-484`. Lever C extends the same pattern onto
a TMA-using fwd kernel — the compiler path and the IR attribute are identical.

If the patch is applied and Phase C autotune misbehaves (e.g. compiler errors
on `s=4` configs with WARPSPEC=True), narrow `_fwd_tma_configs` for the
warp-spec path:
```python
def _fwd_tma_configs():
    forced = _env_force("FWD_TMA")
    if forced:
        return forced
    warpspec = os.environ.get("WHALE_FWD_TMA_WARPSPEC", "0") == "1"
    stages_grid = (2, 3) if warpspec else (2, 3, 4)
    warps_grid = (8,) if warpspec else (4, 8)
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128)]:
        for w in warps_grid:
            for s in stages_grid:
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=160))
    return configs
```

## Correctness gate (post-patch)
Before any Phase C run is trusted, re-run the existing whale numerics test
with `WHALE_FWD_VARIANT=tma WHALE_FWD_TMA_WARPSPEC=1` and confirm
`max(abs(out - ref)) <= 1e-3` (bf16). If the gate fails, revert the vault.
