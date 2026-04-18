"""Triton kernels for whale causal attention with a native Triton backward.

Forward:
  _attn_fwd_kernel  ->  O (same dtype as Q) and LSE (float32, natural log)

Backward:
  _attn_bwd_dkdv_kernel  ->  dK, dV  (one program per (batch, kv_head, n_block))
  _attn_bwd_dq_kernel    ->  dQ     (one program per (batch, q_head,  m_block))

Both the forward and the backward are wrapped as `torch.library.custom_op`s so
that `torch.compile` treats each launch as an opaque node and does not attempt
to trace into the Triton kernel. The forward op returns (O, LSE); the autograd
glue saves (q, k, v, o, lse) and dispatches to the backward custom_op.

GQA is supported: Q has NUM_HEADS, K/V have NUM_KV_HEADS, with NUM_HEADS a
multiple of NUM_KV_HEADS. dK/dV sum contributions from all Q heads that share a
given KV head inside the kernel (no explicit repeat_interleave materialisation).
"""
from __future__ import annotations

import math
import os
from typing import Tuple

import torch
import triton
import triton.language as tl


LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)


_TMA_ALLOCATOR_SET = False


def _ensure_tma_allocator():
    """One-shot install of a Triton allocator backing on-device TMA
    descriptors. Triton calls this callback when a kernel using
    tl.make_tensor_descriptor needs a workspace region."""
    global _TMA_ALLOCATOR_SET
    if _TMA_ALLOCATOR_SET:
        return
    def _alloc(size: int, align: int, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    triton.set_allocator(_alloc)
    _TMA_ALLOCATOR_SET = True


def _env_force(name):
    """If WHALE_<name>_CONFIG is set as 'BM,BN,warps,stages', return a single-config list.
    Optionally honours WHALE_<name>_MAXNREG to set `maxnreg=<int>` on the config."""
    spec = os.environ.get(f"WHALE_{name}_CONFIG")
    if not spec:
        return None
    parts = [int(x) for x in spec.split(",")]
    bm, bn, w, s = parts
    maxnreg_env = os.environ.get(f"WHALE_{name}_MAXNREG")
    kwargs = {}
    if maxnreg_env:
        mr = int(maxnreg_env)
        if mr > 0:
            kwargs["maxnreg"] = mr
    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s, **kwargs)]


def _fwd_configs():
    """maxnreg=160 saved ~3us on the headline fwd (legs/2026-04-16_whale_fwd_warpspec_tma).
    BM=192 attempted 2026-04-17 but Triton 3.6 requires tl.arange(0, BLOCK_M) pow-2;
    reverted. Expansion strategy: maxnreg ablation on small tiles + num_stages=6
    on big tiles. Autotune key includes T_MAX for per-shape winner selection."""
    forced = _env_force("FWD")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (128, 64), (128, 128), (64, 128), (256, 64), (128, 256)]:
        for w in (4, 8):
            for s in (2, 3, 4, 5):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=160))
    # maxnreg=128 ablation on small tiles (higher SM occupancy)
    for bm, bn in [(64, 64), (64, 128)]:
        for w in (4, 8):
            for s in (3, 4):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=128))
    # maxnreg=224 ablation on big tiles (less register spill)
    for bm, bn in [(128, 128), (128, 256), (256, 64)]:
        for w in (4, 8):
            for s in (3, 4):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=224))
    # num_stages=6 on big tiles (deeper async pipeline)
    for bm, bn in [(128, 64), (128, 128), (256, 64)]:
        for w in (4, 8):
            configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                         num_warps=w, num_stages=6, maxnreg=160))
    return configs


def _fwd_tma_configs():
    """Config list for the TMA forward variant. TMA forces BLOCK_D == D so the
    config search is narrower than the non-TMA grid. maxnreg=160 matches the
    non-TMA fwd winner."""
    forced = _env_force("FWD_TMA")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128)]:
        for w in (4, 8):
            for s in (2, 3, 4):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=160))
    return configs


def _bwd_kv_configs():
    forced = _env_force("BWD_KV")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128)]:
        for w in (4, 8):
            for s in (2, 3, 4, 5):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s))
    for bm, bn, w, s in [
        ( 64, 128,  4, 6),
        ( 64, 128,  4, 7),
        (128, 128,  8, 6),
        ( 64, 256,  8, 3),
        ( 64, 256,  8, 4),
        (128, 256, 16, 3),
        (128, 256, 16, 4),
        ( 32, 512,  8, 2),
    ]:
        configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s))
    return configs


def _bwd_kv_inline_configs():
    """Config list for _attn_bwd_dkdv_inline_delta_kernel. The ablation sweep
    in legs/2026-04-16_whale_bwd_ablations picked (64,64,4,3) at the headline
    shape (B=4,T=2048,H=8,KV=4,D=64). maxnreg=224 from legs/2026-04-16_whale_fwd_warpspec_tma
    shaves ~5us by keeping register-heavy dkdv out of spill territory."""
    forced = _env_force("BWD_KV")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128)]:
        for w in (4, 8):
            for s in (3, 4):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=224))
    configs.append(triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},
                                 num_warps=4, num_stages=2, maxnreg=224))
    return configs


def _bwd_kv_inline_tma_configs():
    """Config list for _attn_bwd_dkdv_inline_delta_tma_kernel. TMA forces
    BLOCK_D == D so we keep the same shape-winner spread as the non-TMA list.
    maxnreg=224 applied for the same reason as the non-TMA variant."""
    forced = _env_force("BWD_KV_TMA")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128)]:
        for w in (4, 8):
            for s in (3, 4):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=224))
    configs.append(triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},
                                 num_warps=4, num_stages=2, maxnreg=224))
    return configs


def _bwd_q_configs():
    """Same tile search as the fwd kernel, but maxnreg=224 since the bwd dq
    path has heavier register pressure (see maxnreg sweep in
    legs/2026-04-16_whale_fwd_warpspec_tma)."""
    forced = _env_force("BWD_Q")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (128, 64), (128, 128), (64, 128), (256, 64), (128, 256)]:
        for w in (4, 8):
            for s in (2, 3, 4, 5):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=224))
    return configs


def _bwd_kv_split_h_configs():
    forced = _env_force("BWD_KV_SPLIT_H")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128)]:
        for w in (4, 8):
            for s in (2, 3, 4):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s))
    return configs


def _bwd_fused_configs():
    """Config list for _attn_bwd_fused_kernel (persistent dkdv+dq with fp32
    atomic_add on dQ). Kept separate from _bwd_kv_inline_configs because
    maxnreg=224 breaks atomic_add correctness on this Triton 3.6 stack:
    `tl.atomic_add` of a 2-D fp32 tile amplified values ~19005x when
    maxnreg=224 was forced (row*1 + col*1000 probe, observed on H100 SXM,
    see legs/2026-04-16_whale_bwd_persistent_atomic/hypothesis.md).

    Default: no maxnreg. Override per-sweep via WHALE_BWD_FUSED_MAXNREG or
    WHALE_BWD_FUSED_CONFIG."""
    forced = _env_force("BWD_FUSED")
    if forced:
        return forced
    maxnreg_env = os.environ.get("WHALE_BWD_FUSED_MAXNREG", "").strip()
    extra = {}
    if maxnreg_env:
        mr = int(maxnreg_env)
        if mr > 0:
            if mr >= 224:
                raise ValueError(
                    f"WHALE_BWD_FUSED_MAXNREG={mr} is at/above the 224 threshold"
                    " that corrupts tl.atomic_add on this stack. Refusing."
                )
            extra["maxnreg"] = mr
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128)]:
        for w in (4, 8):
            for s in (3, 4):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, **extra))
    configs.append(triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},
                                 num_warps=4, num_stages=2, **extra))
    return configs


def _bwd_fused_tma_dq_configs():
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


@triton.autotune(configs=_fwd_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
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
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NUM_HEADS
    h = bh % NUM_HEADS
    kv_h = h // (NUM_HEADS // NUM_KV_HEADS)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = (offs_m[:, None] < T_MAX) & (offs_d[None, :] < D)
    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    qk_scale_log2 = SCALE * LOG2E

    if IS_CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, T_MAX)
        # Fully-unmasked upper bound: tile at start_n..start_n+BLOCK_N-1 is
        # strictly below diagonal iff start_n + BLOCK_N <= pid_m * BLOCK_M,
        # and within T_MAX since pid_m*BLOCK_M <= T_MAX when valid.
        lo_end = (pid_m * BLOCK_M // BLOCK_N) * BLOCK_N
    else:
        hi = T_MAX
        lo_end = hi

    # Phase 1: fully-unmasked tiles. Skip tl.where + KV row mask; keep only D-dim mask.
    for start_n in range(0, lo_end, BLOCK_N):
        offs_n_cur = start_n + offs_n
        k_mask = offs_d[None, :] < D
        k_ptrs = K + b * stride_kb + kv_h * stride_kh + offs_n_cur[:, None] * stride_kt + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + kv_h * stride_vh + offs_n_cur[:, None] * stride_vt + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # Guard first update where m_i = -inf; invalid q rows keep m_i at 0 after iter 1, fine.
        alpha = tl.where(m_new > float("-inf"), tl.exp2(m_i - m_new), 1.0)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(q.dtype), v, acc=acc * alpha[:, None], out_dtype=tl.float32)
        m_i = m_new

    # Phase 2: boundary + causal tiles.
    for start_n in range(lo_end, hi, BLOCK_N):
        offs_n_cur = start_n + offs_n
        k_mask = (offs_n_cur[:, None] < T_MAX) & (offs_d[None, :] < D)
        k_ptrs = K + b * stride_kb + kv_h * stride_kh + offs_n_cur[:, None] * stride_kt + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + kv_h * stride_vh + offs_n_cur[:, None] * stride_vt + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

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

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]

    out_mask = (offs_m[:, None] < T_MAX) & (offs_d[None, :] < D)
    out_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out.to(O.dtype.element_ty), mask=out_mask)

    lse = tl.where(l_i > 0, m_i * LN2 + tl.log(l_safe), 0.0)
    lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m * stride_lt
    tl.store(lse_ptrs, lse, mask=offs_m < T_MAX)


# ---------------------------------------------------------------------------
# Forward kernel (TMA variant, opt-in via WHALE_FWD_VARIANT=tma)
#
# Uses on-device tensor descriptors (tl.make_tensor_descriptor) for Q/K/V/O
# loads/stores. Requires BLOCK_D == D (no partial-D masking inside TMA). Falls
# back to the non-TMA kernel when D is not a supported power-of-2 tile width.
# ---------------------------------------------------------------------------


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
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NUM_HEADS
    h = bh % NUM_HEADS
    kv_h = h // (NUM_HEADS // NUM_KV_HEADS)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    Q_desc = tl.make_tensor_descriptor(
        Q + b * stride_qb + h * stride_qh,
        shape=[T_MAX, D], strides=[stride_qt, 1], block_shape=[BLOCK_M, D],
    )
    K_desc = tl.make_tensor_descriptor(
        K + b * stride_kb + kv_h * stride_kh,
        shape=[T_MAX, D], strides=[stride_kt, 1], block_shape=[BLOCK_N, D],
    )
    V_desc = tl.make_tensor_descriptor(
        V + b * stride_vb + kv_h * stride_vh,
        shape=[T_MAX, D], strides=[stride_vt, 1], block_shape=[BLOCK_N, D],
    )
    O_desc = tl.make_tensor_descriptor(
        O + b * stride_ob + h * stride_oh,
        shape=[T_MAX, D], strides=[stride_ot, 1], block_shape=[BLOCK_M, D],
    )

    q = Q_desc.load([pid_m * BLOCK_M, 0])

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    qk_scale_log2 = SCALE * LOG2E

    if IS_CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, T_MAX)
    else:
        hi = T_MAX

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

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    O_desc.store([pid_m * BLOCK_M, 0], out.to(O.dtype.element_ty))

    lse = tl.where(l_i > 0, m_i * LN2 + tl.log(l_safe), 0.0)
    lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m * stride_lt
    tl.store(lse_ptrs, lse, mask=offs_m < T_MAX)


# ---------------------------------------------------------------------------
# Backward preprocessor: delta_i = sum_d o_i,d * do_i,d
# ---------------------------------------------------------------------------


@triton.jit
def _attn_bwd_preprocess_kernel(
    O, DO, DELTA,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_dob, stride_dot, stride_doh, stride_dod,
    stride_deb, stride_deh, stride_det,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NUM_HEADS
    h = bh % NUM_HEADS
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < T_MAX) & (offs_d[None, :] < D)
    o_ptr = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    do_ptr = DO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod
    o = tl.load(o_ptr, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(DELTA + b * stride_deb + h * stride_deh + offs_m * stride_det, delta, mask=offs_m < T_MAX)



# ---------------------------------------------------------------------------
# Backward dK / dV kernel
# ---------------------------------------------------------------------------


@triton.autotune(configs=_bwd_kv_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
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

        for m_block in range(unmasked_start, m_end_block):
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


# ---------------------------------------------------------------------------
# Backward dK / dV kernel with inline Δ (fused-delta variant, H1 ablation)
# Computes Δᵢ = <oᵢ, doᵢ> locally from O and DO loads instead of reading it
# from a preprocessed tensor, allowing the preprocess kernel to be dropped.
# Trade-off: adds an O[BLOCK_M,D] load per (outer N block × Q-head-in-group ×
# inner M block). In GQA this amplifies O reads by H/KV; if that HBM cost is
# small relative to the preprocess-kernel launch + Δ HBM roundtrip this wins.
# ---------------------------------------------------------------------------


@triton.autotune(configs=_bwd_kv_inline_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_bwd_dkdv_inline_delta_kernel(
    Q, K, V, O, DO, DK, DV, LSE,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_dob, stride_dot, stride_doh, stride_dod,
    stride_dkb, stride_dkt, stride_dkh, stride_dkd,
    stride_dvb, stride_dvt, stride_dvh, stride_dvd,
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

    dk = dk * SCALE
    dk_ptrs = DK + b * stride_dkb + kv_h * stride_dkh + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + b * stride_dvb + kv_h * stride_dvh + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
    store_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=store_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=store_mask)


# ---------------------------------------------------------------------------
# Backward dK / dV with inline Δ and on-device TMA on Q/K/V/O/DO/DK/DV.
#
# Same math as _attn_bwd_dkdv_inline_delta_kernel. The bf16 HBM loads and
# stores go through tl.make_tensor_descriptor so Triton emits Hopper TMA
# bulk copies + swizzled SMEM layouts. Requires BLOCK_D == D.
# ---------------------------------------------------------------------------


@triton.autotune(configs=_bwd_kv_inline_tma_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_bwd_dkdv_inline_delta_tma_kernel(
    Q, K, V, O, DO, DK, DV, LSE,
    stride_qb, stride_qt, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_ob, stride_ot, stride_oh,
    stride_dob, stride_dot, stride_doh,
    stride_dkb, stride_dkt, stride_dkh,
    stride_dvb, stride_dvt, stride_dvh,
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
    pid_n = tl.program_id(0)
    bkv = tl.program_id(1)
    b = bkv // NUM_KV_HEADS
    kv_h = bkv % NUM_KV_HEADS
    group = NUM_HEADS // NUM_KV_HEADS

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    K_desc = tl.make_tensor_descriptor(
        K + b * stride_kb + kv_h * stride_kh,
        shape=[T_MAX, D], strides=[stride_kt, 1], block_shape=[BLOCK_N, D],
    )
    V_desc = tl.make_tensor_descriptor(
        V + b * stride_vb + kv_h * stride_vh,
        shape=[T_MAX, D], strides=[stride_vt, 1], block_shape=[BLOCK_N, D],
    )
    DK_desc = tl.make_tensor_descriptor(
        DK + b * stride_dkb + kv_h * stride_dkh,
        shape=[T_MAX, D], strides=[stride_dkt, 1], block_shape=[BLOCK_N, D],
    )
    DV_desc = tl.make_tensor_descriptor(
        DV + b * stride_dvb + kv_h * stride_dvh,
        shape=[T_MAX, D], strides=[stride_dvt, 1], block_shape=[BLOCK_N, D],
    )

    k = K_desc.load([pid_n * BLOCK_N, 0])
    v = V_desc.load([pid_n * BLOCK_N, 0])

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    qk_scale_log2 = SCALE * LOG2E

    if IS_CAUSAL:
        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
    else:
        m_start_block = 0
    m_end_block = tl.cdiv(T_MAX, BLOCK_M)

    for hg in range(group):
        h = kv_h * group + hg
        Q_desc = tl.make_tensor_descriptor(
            Q + b * stride_qb + h * stride_qh,
            shape=[T_MAX, D], strides=[stride_qt, 1], block_shape=[BLOCK_M, D],
        )
        O_desc = tl.make_tensor_descriptor(
            O + b * stride_ob + h * stride_oh,
            shape=[T_MAX, D], strides=[stride_ot, 1], block_shape=[BLOCK_M, D],
        )
        DO_desc = tl.make_tensor_descriptor(
            DO + b * stride_dob + h * stride_doh,
            shape=[T_MAX, D], strides=[stride_dot, 1], block_shape=[BLOCK_M, D],
        )
        for m_block in range(m_start_block, m_end_block):
            start_m = m_block * BLOCK_M
            offs_m_cur = start_m + offs_m
            row_mask = offs_m_cur < T_MAX

            q = Q_desc.load([start_m, 0])
            do = DO_desc.load([start_m, 0])
            o_tile = O_desc.load([start_m, 0])
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

    dk = dk * SCALE
    DK_desc.store([pid_n * BLOCK_N, 0], dk.to(K.dtype.element_ty))
    DV_desc.store([pid_n * BLOCK_N, 0], dv.to(V.dtype.element_ty))


# ---------------------------------------------------------------------------
# Fused backward kernel: computes dK, dV locally and atomically accumulates
# dQ into an fp32 scratch tensor. A postprocess kernel converts the fp32
# scratch to bf16 dQ. Goal: fold the _attn_bwd_dq_inline_delta_kernel work
# into the dkdv pass to eliminate the 2nd kernel launch + K/V reload traffic.
#
# Grid: (T/BN, B*NUM_KV_HEADS). Each program loads K/V once, then per Q-head
# in the group iterates M blocks: recomputes P, accumulates dK/dV in regs,
# computes ds @ K into dq_local and atomic-adds to DQ_F32. dq_local is in
# bf16 (matmul output), cast to fp32 for the atomic_add.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_bwd_fused_configs(),
    key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"],
    reset_to_zero=["DQ_F32"],
)
@triton.jit
def _attn_bwd_fused_kernel(
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

            # dq contribution from this KV block: ds @ K * SCALE -> BLOCK_M x BLOCK_D,
            # atomic-added into the fp32 scratch. The final SCALE is applied here so
            # each per-block contribution is already in the output's scale; the cast
            # kernel only downshifts fp32 -> bf16.
            dq_local = tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32) * SCALE
            dq_ptrs = (DQ_F32 + b * stride_dqfb + h * stride_dqfh
                       + offs_m_cur[:, None] * stride_dqft + offs_d[None, :] * stride_dqfd)
            tl.atomic_add(dq_ptrs, dq_local, mask=q_mask)

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
    DQ_F32, DQ,
    stride_fb, stride_ft, stride_fh, stride_fd,
    stride_qb, stride_qt, stride_qh, stride_qd,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NUM_HEADS
    h = bh % NUM_HEADS
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < T_MAX) & (offs_d[None, :] < D)
    f_ptrs = DQ_F32 + b * stride_fb + h * stride_fh + offs_m[:, None] * stride_ft + offs_d[None, :] * stride_fd
    q_ptrs = DQ + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    val = tl.load(f_ptrs, mask=mask, other=0.0)
    tl.store(q_ptrs, val.to(DQ.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Backward dK / dV kernel, split over Q heads (split-H)
#
# Grid: (T/BN, B*NUM_HEADS). Each program handles a single Q head and a single
# N block of KV. Partial dK/dV (one Q head's contribution, in fp32) is stored
# into a workspace of shape [B, NUM_HEADS, T, D] at the program's h slot. The
# wrapper sums the per-head partials over each KV group and casts back to the
# output dtype. This removes the inner group-loop from the non-split kernel
# and raises launched parallelism by NUM_HEADS/NUM_KV_HEADS.
# ---------------------------------------------------------------------------


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
    bh = tl.program_id(1)
    b = bh // NUM_HEADS
    h = bh % NUM_HEADS
    kv_h = h // (NUM_HEADS // NUM_KV_HEADS)

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

    for m_block in range(m_start_block, m_end_block):
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

    dk = dk * SCALE

    # Store partial (per Q head) into workspace of shape [B, H, T, D]
    # Uses the h axis of the workspace to avoid write collisions across programs
    # that share (b, kv_h, n); wrapper sums over each KV group afterwards.
    store_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
    dk_ptrs = DK_F32 + b * stride_dkb + h * stride_dkh + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    dv_ptrs = DV_F32 + b * stride_dvb + h * stride_dvh + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
    tl.store(dk_ptrs, dk, mask=store_mask)
    tl.store(dv_ptrs, dv, mask=store_mask)



# ---------------------------------------------------------------------------
# Backward dQ kernel
# ---------------------------------------------------------------------------


@triton.autotune(configs=_bwd_q_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_bwd_dq_kernel(
    Q, K, V, DO, DQ, LSE, DELTA,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_dob, stride_dot, stride_doh, stride_dod,
    stride_dqb, stride_dqt, stride_dqh, stride_dqd,
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
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NUM_HEADS
    h = bh % NUM_HEADS
    kv_h = h // (NUM_HEADS // NUM_KV_HEADS)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    row_mask = offs_m < T_MAX
    q_mask = row_mask[:, None] & (offs_d[None, :] < D)

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    do = tl.load(do_ptrs, mask=q_mask, other=0.0)

    lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m * stride_lt
    delta_ptrs = DELTA + b * stride_db + h * stride_dh + offs_m * stride_dt
    lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)
    delta = tl.load(delta_ptrs, mask=row_mask, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    qk_scale_log2 = SCALE * LOG2E

    if IS_CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, T_MAX)
    else:
        hi = T_MAX

    for start_n in range(0, hi, BLOCK_N):
        offs_n_cur = start_n + offs_n
        k_mask = (offs_n_cur[:, None] < T_MAX) & (offs_d[None, :] < D)
        k_ptrs = K + b * stride_kb + kv_h * stride_kh + offs_n_cur[:, None] * stride_kt + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + kv_h * stride_vh + offs_n_cur[:, None] * stride_vt + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
        p = tl.exp2(s - lse[:, None] * LOG2E)

        if IS_CAUSAL:
            p_mask = row_mask[:, None] & (offs_n_cur[None, :] < T_MAX) & (offs_m[:, None] >= offs_n_cur[None, :])
        else:
            p_mask = row_mask[:, None] & (offs_n_cur[None, :] < T_MAX)
        p = tl.where(p_mask, p, 0.0)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds.to(q.dtype), k, acc=dq, out_dtype=tl.float32)

    dq = dq * SCALE

    dq_ptrs = DQ + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqt + offs_d[None, :] * stride_dqd
    store_mask = row_mask[:, None] & (offs_d[None, :] < D)
    tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=store_mask)


# ---------------------------------------------------------------------------
# Backward dQ kernel with inline Δ (fused-delta variant, H1 ablation)
# Computes Δᵢ = <oᵢ, doᵢ> locally instead of reading it from a preprocessed
# [B, H, T] tensor. Saves 1 kernel launch + 1 HBM roundtrip on Δ.
# ---------------------------------------------------------------------------


@triton.autotune(configs=_bwd_q_configs(), key=["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"])
@triton.jit
def _attn_bwd_dq_inline_delta_kernel(
    Q, K, V, O, DO, DQ, LSE,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_dob, stride_dot, stride_doh, stride_dod,
    stride_dqb, stride_dqt, stride_dqh, stride_dqd,
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
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NUM_HEADS
    h = bh % NUM_HEADS
    kv_h = h // (NUM_HEADS // NUM_KV_HEADS)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    row_mask = offs_m < T_MAX
    q_mask = row_mask[:, None] & (offs_d[None, :] < D)

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    do = tl.load(do_ptrs, mask=q_mask, other=0.0)
    o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)
    # Δᵢ = <oᵢ, doᵢ> in fp32, then release o_tile.
    delta = tl.sum(o_tile.to(tl.float32) * do.to(tl.float32), axis=1)

    lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m * stride_lt
    lse = tl.load(lse_ptrs, mask=row_mask, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    qk_scale_log2 = SCALE * LOG2E

    if IS_CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, T_MAX)
    else:
        hi = T_MAX

    for start_n in range(0, hi, BLOCK_N):
        offs_n_cur = start_n + offs_n
        k_mask = (offs_n_cur[:, None] < T_MAX) & (offs_d[None, :] < D)
        k_ptrs = K + b * stride_kb + kv_h * stride_kh + offs_n_cur[:, None] * stride_kt + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + kv_h * stride_vh + offs_n_cur[:, None] * stride_vt + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale_log2
        p = tl.exp2(s - lse[:, None] * LOG2E)

        if IS_CAUSAL:
            p_mask = row_mask[:, None] & (offs_n_cur[None, :] < T_MAX) & (offs_m[:, None] >= offs_n_cur[None, :])
        else:
            p_mask = row_mask[:, None] & (offs_n_cur[None, :] < T_MAX)
        p = tl.where(p_mask, p, 0.0)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds.to(q.dtype), k, acc=dq, out_dtype=tl.float32)

    dq = dq * SCALE
    dq_ptrs = DQ + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqt + offs_d[None, :] * stride_dqd
    store_mask = row_mask[:, None] & (offs_d[None, :] < D)
    tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=store_mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _whale_attn_fwd_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q/k/v must be [B, T, H, D]"
    B, T, H, D = q.shape
    _, Tk, KV, Dk = k.shape
    assert k.shape == v.shape
    assert Tk == T and Dk == D
    assert H % KV == 0, f"num_heads ({H}) must be divisible by num_kv_heads ({KV})"

    o = torch.empty_like(q)
    lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)

    block_d = triton.next_power_of_2(D)
    scale = 1.0 / math.sqrt(D)

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
    else:
        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _attn_fwd_kernel[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
            BLOCK_D=block_d, D=D,
            IS_CAUSAL=causal,
        )
    return o, lse


def _whale_attn_bwd_impl(q, k, v, o, do, lse, causal
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, D = q.shape
    _, _, KV, _ = k.shape

    do_c = do if do.is_contiguous() else do.contiguous()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    block_d = triton.next_power_of_2(D)
    scale = 1.0 / math.sqrt(D)

    # Dispatch:
    #   auto             — fused_delta for short T, baseline for long T (default).
    #   baseline         — preprocess + separate dkdv/dq.
    #   fused_delta      — force inline-Δ in dkdv+dq; skips preprocess entirely.
    #   fused_delta_tma  — same as fused_delta but dkdv uses on-device TMA
    #                      (requires BLOCK_D == D; dq_inline stays as-is).
    #   fused_bwd        — monolithic dkdv+dq in one kernel; dq via fp32
    #                      atomic-add + bf16 cast kernel.
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
        use_fused_bwd = True
        use_fused_bwd_tma_dq = True
    else:
        use_fused_delta = variant == "fused_delta"
        use_tma_dkdv = False
        use_fused_bwd = False
        use_fused_bwd_tma_dq = False
    # TMA requires BLOCK_D == D, which holds iff D is a power of 2.
    if use_tma_dkdv and block_d != D:
        use_tma_dkdv = False
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
        cast_block_m = 128
        cast_grid = (triton.cdiv(T, cast_block_m), B * H)
        _attn_bwd_dq_cast_kernel[cast_grid](
            dq_f32, dq,
            dq_f32.stride(0), dq_f32.stride(1), dq_f32.stride(2), dq_f32.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            T_MAX=T, NUM_HEADS=H,
            BLOCK_M=cast_block_m, BLOCK_D=block_d, D=D,
            num_warps=4, num_stages=2,
        )
        return dq, dk, dv

    # Only the baseline path needs the Δ tensor + preprocess kernel.
    if not use_fused_delta:
        delta = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
        pre_block_m = 128
        pre_grid = (triton.cdiv(T, pre_block_m), B * H)
        _attn_bwd_preprocess_kernel[pre_grid](
            o, do_c, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            T_MAX=T, NUM_HEADS=H,
            BLOCK_M=pre_block_m, BLOCK_D=block_d, D=D,
            num_warps=4, num_stages=2,
        )

    if use_fused_delta:
        grid_kv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * KV)
        if use_tma_dkdv:
            _attn_bwd_dkdv_inline_delta_tma_kernel[grid_kv](
                q, k, v, o, do_c, dk, dv, lse,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                o.stride(0), o.stride(1), o.stride(2),
                do_c.stride(0), do_c.stride(1), do_c.stride(2),
                dk.stride(0), dk.stride(1), dk.stride(2),
                dv.stride(0), dv.stride(1), dv.stride(2),
                lse.stride(0), lse.stride(1), lse.stride(2),
                T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
                D=D,
                IS_CAUSAL=causal,
            )
        else:
            _attn_bwd_dkdv_inline_delta_kernel[grid_kv](
                q, k, v, o, do_c, dk, dv, lse,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
                BLOCK_D=block_d, D=D,
                IS_CAUSAL=causal,
            )
        grid_q = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _attn_bwd_dq_inline_delta_kernel[grid_q](
            q, k, v, o, do_c, dq, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
            BLOCK_D=block_d, D=D,
            IS_CAUSAL=causal,
        )
        return dq, dk, dv

    if use_split_h:
        group = H // KV
        # Per-Q-head partial workspace laid out as [B, T, H, D] to match q strides.
        dk_part = torch.empty((B, T, H, D), device=q.device, dtype=torch.float32)
        dv_part = torch.empty((B, T, H, D), device=q.device, dtype=torch.float32)
        grid_kv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * H)
        _attn_bwd_dkdv_split_h_kernel[grid_kv](
            q, k, v, do_c, dk_part, dv_part, lse, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
            dk_part.stride(0), dk_part.stride(1), dk_part.stride(2), dk_part.stride(3),
            dv_part.stride(0), dv_part.stride(1), dv_part.stride(2), dv_part.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
            BLOCK_D=block_d, D=D,
            IS_CAUSAL=causal,
        )
        # Reduce per-head partials over each KV group: [B, T, KV, group, D] -> [B, T, KV, D]
        dk.copy_(dk_part.view(B, T, KV, group, D).sum(dim=3).to(dk.dtype))
        dv.copy_(dv_part.view(B, T, KV, group, D).sum(dim=3).to(dv.dtype))
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

    grid_q = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
    _attn_bwd_dq_kernel[grid_q](
        q, k, v, do_c, dq, lse, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do_c.stride(0), do_c.stride(1), do_c.stride(2), do_c.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        T_MAX=T, NUM_HEADS=H, NUM_KV_HEADS=KV, SCALE=scale,
        BLOCK_D=block_d, D=D,
        IS_CAUSAL=causal,
    )

    return dq, dk, dv


# ---------------------------------------------------------------------------
# torch.library custom ops
# ---------------------------------------------------------------------------


@torch.library.custom_op("whale::attn_fwd", mutates_args=())
def whale_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    return _whale_attn_fwd_impl(q, k, v, causal)


@whale_attn_fwd.register_fake
def _(q, k, v, causal):
    B, T, H, _ = q.shape
    return torch.empty_like(q), torch.empty(B, H, T, device=q.device, dtype=torch.float32)


@torch.library.custom_op("whale::attn_bwd", mutates_args=())
def whale_attn_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor,
                   causal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _whale_attn_bwd_impl(q, k, v, o, do, lse, causal)


@whale_attn_bwd.register_fake
def _(q, k, v, o, do, lse, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _fwd_setup_context(ctx, inputs, output):
    q, k, v, causal = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.causal = causal


def _fwd_backward(ctx, grad_o, grad_lse):
    q, k, v, o, lse = ctx.saved_tensors
    dq, dk, dv = whale_attn_bwd(q, k, v, o, grad_o.contiguous(), lse, ctx.causal)
    return dq, dk, dv, None


whale_attn_fwd.register_autograd(_fwd_backward, setup_context=_fwd_setup_context)


def custom_whale_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          causal: bool = True) -> torch.Tensor:
    """Training-facing wrapper that returns only the output tensor."""
    o, _lse = whale_attn_fwd(q, k, v, causal)
    return o


# ---------------------------------------------------------------------------
# Fast path: plain torch.autograd.Function, bypasses custom_op dispatch.
# Use this in eager mode for lower latency. The custom_op path above is
# still the right choice under torch.compile (it keeps each launch opaque).
# ---------------------------------------------------------------------------


class _WhaleAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        o, lse = _whale_attn_fwd_impl(q, k, v, causal)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _whale_attn_bwd_impl(q, k, v, o, grad_o, lse, ctx.causal)
        return dq, dk, dv, None


def whale_attn_fast(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = True) -> torch.Tensor:
    """Eager-mode fast path via torch.autograd.Function."""
    return _WhaleAttnFn.apply(q, k, v, causal)


# ---------------------------------------------------------------------------
# Hybrid: whale forward + FA3 backward. whale wins the forward on H100 for
# short/medium sequences; FA3's backward is faster than what Triton 3.6 can
# emit. This autograd.Function stitches them together, giving us the fwd win
# while delegating the (harder) backward to the CUDA-native kernel.
# FA3 must be installed (flash_attn_interface importable).
# ---------------------------------------------------------------------------


class _WhaleFwdFA3BwdFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        o, lse = _whale_attn_fwd_impl(q, k, v, causal)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        return o

    @staticmethod
    def backward(ctx, grad_o):
        from flash_attn_interface import _flash_attn_backward
        q, k, v, o, lse = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _flash_attn_backward(
            grad_o, q, k, v, o, lse,
            None, None,  # cu_seqlens_q, cu_seqlens_k
            None, None,  # seqused_q, seqused_k
            None, None,  # max_seqlen_q, max_seqlen_k
            dq, dk, dv,
            ctx.softmax_scale,
            ctx.causal,
            -1, -1,  # window_size_left, window_size_right
            0.0,     # softcap
            False,   # deterministic
            0,       # sm_margin
        )
        return dq, dk, dv, None


def whale_fwd_fa3_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      causal: bool = True) -> torch.Tensor:
    """Hybrid: whale Triton forward + FA3 CUDA backward."""
    return _WhaleFwdFA3BwdFn.apply(q, k, v, causal)
