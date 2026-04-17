import math

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 128
HEAD_DIM = 128

@triton.jit
def _dense_causal_fwd_kernel(
    Q, K, V, O,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_Q_BLOCKS = tl.cdiv(T_MAX, BS)

    bh_id = pid // NUM_Q_BLOCKS
    q_block_id = pid % NUM_Q_BLOCKS

    b_id = bh_id // NUM_HEADS
    h_id = bh_id % NUM_HEADS
    kv_h_id = h_id // (NUM_HEADS // NUM_KV_HEADS)

    q_start = q_block_id * BS
    offs_tok = q_start + tl.arange(0, BS)
    offs_d = tl.arange(0, BLOCK_D)

    # Base pointers for the current batch and head
    Q_ptr = Q + b_id * stride_qb + h_id * stride_qh
    K_ptr = K + b_id * stride_kb + kv_h_id * stride_kh
    V_ptr = V + b_id * stride_vb + kv_h_id * stride_vh

    q_mask = (offs_tok[:, None] < T_MAX) & (offs_d[None, :] < D)
    q_bf16 = tl.load(
        Q_ptr + offs_tok[:, None] * stride_qt + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0
    )

    m_i = tl.full([BS], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BS], dtype=tl.float32)
    acc = tl.zeros([BS, BLOCK_D], dtype=tl.float32)

    offs_d_arange = tl.arange(0, BS)

    LOG2E: tl.constexpr = 1.4426950408889634
    SCALE_2: tl.constexpr = SCALE * LOG2E

    for k_block_id in range(0, q_block_id):
        k_start = k_block_id * BS
        offs_k = k_start + offs_d_arange

        k_mask = offs_d[None, :] < D

        k_bf16 = tl.load(K_ptr + offs_k[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=k_mask, other=0.0)
        v_bf16 = tl.load(V_ptr + offs_k[:, None] * stride_vt + offs_d[None, :] * stride_vd, mask=k_mask, other=0.0)

        s = tl.dot(q_bf16, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE_2

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.bfloat16), v_bf16, acc=acc * alpha[:, None], out_dtype=tl.float32)
        m_i = m_new

    # Diagonal block
    k_start_d = q_block_id * BS
    offs_k_d = k_start_d + offs_d_arange
    k_mask_d = (offs_k_d[:, None] < T_MAX) & (offs_d[None, :] < D)

    k_bf16_d = tl.load(
        K_ptr + offs_k_d[:, None] * stride_kt + offs_d[None, :] * stride_kd,
        mask=k_mask_d,
        other=0.0
    )
    v_bf16_d = tl.load(
        V_ptr + offs_k_d[:, None] * stride_vt + offs_d[None, :] * stride_vd,
        mask=k_mask_d,
        other=0.0
    )

    s_d = tl.dot(q_bf16, tl.trans(k_bf16_d), out_dtype=tl.float32) * SCALE_2
    causal = offs_tok[:, None] >= offs_k_d[None, :]
    s_d = tl.where(causal & (offs_k_d[None, :] < T_MAX), s_d, float("-inf"))

    m_i = tl.max(s_d, axis=1)
    p_d = tl.exp2(s_d - m_i[:, None])
    l_i = tl.sum(p_d, axis=1)

    p_hi = p_d.to(tl.bfloat16)
    p_lo = ((p_d - p_hi.to(tl.float32)) * 128.0).to(tl.bfloat16)
    acc = tl.dot(p_hi, v_bf16_d, out_dtype=tl.float32)
    acc_lo = tl.dot(p_lo, v_bf16_d, out_dtype=tl.float32)
    acc = acc + acc_lo * (1.0 / 128.0)

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out_mask = (offs_tok[:, None] < T_MAX) & (offs_d[None, :] < D)
    out = tl.where(out_mask, acc / l_safe[:, None], 0.0)

    O_ptr = O + b_id * stride_ob + h_id * stride_oh
    tl.store(
        O_ptr + offs_tok[:, None] * stride_ot + offs_d[None, :] * stride_od,
        out.to(tl.bfloat16),
        mask=out_mask
    )

def _custom_whale_attn_fwd_impl(q, k, v, causal=True):
    B, T, H, D = q.shape
    _, _, H_kv, _ = k.shape

    o = torch.empty_like(q)

    # Grid: one program per (batch * head * q_block)
    num_q_blocks = triton.cdiv(T, BLOCK_SIZE)
    grid = (num_q_blocks * B * H,)

    scale = 1.0 / math.sqrt(D)

    block_d = triton.next_power_of_2(D)

    _dense_causal_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        T_MAX=T,
        NUM_HEADS=H,
        NUM_KV_HEADS=H_kv,
        SCALE=scale,
        BS=BLOCK_SIZE,
        D=D,
        BLOCK_D=block_d,
        num_stages=3,
        num_warps=8,
    )

    return o

@torch.library.custom_op("whale::attn", mutates_args=())
def whale_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    return _custom_whale_attn_fwd_impl(q, k, v, causal)

@whale_attn.register_fake
def _(q, k, v, causal):
    return torch.empty_like(q)

def setup_context(ctx, inputs, output):
    q, k, v, causal = inputs
    ctx.save_for_backward(q, k, v)
    ctx.causal = causal

def backward(ctx, do):
    q, k, v = ctx.saved_tensors
    causal = ctx.causal

    def sdpa(q_, k_, v_):
        q2 = q_.transpose(1, 2)
        k2 = k_.transpose(1, 2)
        v2 = v_.transpose(1, 2)
        if k2.size(1) != q2.size(1):
            rep = q2.size(1) // k2.size(1)
            k2 = k2.repeat_interleave(rep, dim=1)
            v2 = v2.repeat_interleave(rep, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
        return out.transpose(1, 2)

    # Use torch.func.vjp to compute gradients without .backward()
    _, vjp_fn = torch.func.vjp(sdpa, q, k, v)
    dq, dk, dv = vjp_fn(do)
    return dq, dk, dv, None

whale_attn.register_autograd(backward, setup_context=setup_context)

def custom_whale_attn_fwd(q, k, v, causal=True):
    return whale_attn(q, k, v, causal)

