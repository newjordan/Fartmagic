"""Toggle the early-exit split in _attn_bwd_dkdv_kernel between the split
form (current vault, masked + unmasked phases) and the unsplit form
(single M-loop with causal tl.where on every iter) for precision a/b
benching. Operates only inside vault/whale_kernel_triton.py lines ~505-550.

Usage:
  python3 swap_ee.py split    # ensure the split (early-exit) form
  python3 swap_ee.py unsplit  # revert to the single M-loop form
  python3 swap_ee.py status   # print which form is live

Idempotent. Refuses to run if neither form is detected (to avoid
corrupting unrelated edits).
"""
from __future__ import annotations
import sys, pathlib, re

VAULT = pathlib.Path(__file__).resolve().parents[2] / "vault" / "whale_kernel_triton.py"

# The `split` block is the current vault form lines ~505-575. The `unsplit`
# block replaces the two-phase loop with a single `range(m_start_block, m_end_block)`
# that carries the causal tl.where in every iter.
MARKER_SPLIT = "m_masking_max = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)"
MARKER_UNSPLIT_SENTINEL = "# ee_unsplit_sentinel_attn_bwd_dkdv_kernel"

SPLIT_BLOCK = """    if IS_CAUSAL:
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
"""

UNSPLIT_BLOCK = """    # ee_unsplit_sentinel_attn_bwd_dkdv_kernel
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
"""


def current_form(src: str) -> str:
    split_present = SPLIT_BLOCK in src
    unsplit_present = UNSPLIT_BLOCK in src
    if split_present and not unsplit_present:
        return "split"
    if unsplit_present and not split_present:
        return "unsplit"
    return "unknown"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    src = VAULT.read_text()
    form = current_form(src)
    if mode == "status":
        print(f"form={form}")
        return 0
    if form == "unknown":
        print(f"[swap_ee] ERROR: vault has neither canonical split nor unsplit block; aborting. form={form}")
        return 2
    if mode == form:
        print(f"[swap_ee] already {form}; noop")
        return 0
    if mode == "split":
        new = src.replace(UNSPLIT_BLOCK, SPLIT_BLOCK, 1)
    elif mode == "unsplit":
        new = src.replace(SPLIT_BLOCK, UNSPLIT_BLOCK, 1)
    else:
        print(f"[swap_ee] unknown mode={mode}")
        return 2
    if new == src:
        print(f"[swap_ee] ERROR: replacement changed nothing; refusing to write")
        return 3
    VAULT.write_text(new)
    print(f"[swap_ee] set vault to {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
