"""Minimal smoke test for on-device TMA via tl.make_tensor_descriptor on
Triton 3.6 + cu130. Run on the pod:
    PYTHONPATH=. /venv/main/bin/python3 legs/2026-04-16_whale_bwd_tma/tma_smoke.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def tma_copy(A, B, N: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    A_desc = tl.make_tensor_descriptor(A, [N, D], [D, 1], [BLOCK_N, D])
    B_desc = tl.make_tensor_descriptor(B, [N, D], [D, 1], [BLOCK_N, D])
    offs = pid * BLOCK_N
    x = A_desc.load([offs, 0])
    B_desc.store([offs, 0], x)


def _alloc(size: int, align: int, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def main():
    triton.set_allocator(_alloc)
    N, D = 1024, 64
    BLOCK_N = 64
    A = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")
    B = torch.empty_like(A)
    tma_copy[(N // BLOCK_N,)](A, B, N, D, BLOCK_N)
    torch.cuda.synchronize()
    err = (A.float() - B.float()).abs().max().item()
    print(f"TMA copy err: {err}")
    assert err == 0.0, "TMA copy mismatch"
    print("OK: on-device TMA works on this stack")


if __name__ == "__main__":
    main()
