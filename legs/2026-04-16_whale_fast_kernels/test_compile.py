"""Smoke-test that torch.compile can trace fwd+bwd of the whale kernel without
graph breaks. Runs a tiny toy module, compares compiled vs eager on gradients,
and prints timing for both."""
import time

import torch
import torch.nn as nn

from vault.whale_kernel_triton import custom_whale_attn_fwd


class MiniAttn(nn.Module):
    def __init__(self, H=8, KV=4, D=64):
        super().__init__()
        self.H, self.KV, self.D = H, KV, D
        model_dim = H * D
        kv_dim = KV * D
        self.wq = nn.Linear(model_dim, model_dim, bias=False)
        self.wk = nn.Linear(model_dim, kv_dim, bias=False)
        self.wv = nn.Linear(model_dim, kv_dim, bias=False)
        self.wo = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, self.H, self.D)
        k = self.wk(x).reshape(B, T, self.KV, self.D)
        v = self.wv(x).reshape(B, T, self.KV, self.D)
        y = custom_whale_attn_fwd(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
        y = y.reshape(B, T, self.H * self.D)
        return self.wo(y)


def main():
    torch.manual_seed(1337)
    dev = "cuda"
    B, T = 4, 1024
    m = MiniAttn().to(device=dev, dtype=torch.bfloat16)
    x = torch.randn(B, T, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)

    out_eager = m(x)
    loss_eager = out_eager.float().pow(2).mean()
    loss_eager.backward()
    grad_eager = x.grad.detach().clone()
    x.grad = None

    mc = torch.compile(m, mode="reduce-overhead", dynamic=False)
    out_compiled = mc(x)
    loss_compiled = out_compiled.float().pow(2).mean()
    loss_compiled.backward()
    grad_compiled = x.grad.detach().clone()

    err_out = (out_eager.float() - out_compiled.float()).abs().max().item()
    err_grad = (grad_eager.float() - grad_compiled.float()).abs().max().item()
    print(f"eager vs compiled  out max_abs_err = {err_out:.6f}")
    print(f"eager vs compiled  dX  max_abs_err = {err_grad:.6f}")

    for _ in range(5):
        x.grad = None
        out = mc(x)
        out.float().pow(2).mean().backward()
    torch.cuda.synchronize()

    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        x.grad = None
        out = mc(x)
        out.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000 / n
    print(f"compiled fwd+bwd mean_ms = {dt:.3f}")


if __name__ == "__main__":
    main()
