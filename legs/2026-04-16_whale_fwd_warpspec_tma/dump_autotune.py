"""Dump the winning autotune config for _attn_fwd_kernel at the headline shape."""
import os, torch

os.environ.setdefault("WHALE_BWD_VARIANT", "fused_delta")

from vault.whale_kernel_triton import whale_attn_fast, _attn_fwd_kernel

B, T, H, KV, D = 4, 2048, 8, 4, 64
q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda")

# Trigger autotune on forward only (no grad)
for _ in range(3):
    with torch.no_grad():
        _ = whale_attn_fast(q, k, v, True)
torch.cuda.synchronize()

# Inspect the autotuner cache
at = _attn_fwd_kernel
print(f"autotune cache size: {len(at.cache)}")
for key, cfg in at.cache.items():
    print(f"  key={key}")
    print(f"  config={cfg}")
print()

# Also for D=128
q128 = torch.randn(B, T, H, 128, dtype=torch.bfloat16, device="cuda")
k128 = torch.randn(B, T, KV, 128, dtype=torch.bfloat16, device="cuda")
v128 = torch.randn(B, T, KV, 128, dtype=torch.bfloat16, device="cuda")
for _ in range(3):
    with torch.no_grad():
        _ = whale_attn_fast(q128, k128, v128, True)
torch.cuda.synchronize()

print("=== After D=128 warmup ===")
print(f"autotune cache size: {len(at.cache)}")
for key, cfg in at.cache.items():
    print(f"  key={key}")
    print(f"  config={cfg}")
