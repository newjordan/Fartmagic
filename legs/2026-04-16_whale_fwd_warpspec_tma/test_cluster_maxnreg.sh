#!/bin/bash
# Test cluster mode (num_ctas=2) and maxnreg tweaks for the fwd kernel.
# num_ctas and maxnreg are both supported in triton 3.6 via Config, but NOT
# via our current WHALE_FWD_CONFIG env var, so we build a tiny inline script.
set -u
cd /workspace/SOTA_FINAL || exit 1

SHAPE="${SHAPE:-4,2048,8,4,64}"
STEPS="${STEPS:-25}"

for CTAS in 1 2 4; do
  for NREG in none 128 160 192 240; do
    RES=$(CTAS="$CTAS" NREG="$NREG" SHAPE="$SHAPE" STEPS="$STEPS" PYTHONPATH=. /venv/main/bin/python3 << 'PY'
import os, torch, triton, triton.language as tl
CTAS=int(os.environ["CTAS"])
NREG=os.environ["NREG"]
maxnreg = None if NREG == "none" else int(NREG)
B,T,H,KV,D=[int(x) for x in os.environ["SHAPE"].split(",")]
STEPS=int(os.environ["STEPS"])

# Reuse the non-TMA fwd kernel with a forced config that overrides num_ctas / maxnreg
from vault.whale_kernel_triton import _attn_fwd_kernel, whale_attn_fast
import vault.whale_kernel_triton as wkt

# Replace the autotune configs with one hand-built config
single_cfg = [triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3,
                             num_ctas=CTAS, maxnreg=maxnreg)]
_attn_fwd_kernel.configs = single_cfg

q=torch.randn(B,T,H,D,dtype=torch.bfloat16,device="cuda")
k=torch.randn(B,T,KV,D,dtype=torch.bfloat16,device="cuda")
v=torch.randn(B,T,KV,D,dtype=torch.bfloat16,device="cuda")
try:
    for _ in range(12):
        with torch.no_grad():
            _ = whale_attn_fast(q,k,v,True)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(STEPS):
            with torch.no_grad():
                _ = whale_attn_fast(q,k,v,True)
        torch.cuda.synchronize()
    t=0.0
    for e in prof.events():
        if e.name=="_attn_fwd_kernel":
            t+=getattr(e,"self_device_time_total",0) or 0
    print(f"{t/STEPS:.3f}")
except Exception as exc:
    print(f"ERR:{type(exc).__name__}:{str(exc)[:60]}")
PY
2>/dev/null | tail -1)
    printf "  ctas=%-2s maxnreg=%-5s -> %s us\n" "$CTAS" "$NREG" "$RES"
  done
done
