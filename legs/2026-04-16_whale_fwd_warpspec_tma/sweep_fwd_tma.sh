#!/bin/bash
# Sweep (BM, BN, warps, stages) for the TMA forward kernel at the headline shape.
set -u
cd /workspace/SOTA_FINAL || exit 1

SHAPE="${SHAPE:-4,2048,8,4,64}"
STEPS="${STEPS:-30}"
OUT="${OUT:-/tmp/fwd_tma_sweep.txt}"
: > "$OUT"

for bm in 64 128; do
  for bn in 32 64 128 256; do
    for w in 4 8; do
      for s in 2 3 4; do
        RES=$(WHALE_FWD_TMA_CONFIG="$bm,$bn,$w,$s" WHALE_FWD_VARIANT=tma PYTHONPATH=. /venv/main/bin/python3 -c "
import os, torch
B,T,H,KV,D=[int(x) for x in '$SHAPE'.split(',')]
try:
    from vault.whale_kernel_triton import whale_attn_fast
    q=torch.randn(B,T,H,D,dtype=torch.bfloat16,device='cuda')
    k=torch.randn(B,T,KV,D,dtype=torch.bfloat16,device='cuda')
    v=torch.randn(B,T,KV,D,dtype=torch.bfloat16,device='cuda')
    for _ in range(12):
        with torch.no_grad():
            _ = whale_attn_fast(q,k,v,True)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range($STEPS):
            with torch.no_grad():
                _ = whale_attn_fast(q,k,v,True)
        torch.cuda.synchronize()
    t=0.0
    for e in prof.events():
        if e.name=='_attn_fwd_tma_kernel':
            t+=getattr(e,'self_device_time_total',0) or 0
    print(f'{t/$STEPS:.3f}')
except Exception as exc:
    print(f'ERR:{type(exc).__name__}')
" 2>/dev/null | tail -1)
        printf "  BM=%-3s BN=%-3s w=%-2s s=%-2s -> %s us\n" "$bm" "$bn" "$w" "$s" "$RES" | tee -a "$OUT"
      done
    done
  done
done

echo
echo "=== Top 10 (sorted, filtering errors) ==="
grep -v ERR "$OUT" | awk '{for(i=1;i<=NF;i++) if($i=="->") {print $(i+1), $0; break}}' | sort -n | head -10
