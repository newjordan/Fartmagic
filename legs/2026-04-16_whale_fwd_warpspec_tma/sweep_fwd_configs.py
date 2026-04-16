"""Sweep (BLOCK_M, BLOCK_N, warps, stages) for the whale forward kernel at the
headline shape, measuring GPU kernel time via kineto. Prints the top 10."""
import argparse, os, subprocess, sys, json, tempfile
import torch


def measure_one(bm: int, bn: int, w: int, s: int, shape: str, steps: int = 30) -> float | None:
    code = f"""
import os, torch
os.environ['WHALE_FWD_CONFIG']='{bm},{bn},{w},{s}'
B,T,H,KV,D=[int(x) for x in '{shape}'.split(',')]
from vault.whale_kernel_triton import whale_attn_fast
q=torch.randn(B,T,H,D,dtype=torch.bfloat16,device='cuda')
k=torch.randn(B,T,KV,D,dtype=torch.bfloat16,device='cuda')
v=torch.randn(B,T,KV,D,dtype=torch.bfloat16,device='cuda')
try:
    for _ in range(10):
        with torch.no_grad():
            _ = whale_attn_fast(q,k,v,True)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range({steps}):
            with torch.no_grad():
                _ = whale_attn_fast(q,k,v,True)
        torch.cuda.synchronize()
    t=0
    for e in prof.events():
        if e.name=='_attn_fwd_kernel':
            t+=getattr(e,'self_device_time_total',0) or 0
    print(f'US_PER_ITER={{t/{steps}:.3f}}')
except Exception as exc:
    print(f'US_PER_ITER=ERR:{{type(exc).__name__}}:{{exc!s:.60}}')
"""
    r = subprocess.run(
        ["ssh", "vast-whale",
         f"cd /workspace/SOTA_FINAL && PYTHONPATH=. /venv/main/bin/python3 -c {code!r}"],
        capture_output=True, text=True, timeout=180,
    )
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if line.startswith("US_PER_ITER="):
            val = line.split("=", 1)[1]
            if val.startswith("ERR"):
                return None
            return float(val)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="4,2048,8,4,64")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    configs = []
    for bm in (64, 128, 256):
        for bn in (32, 64, 128, 256):
            for w in (4, 8):
                for s in (2, 3, 4, 5):
                    configs.append((bm, bn, w, s))

    results = []
    print(f"Sweeping {len(configs)} configs for shape={args.shape}")
    for i, (bm, bn, w, s) in enumerate(configs):
        t = measure_one(bm, bn, w, s, args.shape, args.steps)
        tag = f"{bm}x{bn} w={w} s={s}"
        if t is None:
            print(f"  [{i:3d}/{len(configs)}] {tag:18s} -> FAILED")
            continue
        print(f"  [{i:3d}/{len(configs)}] {tag:18s} -> {t:7.2f} us")
        results.append((t, bm, bn, w, s))

    results.sort()
    print("\n=== Top 10 ===")
    for t, bm, bn, w, s in results[:10]:
        print(f"  {t:7.2f} us  BM={bm} BN={bn} warps={w} stages={s}")

    if results:
        with open("legs/2026-04-16_whale_fwd_warpspec_tma/evidence/sweep_results.json", "w") as f:
            json.dump([{"us": t, "BM": bm, "BN": bn, "warps": w, "stages": s}
                       for t, bm, bn, w, s in results], f, indent=2)


if __name__ == "__main__":
    main()
