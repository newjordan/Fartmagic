"""Sweep fwd kernel with num_ctas (cluster mode) and maxnreg overrides.
Runs in a single python process by patching _attn_fwd_kernel.configs each trial."""
import os, sys, torch, triton
import subprocess


def measure(ctas: int, maxnreg, bm=64, bn=64, warps=4, stages=3, kernel_name="_attn_fwd_kernel",
            variant="default", steps=30, shape=(4, 2048, 8, 4, 64)) -> str:
    code = f"""
import os, torch, triton
os.environ['WHALE_FWD_VARIANT']='{variant}'
maxnreg_val = {None if maxnreg is None else maxnreg}
B,T,H,KV,D={shape}
from vault.whale_kernel_triton import _attn_fwd_kernel, _attn_fwd_tma_kernel, whale_attn_fast
cfg = triton.Config({{'BLOCK_M': {bm}, 'BLOCK_N': {bn}}}, num_warps={warps},
                     num_stages={stages}, num_ctas={ctas}, maxnreg=maxnreg_val)
_attn_fwd_kernel.configs = [cfg]
# for TMA kernel, BLOCK_D is not needed (it uses D directly)
_attn_fwd_tma_kernel.configs = [cfg]
try:
    q=torch.randn(B,T,H,D,dtype=torch.bfloat16,device='cuda')
    k=torch.randn(B,T,KV,D,dtype=torch.bfloat16,device='cuda')
    v=torch.randn(B,T,KV,D,dtype=torch.bfloat16,device='cuda')
    for _ in range(10):
        with torch.no_grad():
            _ = whale_attn_fast(q,k,v,True)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range({steps}):
            with torch.no_grad():
                _ = whale_attn_fast(q,k,v,True)
        torch.cuda.synchronize()
    t=0.0
    for e in prof.events():
        if e.name=='{kernel_name}':
            t+=getattr(e,'self_device_time_total',0) or 0
    print(f'RESULT={{t/{steps}:.3f}}')
except Exception as exc:
    s = str(exc)
    s = s.replace(chr(10), ' ').replace(chr(13), ' ')[:60]
    print(f'RESULT=ERR:{{type(exc).__name__}}:{{s}}')
"""
    r = subprocess.run(
        ["ssh", "vast-whale",
         f"cd /workspace/SOTA_FINAL && PYTHONPATH=. /venv/main/bin/python3 -c {code!r} 2>/dev/null"],
        capture_output=True, text=True, timeout=180,
    )
    out = r.stdout or ""
    for line in out.splitlines():
        if line.startswith("RESULT="):
            return line[7:]
    return "NO_RESULT"


def main() -> None:
    print("== non-TMA fwd ==")
    print("ctas | maxnreg | us")
    for ctas in (1, 2):
        for maxnreg in (None, 128, 160, 192, 240):
            r = measure(ctas, maxnreg, variant="default")
            print(f"  {ctas}  |  {maxnreg!s:5s}  |  {r}")

    print("\n== TMA fwd ==")
    print("ctas | maxnreg | us")
    for ctas in (1, 2):
        for maxnreg in (None, 128, 160, 192, 240):
            r = measure(ctas, maxnreg, variant="tma", kernel_name="_attn_fwd_tma_kernel")
            print(f"  {ctas}  |  {maxnreg!s:5s}  |  {r}")


if __name__ == "__main__":
    main()
