#!/usr/bin/env bash
# whale_fused_bwd_tma_dq — H8 driver.
# (1) Numerics check: TMA-dq fused_bwd dQ/dK/dV vs fused_delta reference.
# (2) Headline + long-T bench: TMA-dq vs fused_delta vs fused_bwd vs FA3 vs SDPA.
#
# Requires that vault/whale_kernel_triton.py has been patched per
# vault_patch.md so that WHALE_BWD_VARIANT=fused_bwd_tma_dq routes to the
# new _attn_bwd_fused_tma_dq_kernel. Until then this script will fall
# through to the existing dispatch.

set -euo pipefail

LEG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LEG_DIR}/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"
source "${LEG_DIR}/tracked_env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PY=/venv/main/bin/python3

LOG="${LEG_DIR}/logs/run_${TS}.log"
mkdir -p "${LEG_DIR}/logs" "${LEG_DIR}/evidence"
echo "== ${TS}  WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT} (Lever B / TMA dQ)" | tee "${LOG}"

# -----------------------------------------------------------------------
# (1) Numerics — diff TMA-dq fused_bwd vs fused_delta on a small shape.
#     dQ/dK/dV must match within 1e-2 max abs (bf16 bwd tolerance).
# -----------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "-- numerics: fused_bwd_tma_dq vs fused_delta on (2,1024,4,4,64)" | tee -a "${LOG}"
${PY} - <<'PY' 2>&1 | tee -a "${LOG}"
import os, torch
from vault.whale_kernel_triton import whale_attn_fast

torch.manual_seed(0)
B,T,H,KV,D = 2,1024,4,4,64
def mk():
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn((B,T,H,D), generator=g, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((B,T,KV,D), generator=g, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((B,T,KV,D), generator=g, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return q,k,v

def grads(variant):
    os.environ["WHALE_BWD_VARIANT"] = variant
    q,k,v = mk()
    out = whale_attn_fast(q,k,v, causal=True)
    g = torch.randn_like(out)
    out.backward(g)
    return q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone()

dq_ref, dk_ref, dv_ref = grads("fused_delta")
dq_new, dk_new, dv_new = grads("fused_bwd_tma_dq")
for name, a, b in (("dQ", dq_ref, dq_new), ("dK", dk_ref, dk_new), ("dV", dv_ref, dv_new)):
    e = (a.float()-b.float()).abs().max().item()
    print(f"  {name}: max_abs_err={e:.3e}  ref_absmax={a.float().abs().max().item():.3e}")
PY

# -----------------------------------------------------------------------
# (2) Bench — three shapes, four whale variants + fa3 + sdpa.
#     Headline: 4,2048,8,4,64. Long-T: 4,4096,8,4,64 and 2,8192,8,4,64.
# -----------------------------------------------------------------------
SHAPES="4,2048,8,4,64;4,4096,8,4,64;2,8192,8,4,64"

for VARIANT in fused_bwd_tma_dq fused_bwd fused_delta auto; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    echo "" | tee -a "${LOG}"
    echo "-- bench WHALE_BWD_VARIANT=${VARIANT}" | tee -a "${LOG}"
    WHALE_BWD_VARIANT=${VARIANT} ${PY} "${LEG_DIR}/bench_stable.py" \
        --shapes "${SHAPES}" \
        --backends "whale_fast,fa3,sdpa" \
        --rounds  "${WHALE_BENCH_ROUNDS}" \
        --iters   "${WHALE_BENCH_ITERS}" \
        --label   "tma_dq_${VARIANT}" \
        --out     "${LEG_DIR}/evidence/tma_dq_${VARIANT}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

echo "" | tee -a "${LOG}"
echo "== done. log: ${LOG}"
