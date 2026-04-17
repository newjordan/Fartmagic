# Hypothesis — 2026-04-16_whale_fast_kernels

Parent: `legs/2026-04-16_whale_pr1493_triton_kernel`
Kernel parent: `vault/whale_kernel_triton.py` (untracked; treated as editable for this leg)

## One Variable
- Name: `whale attention kernels — fused forward + Triton backward`
- Change: Replace the SDPA-delegating backward with a Flash-Attention-style Triton backward kernel,
  fix a correctness issue in the forward (diagonal block clobbered the off-diagonal running state),
  and autotune both passes. Forward must save `(M, L)` so backward does not redo the forward.

## Why
- Today's forward uses a bespoke Triton kernel but the backward reconstructs `sdpa(q, k, v)` inside
  `torch.func.vjp`. That means training pays ~1 SDPA forward + 1 SDPA backward per step on top of the
  custom forward. A fused Triton backward avoids the duplicate forward and removes the SDPA
  dependency entirely.
- The diagonal-block accumulators in `_dense_causal_fwd_kernel` overwrite `m_i`, `l_i`, `acc` instead
  of combining with the pre-diagonal running values. That is only harmless when `q_block_id == 0`.
  We must verify numerics before trusting any speed number.

## Pass Criteria
1. `|whale_out - sdpa_out|_max < 1e-2` in bf16 across (D=32, 64, 128) × (GQA 1:1, 2:1, 4:1) × causal.
2. `|d{Q,K,V} - sdpa_ref.grad|_max < 5e-2` (bf16) on same matrix.
3. No Dynamo graph break with `torch.compile(mode="reduce-overhead")` on fwd+bwd.
4. Forward + backward wall time (per token) lower than the current kernel with SDPA backward on
   `B=8, T=2048, H=8, KV=4, D=64, bf16` — number recorded in `RESULTS.md`.

## Evidence Plan
- `evidence/baseline_numerics.json`  — current whale fwd vs SDPA.
- `evidence/baseline_speed.json`     — current whale fwd-only + fwd+bwd.
- `evidence/new_numerics.json`       — fwd + dQ/dK/dV vs SDPA autograd.
- `evidence/new_speed.json`          — new fwd-only + fwd+bwd.
- `RESULTS.md` must cite exact JSON paths and raw numbers.
