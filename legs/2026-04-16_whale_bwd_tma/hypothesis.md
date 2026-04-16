# whale_bwd_tma — hypothesis

## Goal

Beat FA3's backward at the headline shape (B=4, T=2048, H=8, KV=4, D=64,
causal, bf16) with pure Triton. Current gap: 275 µs (whale) vs 260 µs (FA3)
= 15 µs behind = +6%.

If successful, pure-Triton whale (fwd+bwd) beats hybrid (Triton fwd + FA3 bwd),
and we drop the FA3 dependency for this shape class while still being faster.

## Hypothesis

`_attn_bwd_dkdv_inline_delta_kernel` is the hotspot at 194 µs (dq_inline is
77 µs). Rewriting its HBM loads/stores through `tl.make_tensor_descriptor`
(on-device TMA on Hopper) should:

1. Replace address-computed loads of Q / O / DO / K / V with async bulk
   copies that hide HBM latency behind MMA compute.
2. Produce swizzled SMEM layouts that feed `wgmma` / `ldmatrix` cleanly,
   reducing register pressure spent on the pointer arithmetic tiles.
3. Let the Triton scheduler pipeline more aggressively (TMA is the kind
   of producer Triton 3.6 explicitly designs around on Hopper).

Target: cut the dkdv_inline kernel by ≥8% (≥16 µs), which by itself would
flip the backward from +6% to ~−0% vs FA3. A 15–20% cut would be a clear win.

## Test conditions

- Variant `WHALE_BWD_VARIANT=fused_delta_tma` — forces the new TMA kernel
  for dkdv_inline (dq_inline stays as-is for now; it's already tight and
  it's a Q-loop kernel that re-loads K/V, which is less TMA-friendly).
- Correctness: bf16 tolerance vs SDPA reference on the same shape panel
  as the ablation leg (T ∈ {1024, 2048, 4096, 8192}, GQA 2:1 and MHA,
  D ∈ {64, 128}).
- Benchmark: `rounds=15, iters=300` on the headline, vs FA3 and the
  existing `fused_delta` variant.

## Risks / exit conditions

1. `tl.make_tensor_descriptor` not usable inside JIT on this stack
   (Triton 3.6 + cu130). If the smoke test fails, revert and document.
2. Alignment constraint violation for any stride we pass (bf16 × D=64 = 128B
   aligned; bf16 × D=128 = 256B aligned; stride_qt = H*D elements which is
   always 16B-aligned for H ≥ 1, D ≥ 8 in bf16). Should be fine.
3. TMA's mandatory tile-size constraints may force BLOCK_D == D exactly,
   which matches our current use (D is already used as BLOCK_D in practice
   via `next_power_of_2(D)`).
4. If the TMA kernel ties or loses at the headline, ship it only under an
   explicit env flag; do not change the default.

## Exit criteria

- **Ship:** TMA kernel beats current `fused_delta` dkdv_inline by ≥5%
  wall at the headline AND correctness holds across the full shape panel.
  Wired into `WHALE_BWD_VARIANT=auto` dispatch for shapes where it wins.
- **Drop:** TMA kernel does not beat current `fused_delta` by ≥2% wall,
  OR correctness fails, OR the Triton stack rejects the descriptor API.
  Keep the code behind `WHALE_BWD_VARIANT=fused_delta_tma` for reference,
  leave default `auto` unchanged.
