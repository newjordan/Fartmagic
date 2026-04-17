# H9 — whale forward at long T

## fact
- At T=8192 (B=2, H=8, KV=4, D=64), whale fwd ~462us/iter; FA3 fwd portion ~280us.
- Gap = ~180us; that is ~25% of the long-T fwd+bwd total loss to FA3.
- The TMA forward kernel (`WHALE_FWD_VARIANT=tma`) exists in `vault/whale_kernel_triton.py`
  (`_attn_fwd_tma_kernel`, lines 302-384) but has not been benchmarked at long T in any
  leg's evidence corpus that compares against FA3 directly.
- FA3 fwd Sm90 path uses TMA + warp-specialized producer/consumer split + WGMMA.
- Triton 3.6 cu130 exposes `tl.range(num_stages=N, warp_specialize=True)` and on-device
  TMA descriptors via `tl.make_tensor_descriptor`. It does NOT expose `tl.async_task`.

## hypothesis
The 180us long-T gap has two recoverable components:

### H9a — autotune, not architecture
`WHALE_FWD_VARIANT=tma` already wins at long T on the right (BM, BN, warps, stages),
but autotune (`_fwd_tma_configs`, lines 81-94) is keyed on
`["D", "IS_CAUSAL", "NUM_HEADS", "NUM_KV_HEADS", "T_MAX"]` and picks a tile that
overfits short-T. At T=8192, FA3-like tiles (BM=BN=128, w=8, s=2-3) should win
clearly over short-T winners (often BM=128, BN=64, s=4).
- Test: force-config sweep on TMA variant at T=8192 via `WHALE_FWD_TMA_CONFIG=BM,BN,W,S`.
- Falsify if every forced config is slower than the autotuned default.

### H9b — warp specialization on K/V loop
The K/V loop in `_attn_fwd_tma_kernel` (line 359, `for start_n in range(0, hi, BLOCK_N)`)
runs as a single warpgroup with no producer/consumer split. Marking that loop
`tl.range(0, hi, BLOCK_N, num_stages=NUM_STAGES, warp_specialize=True)` should let
Triton 3.6 schedule TMA loads and WGMMA on separate warps, closing more of the
180us gap on top of any H9a win.
- Test: patch the kernel (doc-only here, see `vault_patch.md`) and bench at T=8192.
- Falsify if wall time rises or correctness breaks.

## plan
- Phase A — variant sweep, no kernel change. `default` vs `tma` at T in {2048, 4096, 8192}.
- Phase B — TMA + forced configs at T=8192 (FA3-like tiles).
- Phase C — TMA + warp-spec patch (gated by `WHALE_FWD_TMA_WARPSPEC=1`) at T=8192.
  Phase C only runs if the patch from `vault_patch.md` has been applied to the vault
  and the user has approved the vault edit.

## success criterion
- A: choose the better variant at long T; document the autotune winner per shape.
- B: any forced config beats autotune by >=10us at T=8192.
- C: warp-spec config beats best-of-A-and-B by >=20us at T=8192 with no correctness
  regression (max abs err vs reference <= 1e-3 bf16).

## fallback
If neither H9a nor H9b clears 30us at T=8192, log the long-T floor and pivot to
attention-block-level work (e.g. KV cache reuse across the dataset prep step).

## inference / proposal split
- fact: numbers above are from the leg brief; long-T fwd 462us and FA3 fwd 280us
  must be re-confirmed by Phase A before the gap claim is binding.
- inference: H9a is more likely to land than H9b because the autotune key already
  includes T_MAX, but only a small set of T values appears in the existing corpus.
- proposal: Phase C kernel patch is doc-only; do not edit `vault/` until Phase A+B
  evidence justifies it.
