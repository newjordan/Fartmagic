# whale_fused_bwd_no_o — hypothesis

Parent leg: `legs/2026-04-16_whale_bwd_persistent_atomic` (H5 fused_bwd lost
2x wall to `fused_delta` baseline).

## Goal

Resurrect the single-kernel fused dkdv+dq path by removing its biggest
HBM tax (O reload + delta recompute) so it can credibly beat the
2-kernel `fused_delta` winner at the headline (B=4, T=2048, H=8, KV=4,
D=64, causal, bf16) and at long-T (T=4096, T=8192) where we currently
trail FA3 ~2x.

## FA3's delta strategy (`fact`, cite: flash-attention hopper source)

`fact`: FA3 does **not** recompute `delta = sum(O * dO)` inside the main
bwd kernel. It computes delta in a separate preprocess pass and stores
it as an `[B, H, seqlen_q]` fp32 vector in HBM.

Sources (under `/home/frosty40/pytorch-sm121-src/third_party/flash-attention/hopper/`):

- `flash_bwd_preprocess_kernel.h:202-220` — preprocess kernel computes
  `dP_sum(mi) = Allreduce(sum_ni do_fp32(mi, ni) * o_fp32(mi, ni))` and
  stores to `gdPsum` (shape `[seqlen_q, head, batch]`, stride
  `Stride<_1, int64_t, int64_t>`).
- `mainloop_bwd_sm90_tma_gmma_ws.hpp:872-896` — the main bwd mainloop
  loads `tLSErdPsum` via `cute::copy(tLSEsdPsum(_, …), tLSErdPsum)`
  from SMEM (prefetched by the TMA producer warpgroup from the
  pre-computed `dPsum` tensor). Then:

      dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum_cur);

  i.e. the mainloop **never loads O**. O is only touched once, in the
  preprocess kernel.

`inference`: The "fused bwd" label on FA3 means fused **dK/dV/dQ** in
one mainloop, not fused **delta + dK/dV/dQ**. Delta precompute is
cheap (one pass over O and dO, ~B*H*T*D*2*2 = few MB at our shapes)
and moves the O traffic out of the innermost loop.

## Whale's current fused kernels (`fact`, cite: vault/whale_kernel_triton.py)

- `_attn_bwd_fused_kernel` L810+, inner loop (L866-872):
      q_ptrs = ...; do_ptrs = ...; o_ptrs = ...
      q = tl.load(q_ptrs, ...)
      do = tl.load(do_ptrs, ...)
      o_tile = tl.load(o_ptrs, ...)
      delta = tl.sum(o_tile.to(fp32) * do.to(fp32), axis=1)
- `_attn_bwd_fused_tma_dq_kernel` L914+ has the same `o_tile` load
  (L979-983), same delta recompute.

Both kernels iterate over `group * M_blocks` times per (b, kv_h, n)
program. Each iteration reloads an `O` tile of
`BLOCK_M × D` bf16 = 2*BLOCK_M*D bytes. For the headline
(group=2, M_blocks=T/BLOCK_M=32, BLOCK_M=64, D=64) that's
`2 * 32 * 2 * 64 * 64 = 524288 bytes = 0.5 MB` of O-reload per
program, multiplied by the grid `(T/BN=32, B*KV=16) = 512`
programs = **256 MB of redundant O-reload per call at the headline**.

That's in addition to the atomic_add dQ tax already diagnosed in the
H5 RESULTS.md (inference-only, not required for this hypothesis).

## Hypothesis

Replace the in-kernel `o_tile` load + delta recompute with a
precomputed delta tensor (identical to what `_attn_bwd_preprocess_kernel`
already produces for the non-fused path). At bwd time:

1. Launch `_attn_bwd_preprocess_kernel` (already exists, L???) to
   compute `delta [B, H, T] fp32` from O and dO.
2. Launch `_attn_bwd_fused_no_o_kernel` (new) — same as
   `_attn_bwd_fused_kernel` but takes `DELTA` instead of `O`, and
   loads `delta` as a row vector (`BLOCK_M` floats per iter,
   i.e. 512 B per iter at BLOCK_M=128) instead of an `O` tile
   (BLOCK_M*D*2 B per iter, i.e. 16384 B at BLOCK_M=128, D=64).

That's a 32x reduction in the per-iteration delta-related traffic
for the headline shape, at a cost of one extra cheap preprocess launch.

## Expected effect (`inference`)

- Remove 256 MB of O reload at headline (above). HBM BW on H100 is
  ~3 TB/s effective, so 256 MB is worth ~85 µs of pure BW in the
  best case, and more in practice because it's read-modify inside the
  wgmma-dominated inner loop and it costs register pressure.
- Add ~2 MB of delta traffic (preprocess writes + fused reads), worth
  ~0.7 µs each side.
- Preprocess kernel itself: O(BH * T * D) work, same compute as the
  non-fused path's already measured ~15 µs preprocess overhead (from
  the 2026-04-16_whale_bwd_ablations leg).
- Net model: fused_bwd at headline was 0.709 ms – 0.359 ms = **+350 µs
  vs fused_delta**. The O reload is one of two known taxes (the
  other is atomic_add on dQ). Killing O reload alone is worth ~85 µs
  BW + register-pressure reduction (harder to quantify). If this
  closes half the gap, we're at ~0.55 ms — still losing, and the
  atomic_add tax must then be attacked separately (Lever C).

## Exit criteria

- **Ship (narrow):** fused_bwd_no_o at headline is ≥10% faster
  than the current fused_bwd (H5) AND correctness holds bf16 vs
  SDPA ref. Keep behind `WHALE_BWD_VARIANT=fused_bwd_no_o`, do **not**
  change `auto` dispatch.
- **Ship (win):** fused_bwd_no_o beats `fused_delta` at any of the 4
  primary shapes AND holds at the others AND correctness holds.
- **Drop:** fused_bwd_no_o is ≤5% different from fused_bwd, or
  correctness fails, or register pressure forces a small BLOCK_M
  that wipes the bandwidth win.

## Primary shapes

Same 4 as the bench queue; these cover the headline + long-T
regressions we still need to close vs FA3:

1. `4,2048,8,4,64`   — headline, current parity vs FA3.
2. `4,4096,8,4,64`   — whale fused_delta trails FA3 ~2x.
3. `2,8192,8,4,64`   — same regression, worse.
4. `2,2048,8,4,128`  — D=128 path (different MMA shape).

## Non-goals for this leg

- Not touching dQ atomic_add (that's Lever C, separate leg).
- Not touching TMA variant (fused_tma_dq) — fused_bwd_no_o should
  be testable without TMA first, then add a TMA twin if the plain
  variant wins.
- Not modifying forward. Not modifying inline-delta (2-kernel)
  variant — that path already doesn't load O because delta is
  separate.
