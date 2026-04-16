# whale forward: close 25us gap to FA3 Sm90

## fact
- `_attn_fwd_kernel` GPU time: 72us. FA3 `FlashAttnFwdSm90`: 47us.
- whale fwd wall time: 79us (CPU ~3us). FA3 fwd wall: 95us (CPU ~47us).
- we already win on wall; goal here is pure GPU throughput.

## hypothesis
Gap is due to one or more of:
1. autotune picking suboptimal (BLOCK_M, BLOCK_N, warps, stages).
2. missing TMA for K/V loads (memory-bound inner loop).
3. missing warp specialization (compute-vs-load overlap inside the K/V loop).

## plan
1. Dump which autotune config wins on the pod for the headline shape + D=128.
2. Sweep a wider grid via `WHALE_FWD_CONFIG` to see if a better manual config exists.
3. Try a TMA-load variant of the fwd kernel (K/V desc loads); measure.
4. Try `tl.async_task` warp specialization for load/compute overlap; measure.
5. Each step: keep only if kineto GPU time drops and correctness holds.

## success criterion
`_attn_fwd_kernel` GPU time drops to ≤ 50us (within 10% of FA3) at headline.
Correctness: max abs err on output vs reference ≤ 1e-3 bf16.

## fallback
If nothing closes below 55us in Triton 3.6, document the floor and move to backward.
