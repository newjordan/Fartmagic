# whale_bwd_tma — results

Pod: 1×H100 SXM (35082273), CUDA 13.0, PyTorch 2.11.0+cu130, Triton 3.6,
FA3 3.0.0 cu130 abi3.

## Goal (from hypothesis.md)

Beat FA3's backward on the headline shape with pure Triton by rewriting
`_attn_bwd_dkdv_inline_delta_kernel` through `tl.make_tensor_descriptor`
(on-device TMA on Hopper). Close the 15 µs backward gap and make
pure-Triton whale the fastest path.

## Outcome: **goal not met**

TMA gives a real kernel-level win on `dkdv_inline` but the magnitude is
too small to flip the wall-time comparison against FA3 on any tested
shape. Default dispatch unchanged.

## Kernel-level (torch profiler, 50 iters, headline)

| kernel                                   | µs/iter (self CUDA) | Δ vs non-TMA |
|:-----------------------------------------|--------------------:|:------------|
| `_attn_bwd_dkdv_inline_delta_kernel`     | 194.8               | —            |
| `_attn_bwd_dkdv_inline_delta_tma_kernel` | **175.8**           | **−19.0 µs** (−9.8%) |

Real 19 µs kernel improvement. Autotune on TMA picks `(BM=64, BN=64, w=4, s=4)`
at the headline, same shape class as the non-TMA winner.

## Wall-time (bench_stable.py, same methodology as ablation leg)

Six interleaved subprocess runs, alternating order to cancel thermal drift,
reporting `whale_fast − fa3` within each run (pairing cancels pod warmth):

| variant             | mean Δ vs fa3 (fwd+bwd) |
|:--------------------|-----------------------:|
| `fused_delta`       | +9.5 µs                |
| `fused_delta_tma`   | +10.7 µs               |

Within noise. TMA ≈ fused_delta at the headline. Neither beats FA3. The
earlier "parity" claim in `legs/2026-04-16_whale_bwd_ablations/RESULTS.md`
is a single-run artifact that does not reproduce across repeats.

## Cross-shape A/B (in-process, TMA vs fused_delta, interleaved)

| shape                    | fused_delta | fused_delta_tma | Δ      |
|:-------------------------|------------:|-----------------:|:-------|
| (4,1024,8,4,64)          | 0.366 ms    | 0.378 ms         | +3.2%  |
| (4,2048,8,4,64) headline | 0.379 ms    | 0.378 ms         | −0.3%  |
| (4,2048,8,8,64)          | 0.329 ms    | 0.346 ms         | +5.0%  |
| (4,2048,16,16,64) MHA    | 0.556 ms    | 0.609 ms         | +9.6%  |
| (2,2048,8,4,128)         | 0.414 ms    | 0.391 ms         | −5.4%  |
| (4,4096,8,4,64)          | 1.121 ms    | 1.014 ms         | −8.4%  |
| (2,8192,8,4,64)          | 2.115 ms    | 1.914 ms         | −8.5%  |

**TMA helps at long T (≥4096) and D=128, hurts on MHA.** Pattern: TMA wins
where dkdv is re-loading the most bytes per Q-head iteration; it regresses
where the group loop expands the number of descriptor builds.

## vs. baseline (3-kernel) at long T — TMA does not close the gap

| shape                | baseline | fused_delta_tma | fa3     | winner    |
|:---------------------|---------:|----------------:|--------:|:----------|
| (4,4096,8,4,64)      | 0.916 ms | 1.014 ms        | 0.604 ms| **fa3**   |
| (2,8192,8,4,64)      | 1.757 ms | 1.914 ms        | 1.065 ms| **fa3**   |
| (2,2048,8,4,128)     | 0.400 ms | 0.393 ms        | 0.345 ms| **fa3**   |

Even `baseline` (still whale's best at these shapes) is 50–60% behind FA3
at long T. TMA recovers some ground over `fused_delta` but cannot close
the architectural gap — FA3's backward at long T uses warp specialization
and wgmma pipelining that Triton 3.6 does not expose at the Python level.

## What shipped

- `_attn_bwd_dkdv_inline_delta_tma_kernel` in `vault/whale_kernel_triton.py`.
- Dispatch knob: `WHALE_BWD_VARIANT=fused_delta_tma` (opt-in only).
- `_ensure_tma_allocator()` helper (required once on Triton 3.6).
- Default `auto` dispatch **unchanged** — still picks `fused_delta` for
  T ≤ 3072, `baseline` otherwise.

## Why TMA did not win more

1. `dkdv_inline` on the headline already runs at 194 µs for ~6.8 GB/s
   effective bandwidth (loads Q, K, V, O, DO + stores dK, dV for the
   whole T per `kv_h`). It is compute-bound on the two MMAs
   (`p = q·kᵀ`, `dk += dsᵀ·q`, `dv += pᵀ·do`, `dp = do·vᵀ`) more than
   bandwidth-bound, so TMA's latency-hiding only recovers a fraction.
2. At small T (1024, 2048), the descriptor-build cost per inner loop
   iteration is visible; at long T it amortises and TMA wins.
3. The MHA regression is the clearest signal: with `group=1` we build
   a fresh `(Q/O/DO)` descriptor trio per Q-head × per `m_block`
   iteration, and the setup overhead exceeds the bulk-copy saving.

## Recommendation

Ship TMA variant behind the env flag, do not change default dispatch.
Keep the file as reference for a future warp-specialised rewrite when
Triton stack exposes `async_task` / consumer groups on this pod image.
Close the leg.
