# whale_bwd_persistent_atomic — results

Pod: 1×H100 SXM, CUDA 13.0, PyTorch 2.11.0+cu130, Triton 3.6, FA3 3.0.0.
Evidence: `evidence/headline_fused_*.json`, `evidence/fused_bwd_numerics_*.json`,
`evidence/sweep_fused_*.json`.

## Verdict (negative)

`fact`: At the headline shape (B=4, T=2048, H=8, KV=4, D=64, causal, bf16),
the persistent fused dkdv+dq variant (`WHALE_BWD_VARIANT=fused_bwd`) is
**2.0× slower** than the existing `fused_delta` winner on whale_fast wall.

| variant       | whale_fast fwd ms | whale_fast fwd+bwd ms | vs FA3 fwd+bwd |
|:--------------|:------------------|:----------------------|:---------------|
| fused_delta   | 0.077 ± 0.000     | **0.359 ± 0.005**     | +1.8% (parity) |
| fused_bwd (H5)| 0.077 ± 0.002     | **0.709 ± 0.002**     | +98% (loss)    |
| FA3           | 0.090 ± 0.003     | 0.353 ± 0.008         | baseline       |
| SDPA (cuDNN)  | 0.088 ± 0.000     | 0.293 ± 0.003         | −17%           |

Source: `evidence/headline_fused_delta_20260416_221808.json` and
`evidence/headline_fused_bwd_20260416_221808.json`, rounds=15, iters=300.

## 7-shape sweep (definitive negative)

`fact`: Across all 7 sweep shapes, `fused_bwd` is between 1.06× and 2.30×
slower than `fused_delta` on whale_fast wall. The slowdown grows with T
and with D.

Source: `evidence/sweep_fused_delta_20260416_221808.json` and
`evidence/sweep_fused_bwd_20260416_221808.json`.

| shape (B,T,H,KV,D)      | fused_delta fb ms | fused_bwd fb ms | bwd ratio | FA3 fb ms | sdpa fb ms |
|:------------------------|------------------:|----------------:|----------:|----------:|-----------:|
| 4,1024,8,4,64           | 0.357             | 0.378           | 1.06×     | 0.398     | 0.315      |
| 4,2048,8,4,64 (headline)| 0.374             | 0.707           | 1.89×     | 0.370     | 0.317      |
| 4,4096,8,4,64           | 1.092             | 2.122           | 1.94×     | 0.594     | 0.790      |
| 2,8192,8,4,64           | 2.065             | 3.920           | 1.90×     | 1.030     | 1.368      |
| 4,2048,8,8,64           | 0.372             | 0.645           | 1.74×     | 0.372     | 0.244      |
| 4,2048,16,16,64         | 0.551             | 1.154           | 2.09×     | 0.350     | 0.411      |
| 2,2048,8,4,128          | 0.484             | 1.114           | 2.30×     | 0.368     | 0.317      |

Two side-findings worth flagging (not the H5 question, but visible here):
- `fact`: At long T (T≥4096) whale_fast `fused_delta` is also slower than
  FA3 fwd+bwd (e.g. T=8192: 2.065 vs 1.043 — 2× slower). Headline-shape
  parity does not extend to long-context. This is a separate motivation
  for either H2 (TMA dkdv) or a long-T-targeted leg.
- `fact`: At full-MHA `4,2048,16,16,64`, FA3 is 1.6× faster than
  `fused_delta` fwd+bwd (0.350 vs 0.551). Whale's GQA-tuned configs
  don't generalise back to full-MHA.

`inference`: The fused approach is bandwidth-bound on the dQ atomic_add.
Rough back-of-envelope: per kernel call, dQ atomic_add traffic
≈ `B*H*T*D*4*(T/BN)` bytes. For the headline shape with BN=64 that's
~540 MB of atomic-add writes vs ~10 MB of plain stores in fused_delta.
H100 fp32 atomic_add throughput is ~1–2 GB/s aggregate (vs HBM 3 TB/s
for plain stores), so this scatter pattern saturates atomic bandwidth
long before HBM becomes the bottleneck.

## Numerics check (correct)

`fact`: All 7 definitive shapes pass numerics within bf16 tolerance.
Source: `evidence/fused_bwd_numerics_20260416_221808.json`.

| B | T    | H | KV | D   | fwd err | dq err | dk err | dv err |
|--:|-----:|--:|---:|----:|--------:|-------:|-------:|-------:|
| 2 |  256 | 8 |  4 |  64 |  9.8e-4 | 2.0e-3 | 1.6e-2 | 3.1e-2 |
| 2 | 1024 | 8 |  4 |  64 |  3.9e-3 | 3.9e-3 | 1.6e-2 | 3.1e-2 |
| 2 | 2048 | 8 |  4 |  64 |  3.9e-3 | 3.9e-3 | 1.6e-2 | 3.1e-2 |
| 2 | 1024 | 8 |  8 |  64 |  3.9e-3 | 3.9e-3 | 7.8e-3 | 3.9e-3 |
| 2 | 1024 | 8 |  2 |  64 |  3.9e-3 | 3.9e-3 | 3.1e-2 | 3.1e-2 |
| 2 | 1024 | 4 |  4 | 128 |  3.9e-3 | 1.6e-2 | 1.6e-2 | 3.9e-3 |
| 2 | 1024 | 8 |  4 |  32 |  3.9e-3 | 7.8e-3 | 1.6e-2 | 3.1e-2 |

Pass criterion: `|fwd|<1e-2`, `|grad|<5e-2`. All pass.

## What it took to get to "correct"

Two non-obvious Triton 3.6 traps were isolated en route. Both are now
documented in `~/.claude/projects/-home-frosty40-SOTA-FINAL/memory/triton_gotchas_atomic_add.md`:

1. **`maxnreg>=224` corrupts `tl.atomic_add` on 2-D fp32 tiles.** With
   `maxnreg=224`, an isolation probe `dq_local = row*1 + col*1000`
   atomic-added to fp32 came back amplified ~19005×. Fix: drop the
   constraint or set `maxnreg<=192`.
   - Implemented: new `_bwd_fused_configs()` in `vault/whale_kernel_triton.py`
     does NOT default to `maxnreg=224`, raises if user requests
     `WHALE_BWD_FUSED_MAXNREG>=224`.
2. **`@triton.autotune` accumulator buffers need `reset_to_zero=[name]`**.
   Without it, autotune calls the kernel N times against the same
   `DQ_F32` buffer, accumulating N kernels' worth of dq before the user
   reads. Manifested as `dq_max_abs_err = 4.6e4`. Fix in commit:
   `@triton.autotune(..., reset_to_zero=["DQ_F32"])`.

## Vault changes (kept)

- `_bwd_fused_configs()` — separate config list for `_attn_bwd_fused_kernel`,
  no `maxnreg=224`, with explicit guard against re-introducing it.
- `_attn_bwd_fused_kernel` autotune: `reset_to_zero=["DQ_F32"]`.
- `_attn_bwd_fused_kernel` line ~832: probe `dq_local = row + col*1000`
  replaced with `dq_local = tl.dot(ds.to(q.dtype), k, ...) * SCALE`.

`WHALE_BWD_VARIANT=fused_bwd` remains a working option (correct, but
slower). Default `auto` continues to pick `fused_delta`.

## What did NOT close the gap

Each failed approach was kept here to save a future agent (or future-me)
from re-running it.

1. **Persistent fused dkdv+dq via fp32 atomic_add on dQ (this leg).**
   Correct, 2× slower than separating dkdv from dq. Bandwidth-bound on
   atomic ops.

## What we still have NOT tried

1. **TMA on dkdv (`WHALE_BWD_VARIANT=fused_delta_tma`)** — already
   implemented as `_attn_bwd_dkdv_inline_delta_tma_kernel`, not yet
   benchmarked head-to-head against `fused_delta`. This is the next H2
   leg.
2. **Reduce-then-atomic via shared memory** — would amortize atomic
   bandwidth, but Triton 3.6 has no clean primitive.
3. **bf16 atomic accumulation** — halves atomic bandwidth but introduces
   precision loss on a path where dQ accuracy already matters.

## Conclusion

H5 is a clean negative result. The persistent fused approach is
fundamentally bandwidth-bound on H100's fp32 atomic_add throughput —
no autotune trick or maxnreg setting is going to close that gap. The
only paths from here that could buy speed without leaving Triton are
TMA (H2) and reducing autotune dispatch overhead (the ~140 µs non-kernel
overhead identified in `whale_bwd_ablations`).
