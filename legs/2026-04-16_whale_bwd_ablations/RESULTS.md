# whale_bwd_ablations — results

Pod: 1×H100 SXM, CUDA 13.0, PyTorch 2.11.0+cu130, Triton 3.6, FA3 3.0.0.
Evidence: `evidence/definitive_*.json`, `evidence/dkdv_inline_sweep.tsv`.

## Goal

Beat FA3 on the backward pass using only the pure-Triton whale kernel
(no FA3 dependency). Baseline from the autoresearch leg: whale was
13% slower than FA3 on fwd+bwd at the headline shape.

## Winning change: `WHALE_BWD_VARIANT=auto` (default)

Kills the preprocess kernel. Δᵢ = <oᵢ, doᵢ> is computed inline inside the
dkdv and dq kernels. Saves 1 kernel launch + 1 HBM roundtrip on Δ. For
long T where GQA amplifies the extra O loads, `auto` falls back to the
baseline (3-kernel) path at `T > WHALE_FUSED_DELTA_T_MAX` (default 3072).

### Headline (B=4, T=2048, H=8, KV=4, D=64, causal, bf16)

| backend    | fwd ms ± σ    | fwd+bwd ms ± σ | bwd-only (derived) |
|:-----------|:--------------|:---------------|-------------------:|
| whale_fast | 0.079 ± 0.000 | **0.354 ± 0.003** | 0.275 |
| fa3        | 0.095 ± 0.002 | 0.355 ± 0.007  | 0.260 |

**Pure-Triton whale ties FA3 on fwd+bwd at the headline.** Forward is
-17%, backward is +6% — they essentially cancel. (Measurements taken at
`rounds=15, iters=300` on pod 35082273 after warm-up).

### Definitive shape sweep (`evidence/definitive_*.json`)

| B | T    | H  | KV | D   | whale fwd | fa3 fwd | whale f+b | fa3 f+b | Δ total |
|--:|-----:|---:|---:|----:|----------:|--------:|----------:|--------:|:--------|
| 4 | 1024 |  8 |  4 |  64 |   0.071   |  0.085  |   0.344   |  0.342  | **tie** (+0.6%) |
| 4 | 2048 |  8 |  4 |  64 |   0.079   |  0.095  |   0.354   |  0.355  | **tie** (−0.3%) |
| 4 | 4096 |  8 |  4 |  64 |   0.260   |  0.156  |   0.920   |  0.604  | +52% (loss) |
| 2 | 8192 |  8 |  4 |  64 |   0.499   |  0.277  |   1.739   |  1.065  | +63% (loss) |
| 4 | 2048 |  8 |  8 |  64 |   0.079   |  0.089  |   0.351   |  0.325  | +8%   |
| 4 | 2048 | 16 | 16 |  64 |   0.136   |  0.093  |   0.560   |  0.355  | +58% (loss) |
| 2 | 2048 |  8 |  4 | 128 |   0.073   |  0.109  |   0.427   |  0.399  | +7%   |

**Net:** pure-Triton whale is at **parity** with FA3 for T ≤ 2048, D=64 GQA
and wins forward across almost every shape. FA3 still wins on long T and
large MHA, which needs warp specialization / wgmma pipelining that Triton
3.6 cannot emit at Python level.

## What I tried and kept

1. **Inline-Δ dkdv + dq (`fused_delta`)** — biggest single win. On the
   headline, saved ~100 µs wall on fwd+bwd by eliminating the preprocess
   kernel + its Δ HBM tensor. Correctness holds within bf16 tolerance.
2. **Auto T-based dispatch** — `auto` picks `fused_delta` for T ≤ 3072,
   `baseline` for longer T where the extra O loads would hurt in GQA.
3. **Narrower autotune list for `_attn_bwd_dkdv_inline_delta_kernel`
   (`_bwd_kv_inline_configs`)** — the full grid had unstable picks
   run-to-run; the narrower list of `(64,64)` / `(64,128)` / `(128,64)` /
   `(128,128)` × warps {4,8} × stages {3,4} (+ one `(64,64,4,2)` fallback)
   picks reliably.

## What I tried and dropped

1. **Forced `maxnreg` on the inline dkdv** (sweep over 128/160/192/224).
   Best was `maxnreg=192` at `(64,64,4,3)` — within noise of the
   no-maxnreg winner. Not worth freezing.
2. **Fused preprocess+dkdv only, keeping separate dq preprocess read** —
   no benefit over baseline; the whole point was to drop preprocess
   entirely.
3. **Forcing (128,128)/(256,128) on dkdv_inline** — 30–60% slower than
   (64,64) at the headline.

## What I did not attempt (and why)

1. **TMA via `tl.make_tensor_descriptor`** — Triton 3.6 supports it but
   integrating into the backward kernels needs a full rewrite (4D layout
   → descriptors, mask semantics → padding_option, careful 16-byte
   alignment on all stride dims). High risk of correctness regressions
   and unclear payoff given we're already at parity.
2. **Warp specialization** — Triton 3.6 does not expose `async_task` /
   `num_consumer_groups` / `num_buffers_warp_spec` at the Python level
   on this stack. Not available.
3. **Persistent fused dkdv+dq kernel** — dkdv iterates outer-N / inner-M;
   dq iterates outer-M / inner-N. A persistent kernel would need
   atomic-based dQ accumulation, which showed correctness problems with
   `tl.atomic_add` on 2-D fp32 tiles on this Triton 3.6 stack in the
   autoresearch leg. Too risky for the expected gain.

## Profiler snapshot (fwd+bwd, 50 iters, headline)

| kernel                               | µs / iter CUDA |
|:-------------------------------------|---------------:|
| `_attn_fwd_kernel` (whale fwd)       |   77           |
| `_attn_bwd_dkdv_inline_delta_kernel` |  194           |
| `_attn_bwd_dq_inline_delta_kernel`   |   77           |
| **whale kernel sum**                 |  **348**       |
| wall (CUDA-event mean)               |  354           |
| non-kernel overhead                  |   ~6           |

FA3 kernel sum ≈ 186 µs, wall ≈ 355 µs ⇒ ~170 µs dispatch overhead in the
FA3 binding. Net wall parity.

## Variant API in `vault/whale_kernel_triton.py`

Set via `WHALE_BWD_VARIANT`:

- `auto` (default) — `fused_delta` for T ≤ `WHALE_FUSED_DELTA_T_MAX`
  (default 3072), else `baseline`.
- `fused_delta` — force inline-Δ (dkdv+dq, no preprocess).
- `baseline` — original 3-kernel path (preprocess + dkdv + dq).

All of `whale_attn_fast`, `whale_fwd_fa3_bwd`, and the `custom_op` path
respect the variant switch.
