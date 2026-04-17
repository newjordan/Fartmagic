# whale_bwd_persistent_atomic — hypothesis (H5)

## Goal

Close the remaining ~170 µs FA3 backward dispatch-overhead gap identified in
`legs/2026-04-16_whale_bwd_ablations/RESULTS.md` (profiler snapshot) by
folding `_attn_bwd_dq_inline_delta_kernel` into the dkdv pass:

- One persistent kernel per (b, kv_head, n_block).
- Per K/V tile, iterate Q-group M-blocks; compute dK/dV in registers and
  atomic-add the dQ contribution into a global fp32 scratch.
- A tiny cast kernel writes the bf16 dQ from the fp32 scratch.

Expected save on headline (B=4, T=2048, H=8, KV=4, D=64, bf16 causal):
- ~1 kernel launch eliminated (dq) → ~6–10 µs.
- 1 HBM reload of K/V eliminated for dq → ~15–25 µs.
- Total expected: 20–35 µs wall reduction vs `fused_delta`.

## Blocker found (2026-04-16)

The H5 path was scaffolded but blocked by an atomic_add correctness bug:

- `_bwd_kv_inline_configs()` uses `maxnreg=224` for every config
  (vault/whale_kernel_triton.py:122, 124).
- Under `maxnreg=224`, `tl.atomic_add` of a 2-D fp32 tile amplified probe
  values by ~1.9e4× — in the probe `dq_local = row*1 + col*1000`, the
  observed post-atomic values were ~19005× the expected sum.
- Removing `maxnreg` (letting the compiler pick) → ratio 1.0, probe
  correct.

## Fix plan

1. Introduce `_bwd_fused_configs()` — a separate config list for
   `_attn_bwd_fused_kernel` that **does not** apply `maxnreg=224`.
   (Env override `WHALE_BWD_FUSED_MAXNREG=<int>` still available for
   intentional sweeps, but default-off.)
2. Switch `_attn_bwd_fused_kernel`'s autotune decorator from
   `_bwd_kv_inline_configs` → `_bwd_fused_configs`.
3. Replace the probe `dq_local = row*1 + col*1000` with the real dQ
   contribution:
       dq_local = tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32) * SCALE
   where `ds = p * (dp - delta[:, None])` is already computed for dK.
   The scaling factor for dQ is `SCALE = 1/sqrt(D)` (same as forward);
   `qk_scale_log2` includes `LOG2E`, which does NOT go into dQ.
4. Re-run numerics check vs SDPA reference: all of dq/dk/dv within the
   `|grad| < 5e-2` bf16 tolerance used in `bench_numerics.py`.

## Success criterion

Pure-Triton `whale fused_bwd` wall ≤ pure-Triton `whale fused_delta` wall
on the headline shape (~354 µs). Stretch: ≤ FA3 backward wall (~355 µs
fwd+bwd total). Correctness within bf16 tol on every shape in the
whale_bwd_ablations definitive sweep.

## Variables

Single variable: switch `WHALE_BWD_VARIANT=auto`/`fused_delta` →
`fused_bwd` and measure.

## What this does NOT test

- TMA descriptors (that's H2, separate leg).
- D=32 or MHA 16:16 perf (covered by subsequent shape sweeps, not here).
- End-to-end 8×H100 training step (blocked on multi-GPU phase).
