# whale_dkdv_warpspec — hypothesis (H7)

## Lever A — warp specialization on dkdv M-loop

Add `warp_specialize=True` (with tuned `num_stages`) to the inner M-loop
inside `_attn_bwd_dkdv_kernel` (the baseline-path kernel that runs at
T>3072 under `WHALE_BWD_VARIANT=auto|baseline`). Triton 3.6 cu130
exposes `tl.range(start, stop, step, num_stages=N, warp_specialize=True)`
which lets the compiler split the loop body into producer (TMA / async
HBM loads) and consumer (MMA + softmax) warp groups, mirroring the
producer/consumer pattern FA3 uses.

The new code path is gated on a `tl.constexpr` flag `WARPSPEC` so the
existing baseline path is untouched when the flag is `False`. The flag
is selected by a NEW env knob `WHALE_BWD_KV_WARPSPEC=1` read in the
dispatch site at `vault/whale_kernel_triton.py:~1223`.

`async_task` / `num_consumer_groups` are NOT exposed in this Triton 3.6
build, so we cannot match the FA3 explicit producer/consumer split — we
rely on the `warp_specialize=True` automatic split.

## Kineto baseline (H6 evidence)

- whale `auto`/`baseline` at (2, 8192, 8, 4, 64): dkdv kernel = **747 µs/iter**
- FA3 equivalent dkdv path: **~280 µs/iter**
- Headline gap: **~467 µs/iter** dkdv-only; whale fwd+bwd ≈ 2.0 ms vs
  FA3 ≈ 1.05 ms at the same shape.

## Success criteria

Headline shape (2, 8192, 8, 4, 64), `WHALE_BWD_VARIANT=baseline`,
`WHALE_BWD_KV_WARPSPEC=1`:

- **Primary (≥30% of dkdv gap closed):** dkdv kernel ≤ **607 µs/iter**
  (i.e. shaves ≥140 µs of the 467 µs).
- **Headline target:** whale_fast fwd+bwd at (2, 8192, 8, 4, 64) drops
  below **1.5 ms** (currently ~2.0 ms).
- **Stretch:** whale_fast fwd+bwd < **1.2 ms** (FA3 ≈ 1.05 ms).

Numerics: max_abs error vs SDPA reference for fwd, dq, dk, dv must
match the existing `bench_numerics.py` thresholds — i.e. no regression
vs `WHALE_BWD_KV_WARPSPEC=0`.

## Constraints / known gotchas

- **maxnreg ≥ 224 corrupts `tl.atomic_add`** on this Triton 3.6 / cu130
  stack (~19000× value amplification — see
  `legs/2026-04-16_whale_bwd_persistent_atomic/hypothesis.md` and
  memory entry `triton_gotchas_atomic_add`). The `_attn_bwd_dkdv_kernel`
  uses plain `tl.store` for dK/dV (no atomics), so the maxnreg ≥ 224
  bug does not gate this lever. We document it because warp
  specialization is sometimes paired with higher register budgets
  elsewhere; do NOT propagate maxnreg ≥ 224 to any kernel that uses
  atomics.
- `warp_specialize=True` requires `num_stages >= 2` and (per Triton 3.6
  empirical experience) `num_warps in {4, 8}` — both already covered by
  `_bwd_kv_configs()`. No new configs are strictly required, but we
  document a config-list trim in `vault_patch.md` for cache-warmth
  reasons.

## What this leg does NOT test

- TMA on the baseline dkdv kernel (separate lever, would need a new
  kernel variant since this kernel uses pointer loads, not
  `tl.make_tensor_descriptor`).
- Persistent dkdv+dq fused (already ruled out in H5).
- Warp specialization on `_attn_bwd_dq_kernel` (Lever B; tracked
  separately).
- Warp specialization on `_attn_bwd_dkdv_inline_delta_kernel` (the
  short-T path; H6 showed `auto` already routes long-T to baseline so
  the inline-delta path is not on the critical path at T=8192).
