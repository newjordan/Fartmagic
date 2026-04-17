# Hypothesis — whale dkdv stacked-all (4 levers)

Date: 2026-04-17
Parent kernel: `vault/whale_kernel_triton.py` @ `_attn_bwd_dkdv_inline_delta_kernel`
                (declared L587; also tracks `_attn_bwd_dkdv_kernel` L455, TMA sibling L686).
Parent leg family:
- `legs/2026-04-16_whale_dkdv_early_exit/` — mask-split landed in `_attn_bwd_dkdv_kernel` (vault L455-L572)
- `legs/2026-04-16_whale_dkdv_early_exit_inline_delta/` — mask-split prepped for inline-Δ (pending, sibling leg)
- `legs/2026-04-17_whale_dkdv_autotune_expand/` — config sweep (prepped; NOT YET BENCHED)
- `legs/2026-04-17_whale_dkdv_tma_loads/` — TMA Q/K/V/DO loads (prepped; NOT YET BENCHED)
- `legs/2026-04-17_whale_dkdv_persistent/` — persistent dkdv (prepped; NOT YET BENCHED)

Hardware: 1×H100 SXM (cu130 stack, Triton 3.6, FA3 wheel cu130).

## Primary shape
`B=2, T=8192, H=8, KV=4, D=64`
- FA3 fb   = 1.028 ms (baseline from prior bench)
- whale fb (current, pre-stack) = 1.700 ms (approx)
- Target fb <= 0.51 ms  (0.50× FA3)
- Stretch  <= 0.31 ms   (0.30× FA3)
- Required speedup: 3.33× (target) / 5.48× (stretch)

## Secondary shapes (auto-routed through inline-Δ with `WHALE_FUSED_DELTA_T_MAX=8192`)
- `B=2, T=8192, H=8, KV=4, D=64`   (primary, forced into inline-Δ path)
- `B=2, T=3072, H=8, KV=4, D=64`   (boundary shape, default inline-Δ)
- `B=2, T=2048, H=8, KV=4, D=64`   (short-T, default inline-Δ)
- `B=2, T=8192, H=8, KV=4, D=128`  (wider-D sanity)

## Fact — per-lever status (as of 2026-04-17)

1. **Early-exit / mask-split (Lever 0)** — LANDED in `_attn_bwd_dkdv_kernel`
   (vault `whale_kernel_triton.py` L496-L564). The inline-Δ sibling patch is
   prepped in `legs/2026-04-16_whale_dkdv_early_exit_inline_delta/vault_patch.md`
   but NOT yet applied. Stacking requires landing it on
   `_attn_bwd_dkdv_inline_delta_kernel` (vault L587+) and
   `_attn_bwd_dkdv_inline_delta_tma_kernel` (vault L686+) FIRST, before any
   Lever A/B/C change — otherwise the TMA and persistent variants will
   regress the mask-split win by reintroducing the per-tile causal `tl.where`.
2. **Autotune config expand (Lever A)** — prepped, expands
   `_bwd_kv_inline_configs()` (vault L109-L125) from the current 17 configs
   (4 BM/BN × 2 warps × 2 stages + the (64,64,4,2) escape hatch) to a wider
   grid with extra `num_stages=5`, BM=256/BN=128 wide-tile rows, and a
   maxnreg=192 variant. Pure-Python config addition; zero kernel edits.
3. **TMA loads for Q/K/V/DO (Lever B)** — prepped, converts the bf16 HBM
   `tl.load` pointers in `_attn_bwd_dkdv_inline_delta_kernel` to
   `tl.make_tensor_descriptor(...).load(...)`. Env-gated via
   `WHALE_DKDV_TMA_LOADS=1`; the default path remains pointer-based.
4. **Persistent dkdv (Lever C)** — prepped, introduces a new
   `_attn_bwd_dkdv_inline_delta_persistent_kernel` that owns one persistent
   program per SM, iterates over `(b, kv_h, pid_n)` tiles by index, and reuses
   a single warp-group's pipeline across tiles. Env-gated via
   `WHALE_DKDV_PERSISTENT=1`. Dispatcher in `_whale_attn_bwd_impl`
   (vault L1474-L1505) chooses the persistent kernel when the env is set.

## Inference — expected per-lever contribution at the primary shape

| Lever                | Expected fb delta (µs) | Mechanism                                                                 |
|----------------------|------------------------|---------------------------------------------------------------------------|
| 0 Mask-split (inline)| −150 to −250 µs        | Saves per-tile `tl.where`+predicate on unmasked M-blocks; T=8192 has ~64 M-blocks, only 1 straddles the diagonal for any given pid_n.|
| A Autotune expand    | −30 to −80 µs          | Picks a deeper-pipelined (stages=5) or wider-tile config specifically at T=8192. Bounded by the autotune search budget's coverage.|
| B TMA loads          | −100 to −250 µs        | Hopper bulk copies + swizzled SMEM for Q/DO/K/V; saves LDG latency & allows the pipeline to hide it behind MMA. Matches the 157 µs GPU gap kineto already showed.|
| C Persistent         | −50 to −200 µs         | Amortizes K/V SMEM setup over multiple tiles on the same SM; reduces scheduling jitter. At B=2, KV=4, T=8192/BN=64 → 128 tiles × 8 programs; persistent can keep 132 SMs saturated without re-dispatch overhead.|

Sum of midpoints: ~−540 µs; whale fb 1.700 → 1.160 ms. That still misses 0.51 ms.
Sum of optimistic ends: ~−780 µs; whale fb 1.700 → 0.92 ms. Still misses.

**Consequence**: none of the 4 levers alone or in sum is enough. The brief is
honest that the gap is 3-6× and stacking is necessary but probably not
sufficient. Expected realistic end-state from this leg:
- fb ~0.95-1.10 ms (0.92-1.07× FA3 ratio ≈ 2× better than current).
- Target of 0.51 ms is **not expected** to fall on this leg alone; this leg
  produces the evidence needed to decide whether to pursue the
  `_attn_bwd_fused_kernel` persistent-with-atomic-dQ path (vault L810) as
  the follow-on, which collapses dkdv+dq into one kernel launch and removes
  K/V re-read.

## Independence analysis — do the levers interfere?

### Lever 0 (mask-split) vs A (autotune expand)
Independent. Autotune key is `(D, IS_CAUSAL, NUM_HEADS, NUM_KV_HEADS, T_MAX)`;
mask-split changes the IR but leaves the key unchanged. New configs apply to
the new IR only — cache must be flushed once
(`WHALE_FLUSH_TRITON_CACHE=1`). No logical interaction.

### Lever 0 (mask-split) vs B (TMA loads)
Mostly independent, with ONE interaction: the TMA `make_tensor_descriptor`
loads have a `block_shape=[BLOCK_M, D]` compile-time bound. The mask-split
structure keeps the same `BLOCK_M` across both phases, so descriptor shape
is preserved. The unmasked phase drops the causal `tl.where` but still
issues identical Q/DO tile loads — TMA loads the same tile either way. See
`stacking_analysis.md` for the line-by-line verification.

### Lever 0 vs C (persistent)
Low-risk interaction. Persistent changes the outer loop (`pid_n` driven by
a persistent loop over work-tile ids) but the inner M-loop body — including
the mask-split — is reused verbatim. The persistent driver wraps the kernel
body; it does not touch the split.

### A (autotune expand) vs B (TMA loads)
Interaction. TMA forces `BLOCK_D == D` (see vault L680 comment). New
autotune rows that would vary `BLOCK_D` must be stripped from
`_bwd_kv_inline_tma_configs()` (vault L128-L143). The TMA-path config list
is already shape-constrained; Lever A changes apply to the non-TMA
`_bwd_kv_inline_configs` (L109). If `WHALE_DKDV_TMA_LOADS=1`, the TMA
autotune list must be updated in parallel — this leg ships both updates.

### A (autotune expand) vs C (persistent)
Interaction. Persistent kernel has different grid semantics
(`grid = (NUM_SMS,)` instead of `(cdiv(T, BN), B*KV)`). Its autotune config
set must be a separate function (`_bwd_kv_inline_persistent_configs`) because
the register-pressure profile is different (persistent kernel holds more
state across tiles). This leg's autotune expansion touches ONLY the
non-persistent list; the persistent kernel gets its own narrower list.

### B (TMA loads) vs C (persistent)
Highest interaction. The per-tile `tl.make_tensor_descriptor` call inside
the outer loop is expensive in compiled SASS (descriptor init), so under
persistent we must:
- Hoist K/V descriptor creation outside the persistent tile loop (they depend
  only on `b` and `kv_h`, which change per work-tile).
- Recreate Q/O/DO descriptors per Q-head-group inside the inner loop
  (already the pattern in `_attn_bwd_dkdv_inline_delta_tma_kernel` L747-L758).
This requires splitting descriptor creation from tile iteration, which is a
moderate kernel rewrite. See `stacking_analysis.md` for exact line patterns.

## Proposal — stacking order (lowest risk first)

1. **Lever 0 — inline-Δ mask-split** (sibling leg's pending patch). Baseline.
   Bench. If regression, abort — stacked leg does not proceed.
2. **Lever A — autotune expand** (zero kernel change, only config list
   growth). Bench. If neutral-or-positive, keep.
3. **Lever B — TMA loads** (moderate kernel change, env-gated). Bench with
   `WHALE_DKDV_TMA_LOADS=1` on. If regression, leave env off (Lever B is
   shipped but dormant).
4. **Lever C — persistent dkdv** (highest-risk, new kernel path). Bench with
   `WHALE_DKDV_PERSISTENT=1` on. If regression, leave env off.

## Success criterion (per shape, fb = fwd+bwd latency)

| Shape                       | Pre-stack fb | Target fb (<=) | Stretch fb (<=) |
|-----------------------------|--------------|----------------|-----------------|
| 2,8192,8,4,64 (primary)     | 1.700 ms     | 0.51 ms        | 0.31 ms         |
| 2,3072,8,4,64               | TBD (bench)  | −15% vs pre    | −25% vs pre     |
| 2,2048,8,4,64               | TBD (bench)  | −10% vs pre    | −20% vs pre     |
| 2,8192,8,4,128              | TBD (bench)  | −25% vs pre    | −40% vs pre     |

Numerics tolerance (reused from `legs/2026-04-16_whale_fast_kernels/bench_numerics.py`):
- max |Δ| fwd out vs sdpa <= 5e-3 in bf16
- max |Δ| dQ/dK/dV vs sdpa <= 1e-2 in bf16

## Non-goals for this leg
- Do NOT touch `_attn_bwd_fused_kernel` (vault L810) or
  `_attn_bwd_fused_tma_dq_kernel` (vault L914). Those are the dkdv+dq fused
  persistent-atomic path and belong to `legs/2026-04-16_whale_bwd_persistent_atomic/`.
- Do NOT edit `_attn_bwd_dq_kernel` or `_attn_bwd_dq_inline_delta_kernel`
  in this leg. dq-side work is tracked separately.
- Do NOT change `_whale_attn_fwd_impl` dispatch (vault L1321-L1355). Fwd
  kernel is not on the critical path here — the target is fb, and bwd
  dkdv dominates the gap per the kineto profile in commit `f79d593`.

## Sequencing hard-stops
- If Lever 0 (inline-Δ mask-split) shows a regression on ANY of the 4
  shapes, this stacked leg does not proceed.
- If Lever A (autotune expand) shows >5% regression vs Lever 0 end-state,
  roll back the config list change before proceeding to B.
- If Lever B (TMA loads) breaks numerics tolerance, keep the env knob OFF
  and proceed to C without B.
- If Lever C (persistent) breaks numerics tolerance, keep the env knob OFF
  and report whichever of (0+A) or (0+A+B) is the best end-state.
