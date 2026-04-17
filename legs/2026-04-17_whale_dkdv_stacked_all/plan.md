# Plan — whale dkdv stacked-all (4 levers)

Date: 2026-04-17
Goal: land all 4 levers on `_attn_bwd_dkdv_inline_delta_kernel` and verify
fb on the 4 primary shapes. This leg is a **coordination leg** — it does
NOT introduce new kernel edits by itself. Each lever lives in its own
sibling leg and lands independently. This leg only runs the final
full-stack bench once every upstream leg shows neutral-or-positive delta.

## Ordering principle
Apply levers in increasing order of risk. Bench after each. Bail if any
step regresses the primary fb or breaks numerics.

- **Risk rank 1 (lowest)** — Autotune expand (Lever A). Pure config list,
  no kernel edit, no dispatcher edit. Worst case: autotune picks the same
  config it already picks; new configs are wasted.
- **Risk rank 2** — Mask-split on inline-Δ (Lever 0). Kernel IR change, but
  identical numerics (causal predicate is compile-time-identical between
  the masking and unmasked phases; unmasked phase just skips the `tl.where`).
  Already proven on base `_attn_bwd_dkdv_kernel` (vault L455-L572).
- **Risk rank 3** — TMA loads (Lever B). Moderate kernel edit. Risk:
  descriptor init overhead on small-tile shapes, swizzle mismatch, TMA
  allocator not initialized when env is flipped mid-process.
  Env-gated (`WHALE_DKDV_TMA_LOADS=1`) so default behavior is unchanged.
- **Risk rank 4 (highest)** — Persistent (Lever C). New kernel path with
  new grid semantics. Risk: wrong work-tile ordering, register spill,
  interaction with autotune. Env-gated (`WHALE_DKDV_PERSISTENT=1`).

## Canonical sequencing (5 phases)

**All phases run on 1×H100 at the 4 shapes from `tracked_env.sh`.**

### Phase 0 — Baseline capture
- Working tree: clean `whale/2026-04-16_pod_autoresearch` branch.
- Run `bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase0_baseline`.
- Expected: whale fb ~1.7 ms @ (2,8192,8,4,64). Record per-shape fb.
- Artifact: `evidence/phase0_baseline_<ts>_*.json`.

### Phase 1 — Lever A: autotune expand
- Land `legs/2026-04-17_whale_dkdv_autotune_expand/vault_patch.md` on
  `vault/whale_kernel_triton.py::_bwd_kv_inline_configs` (L109) and its
  TMA sibling `_bwd_kv_inline_tma_configs` (L128). Config-list only edit.
- **Autotune-cache flush is REQUIRED**: `WHALE_FLUSH_TRITON_CACHE=1`.
- Run `bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase1_autotune`.
- Gate: fb at (2,8192,8,4,64) must be <= phase0 fb × 1.02 (i.e., no more
  than 2% slower). If regressed, revert the config-list change before
  phase 2.
- Artifact: `evidence/phase1_autotune_<ts>_*.json`.

### Phase 2 — Lever 0: inline-Δ mask-split
- Land `legs/2026-04-16_whale_dkdv_early_exit_inline_delta/vault_patch.md`
  on `_attn_bwd_dkdv_inline_delta_kernel` (vault L587-L676) and mirror it
  on `_attn_bwd_dkdv_inline_delta_tma_kernel` (vault L686-L788). Same split
  pattern as the base kernel already at vault L496-L564.
- Flush autotune cache: `WHALE_FLUSH_TRITON_CACHE=1`.
- Run `bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase2_masksplit`.
- Gate: fb at (2,8192,8,4,64) must be strictly less than phase1 fb.
  Expected delta: −150 to −250 µs.
- If regressed or neutral, abort the entire stacked leg.
- Artifact: `evidence/phase2_masksplit_<ts>_*.json`.

### Phase 3 — Lever B: TMA loads (env-gated)
- Land `legs/2026-04-17_whale_dkdv_tma_loads/vault_patch.md` (stub — leg
  currently empty; this leg's `stacking_analysis.md` pins the exact
  insertion points).
- Edit is additive: pointer-based tile loads remain in the kernel under
  `if not DKDV_TMA_LOADS:` else TMA-descriptor loads. Env knob routes
  `WHALE_DKDV_TMA_LOADS=1` through to the autotune constexpr.
- Flush autotune cache.
- Run `bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase3_tma` with
  `WHALE_DKDV_TMA_LOADS=1` exported.
- Gate: fb at (2,8192,8,4,64) must be <= phase2 fb × 1.00, AND numerics
  tolerance holds (max |Δ| dQ/dK/dV <= 1e-2 in bf16).
- If regressed or numerics break, leave `WHALE_DKDV_TMA_LOADS=0`
  (default) and proceed to phase 4. The code change stays in place but
  dormant.
- Artifact: `evidence/phase3_tma_<ts>_*.json`.

### Phase 4 — Lever C: persistent (env-gated, highest risk)
- Land `legs/2026-04-17_whale_dkdv_persistent/vault_patch.md` (not yet
  prepped; stub for future work). Adds
  `_attn_bwd_dkdv_inline_delta_persistent_kernel` + dispatcher branch.
- Flush autotune cache.
- Run `bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase4_persistent`
  with `WHALE_DKDV_PERSISTENT=1` and `WHALE_DKDV_TMA_LOADS=<phase3-winner>`.
- Gate: fb at (2,8192,8,4,64) must be <= phase3 fb × 1.00, AND numerics
  tolerance holds.
- If regressed, default knob OFF and report phase3 (or phase2) as
  best-of-stack.
- Artifact: `evidence/phase4_persistent_<ts>_*.json`.

### Phase 5 — Final stacked bench (optional, only if phases 1-4 clean)
- Re-run all 4 shapes with both env knobs ON and fresh autotune cache.
- Produce `RESULTS.md` delta table. 1 row per shape × per phase.

## Expected end-state delta table (target numbers)

| Phase            | (2,8192,8,4,64)   | (2,3072,8,4,64)   | (2,2048,8,4,64)   | (2,8192,8,4,128)  |
|------------------|-------------------|-------------------|-------------------|-------------------|
| 0 baseline       | 1.70 ms (current) | TBD               | TBD               | TBD               |
| 1 +autotune      | 1.60-1.67 ms      | -3-5%             | -3-5%             | -3-5%             |
| 2 +mask-split    | 1.35-1.50 ms      | -5-10%            | -5-10%            | -8-15%            |
| 3 +TMA loads     | 1.10-1.30 ms      | -10-20%           | -10-20%           | -15-25%           |
| 4 +persistent    | 0.95-1.10 ms      | -15-25%           | -15-25%           | -25-35%           |
| Target (<=0.5×FA3)| 0.51 ms          | —                 | —                 | —                 |

**Honest caveat**: phase-4 end-state likely still misses 0.51 ms by
0.4-0.6 ms. Gap closure beyond 4-lever stack requires
`_attn_bwd_fused_kernel` (dkdv+dq+atomic-dQ in one launch) which is
tracked by `legs/2026-04-16_whale_bwd_persistent_atomic/`.

## Bailout rules
- Numerics fail at any phase → revert that lever, mark leg as "partial"
  and land the phases that did pass.
- Compile failure (Triton IR crash, NVGPUWarpSpecialization) → the
  offending lever stays dormant. Keep `range(...)` not `tl.range(...)`
  on the inline-Δ family — the base kernel at vault L455-L572 already
  established this workaround and it must not be undone.
- Autotune picks a bad config after cache flush → check the picked
  `(BM, BN, num_warps, num_stages, maxnreg)` against the flagged bad set
  from `triton_gotchas_atomic_add.md` (maxnreg>=224 corrupts
  `tl.atomic_add` by ~19000×; this leg does NOT use atomic_add but the
  gotcha is a reminder that maxnreg tuning can produce silent corruption).

## Launch command sequence (single-GPU, serial — see MEMORY)

```bash
# 0. Preflight (required by CLAUDE.md startup safety protocol)
pwd
git remote -v
git rev-parse --abbrev-ref HEAD
test -f legs/2026-04-17_whale_dkdv_stacked_all/run.sh
test -f vault/whale_kernel_triton.py
test -f scripts/whale_attention_bench.py

# 1. Baseline
bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase0_baseline

# 2. Apply Lever A, then bench
# (patch vault per legs/2026-04-17_whale_dkdv_autotune_expand/vault_patch.md)
WHALE_FLUSH_TRITON_CACHE=1 \
  bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase1_autotune

# 3. Apply Lever 0, then bench
# (patch vault per legs/2026-04-16_whale_dkdv_early_exit_inline_delta/vault_patch.md
#  AND mirror the same split onto _attn_bwd_dkdv_inline_delta_tma_kernel)
WHALE_FLUSH_TRITON_CACHE=1 \
  bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase2_masksplit

# 4. Apply Lever B, then bench (env-gated)
# (patch vault per legs/2026-04-17_whale_dkdv_tma_loads/vault_patch.md)
WHALE_FLUSH_TRITON_CACHE=1 WHALE_DKDV_TMA_LOADS=1 \
  bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase3_tma

# 5. Apply Lever C, then bench (env-gated)
# (patch vault per legs/2026-04-17_whale_dkdv_persistent/vault_patch.md)
WHALE_FLUSH_TRITON_CACHE=1 WHALE_DKDV_PERSISTENT=1 WHALE_DKDV_TMA_LOADS=1 \
  bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase4_persistent
```

## Evidence to capture per phase
- `pwd`, `git rev-parse --short HEAD`, `sha256sum vault/whale_kernel_triton.py`
- Picked autotune config per kernel per shape (Triton cache inspection
  or `TRITON_PRINT_AUTOTUNING=1`).
- whale fb + sdpa fb per shape.
- Numerics JSON (from `bench_numerics.py`).
- Kineto trace JSON for the primary shape at every phase (compare to the
  existing 157 µs GPU gap noted in commit `f79d593`).
