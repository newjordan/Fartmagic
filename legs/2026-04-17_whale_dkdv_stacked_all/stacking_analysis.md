# Stacking analysis — 4-lever interference map

Date: 2026-04-17
Target kernel: `_attn_bwd_dkdv_inline_delta_kernel` (vault
`whale_kernel_triton.py` L587-L676).
Reference sibling: `_attn_bwd_dkdv_inline_delta_tma_kernel` (vault L686-L788).
Baseline sibling with mask-split already landed: `_attn_bwd_dkdv_kernel`
(vault L455-L572).

## Fact — current vault state of each lever (line-cited)

### Lever 0 (mask-split) — LANDED only on base kernel
`vault/whale_kernel_triton.py` L496-L564 (from `_attn_bwd_dkdv_kernel`):
```
496:    if IS_CAUSAL:
497:        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
498:        m_masking_max = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)
499:    else:
500:        m_start_block = 0
501:        m_masking_max = 0
502:    m_end_block = tl.cdiv(T_MAX, BLOCK_M)
503:
504:    for hg in range(group):
505:        h = kv_h * group + hg
506:
507:        if IS_CAUSAL:
508:            m_mask_end = tl.minimum(m_masking_max, m_end_block)
509:            for m_block in range(m_start_block, m_mask_end):
...
535:            unmasked_start = m_mask_end
536:        else:
537:            unmasked_start = m_start_block
538:
539:        for m_block in range(unmasked_start, m_end_block):
```
The inline-Δ kernel at L587-L676 still shows the **pre-split** pattern:
```
628:    if IS_CAUSAL:
629:        m_start_block = (pid_n * BLOCK_N) // BLOCK_M
630:    else:
631:        m_start_block = 0
632:    m_end_block = tl.cdiv(T_MAX, BLOCK_M)
633:
634:    for hg in range(group):
635:        h = kv_h * group + hg
636:        for m_block in range(m_start_block, m_end_block):
```
**Inference**: phase 2 of the plan MUST land the mask-split on both
`_attn_bwd_dkdv_inline_delta_kernel` (L628-L632, L636-L664 loop body) AND
`_attn_bwd_dkdv_inline_delta_tma_kernel` (L739-L784 parallel structure).
Otherwise phase 3 (TMA loads) and phase 4 (persistent) inherit an unsplit
M-loop and lose the phase-2 gain for those env knobs.

### Lever A (autotune expand) — prepped, pure config
`vault/whale_kernel_triton.py` L109-L125 (`_bwd_kv_inline_configs`):
- Current grid: 4 (BM,BN) × 2 warps × 2 stages, all at `maxnreg=224` +
  1 escape hatch `(64,64,4,2,maxnreg=224)`. Total: 17 configs.
- Proposed expansion (tracked in `legs/2026-04-17_whale_dkdv_autotune_expand/`):
  - add `num_stages=5` to the grid (extends the pipeline depth search),
  - add `(256, 128)` and `(128, 256)` wide-tile rows,
  - add a `maxnreg=192` variant per row to let autotune choose against
    register-pressure if the TMA pipeline triggers more spill.
- Autotune key `(D, IS_CAUSAL, NUM_HEADS, NUM_KV_HEADS, T_MAX)` unchanged.

### Lever B (TMA loads) — prepped, leg dir empty
`legs/2026-04-17_whale_dkdv_tma_loads/` currently exists but is empty
(`ls -la` shows `. ..` only). Expected vault patch pins the following
insertion points inside `_attn_bwd_dkdv_inline_delta_kernel`:

1. Add constexpr `DKDV_TMA_LOADS: tl.constexpr` to the kernel signature
   (vault L587-L606 body params). Default: 0.
2. At vault L617-L621 (K/V pointer loads):
   ```
   617:    k_mask = (offs_n[:, None] < T_MAX) & (offs_d[None, :] < D)
   618:    k_ptrs = K + b * stride_kb + kv_h * stride_kh + ...
   619:    v_ptrs = V + b * stride_vb + kv_h * stride_vh + ...
   620:    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
   621:    v = tl.load(v_ptrs, mask=k_mask, other=0.0)
   ```
   Wrap in `if DKDV_TMA_LOADS:` / `else:` branch, with the TMA path using
   `tl.make_tensor_descriptor(...).load([pid_n * BLOCK_N, 0])` — exact
   pattern from the TMA sibling kernel at L714-L732.
3. At vault L642-L647 (Q/DO/O tile loads inside M-loop), same
   `if DKDV_TMA_LOADS:` wrap.
4. Dispatcher in `_whale_attn_bwd_impl` at L1492-L1505 must thread the
   env knob through: read `WHALE_DKDV_TMA_LOADS` and pass as the
   `DKDV_TMA_LOADS` constexpr, then call `_ensure_tma_allocator()` when
   it's 1.

### Lever C (persistent) — prepped, leg dir does NOT YET exist
`legs/2026-04-17_whale_dkdv_persistent/` is not in the tree. This leg
pins the design here so the authoring agent can scaffold it without
re-deriving the structure:
- New kernel `_attn_bwd_dkdv_inline_delta_persistent_kernel`, signature
  mirrors the non-persistent kernel plus `NUM_SMS: tl.constexpr` and
  `TILES: tl.constexpr = cdiv(T_MAX, BLOCK_N) * B * NUM_KV_HEADS`.
- Grid: `(NUM_SMS,)`. Each program iterates `for tile_id in range(pid, TILES, NUM_SMS)`.
- Per tile, decode `(b, kv_h, pid_n)` from `tile_id`, execute the
  existing dkdv body.
- Env-gated by `WHALE_DKDV_PERSISTENT=1`; dispatcher branch in
  `_whale_attn_bwd_impl` L1474-L1505.

## Inference — pairwise interference map

### Pair (0, A): mask-split × autotune expand
**Does mask-split invalidate picked autotune configs?**
- Autotune key is shape-only (`D, IS_CAUSAL, NUM_HEADS, NUM_KV_HEADS, T_MAX`).
  Mask-split produces new IR with same key → picked config from unsplit IR
  is stale.
- **Mitigation**: `WHALE_FLUSH_TRITON_CACHE=1` before phase 2.
- No logical interference; autotune search re-runs over the expanded list.

### Pair (0, B): mask-split × TMA loads
**Does TMA break the mask-split?**
No, but with a subtle condition.
- The mask-split produces TWO inner loops with identical tile shapes
  (both load `BLOCK_M × BLOCK_D` Q/DO/O tiles). A TMA
  `make_tensor_descriptor` at `block_shape=[BLOCK_M, D]` is invariant to
  which inner loop consumes the load.
- **BUT**: the current inline-Δ kernel creates Q/DO/O pointers fresh per
  m_block (L642-L646). Under TMA, `make_tensor_descriptor` is expensive
  and must be hoisted outside the m_block loop, per the existing TMA
  kernel at L747-L758 where `Q_desc`/`O_desc`/`DO_desc` are created once
  per `hg` iteration and indexed by `[start_m, 0]` per m_block (L764-L766).
- Under mask-split, the descriptor must be created **once per `hg`** and
  then used by BOTH the masking loop AND the unmasked loop. If the agent
  authoring Lever B naively creates the descriptor inside each inner
  loop, the split will duplicate descriptor inits and regress.
- **Pin**: Lever B vault patch must place descriptor creation between the
  `for hg in range(group):` line and the `if IS_CAUSAL:` mask-split
  header, so both inner loops share descriptors. Reference layout is the
  TMA sibling L745-L758 (creation) + L759 (m_block loop).

### Pair (0, C): mask-split × persistent
No structural interference.
- The persistent loop wraps `(b, kv_h, pid_n)` iteration. The mask-split
  lives strictly inside the per-tile body.
- Only concern: `m_masking_max = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)`
  depends on `pid_n`, which is now loop-carried in persistent. It must
  be recomputed inside the tile loop, not hoisted. This is the natural
  placement but must be explicit in the vault patch.

### Pair (A, B): autotune expand × TMA loads
**Highest interference pair for compile correctness.**
- TMA requires `BLOCK_D == D` (vault L680 comment:
  "Requires BLOCK_D == D"). The autotune expansion as written does NOT
  vary `BLOCK_D` (it's driven by `triton.next_power_of_2(D)` at dispatch
  time, not per-config), so this holds.
- `maxnreg=192` variant added by Lever A is safe under TMA (TMA path
  does not use `tl.atomic_add`, so the 224-threshold gotcha is not in
  play — see `triton_gotchas_atomic_add.md`). BUT: TMA descriptor init
  uses additional registers; maxnreg=192 under TMA may force spill.
  Autotune will resolve this; we just need the search space to include
  both 192 and 224 to let Triton pick.
- Lever A must update `_bwd_kv_inline_tma_configs` (L128) in parallel
  with `_bwd_kv_inline_configs` (L109). If only the non-TMA list is
  expanded, Lever B with TMA on would use the narrow list and may pick
  a suboptimal config.

### Pair (A, C): autotune expand × persistent
- Persistent kernel has different register pressure profile (holds more
  state across tiles). Its autotune config set should NOT reuse
  `_bwd_kv_inline_configs` — it needs a separate `_bwd_kv_inline_persistent_configs`
  with a narrower range (prefer `maxnreg=192` or lower, stages=2-3).
- **Pin**: Lever C vault patch must introduce a new config fn, NOT reuse
  the expanded list. Otherwise Lever A's wide sweep (256x128) under
  persistent will trigger compile OOM or massive spill.

### Pair (B, C): TMA × persistent
**The operational hazard.**
- Under persistent, each SM processes N tiles sequentially. The TMA
  descriptor for K/V depends on `(b, kv_h)` which changes per tile;
  descriptors must be rebuilt inside the persistent loop at tile-start.
- Under persistent, the Q/DO/O descriptors depend on `(b, h)` and must
  be rebuilt per (tile, hg) pair.
- **Descriptor init cost**: `tl.make_tensor_descriptor` compiles to a
  Hopper TMA descriptor setup in SASS. Measured in the TMA sibling
  kernel tests (commit `2e07aea` — "add opt-in TMA kernel; confirms
  Triton 3.6 ceiling at 72us") to cost ~1-2 µs per descriptor init.
  Over 128 tiles × 3 Q-heads-per-group × 3 descriptors = 1152 inits ×
  1.5 µs = 1.7 ms of pure descriptor overhead if done naively.
- **Mitigation**: under `WHALE_DKDV_PERSISTENT=1 AND WHALE_DKDV_TMA_LOADS=1`,
  the persistent kernel must hoist K_desc/V_desc creation to the top of
  each tile and reuse for that tile's M-loop. Per-Q-head descriptors
  (Q/O/DO) must be created once per (tile, hg) and used by both the
  masking and unmasked inner loops.
- If this hoisting is missed, Lever C+B together will REGRESS past the
  Lever-B-only baseline. The gating in phase 4 of the plan catches this.

## Proposal — interference-safe merge recipe

1. Phase 1 (Lever A): edit `_bwd_kv_inline_configs` AND
   `_bwd_kv_inline_tma_configs` in the same vault diff. No kernel change.
2. Phase 2 (Lever 0): edit BOTH `_attn_bwd_dkdv_inline_delta_kernel` AND
   `_attn_bwd_dkdv_inline_delta_tma_kernel` to add the masking/unmasked
   split. Mirror the pattern already established in base
   `_attn_bwd_dkdv_kernel` L496-L564.
3. Phase 3 (Lever B): introduce constexpr gate `DKDV_TMA_LOADS` on
   `_attn_bwd_dkdv_inline_delta_kernel`. Descriptor creation MUST sit
   between the `for hg in range(group):` line and the mask-split `if
   IS_CAUSAL:` header so both phases share descriptors.
4. Phase 4 (Lever C): new kernel with its own config fn. Tile loop wraps
   the existing (post-phase-3) body. TMA descriptor hoisting inside the
   tile loop, not outside.

## Non-goals
- Do NOT change the autotune key. Dispatch-time key mutation invalidates
  cache in surprising ways.
- Do NOT attempt `tl.range(..., warp_specialize=True)` on any inline-Δ
  kernel — Triton 3.6's NVGPUWarpSpecialization pass crashes on this
  kernel family (established in the early-exit leg's hypothesis and the
  existing comment in the sibling vault_patch.md). Keep bare `range(...)`.
- Do NOT enable `maxnreg>=224` on any kernel that uses `tl.atomic_add`
  (reminder from `triton_gotchas_atomic_add.md`; the dkdv family does not
  use atomic_add, so this is a reminder only — the persistent-atomic
  dkdv+dq path at vault L810 is NOT in scope for this leg).
