# Hypothesis — whale dkdv (inline-delta variant) SeparateMaskingIterations

Date: 2026-04-16
Parent: `vault/whale_kernel_triton.py` @ `_attn_bwd_dkdv_inline_delta_kernel` (L559-L644)
Sibling: `legs/2026-04-16_whale_dkdv_early_exit/` (base `_attn_bwd_dkdv_kernel` patch)
Primary shape: B=2, T=8192, H=8, KV=4, D=64 (forced via `WHALE_FUSED_DELTA_T_MAX=8192`)
Secondary shapes (auto-routed): B=2, T=2048 / 3072, H=8, KV=4, D=64
Hardware: 1×H100 SXM

## Fact — kernel routing
`vault/whale_kernel_triton.py` L1254-L1257:
```
variant = os.environ.get("WHALE_BWD_VARIANT", "auto")
fused_delta_t_max = int(os.environ.get("WHALE_FUSED_DELTA_T_MAX", "3072"))
if variant == "auto":
    use_fused_delta = T <= fused_delta_t_max
```
And L1324-L1355: when `use_fused_delta and not use_tma_dkdv`, the dispatcher
routes into `_attn_bwd_dkdv_inline_delta_kernel`. Default `auto` mode therefore
picks the inline-delta path for T <= 3072. To exercise it on the long-T
primary shape (T=8192), this leg sets `WHALE_FUSED_DELTA_T_MAX=8192` in
`tracked_env.sh` / `run.sh`.

## Fact — what the inline-delta kernel currently does
`vault/whale_kernel_triton.py` L600-L632 (M-loop body, inline-Δ path):
```
if IS_CAUSAL:
    m_start_block = (pid_n * BLOCK_N) // BLOCK_M
else:
    m_start_block = 0
m_end_block = tl.cdiv(T_MAX, BLOCK_M)

for hg in range(group):
    h = kv_h * group + hg
    for m_block in range(m_start_block, m_end_block):
        ...
        if IS_CAUSAL:
            p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX) & (offs_m_cur[:, None] >= offs_n[None, :])
        else:
            p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX)
        p = tl.where(p_mask, p, 0.0)
        ...
```
Identical structural waste as the base `_attn_bwd_dkdv_kernel`: the causal
predicate `offs_m_cur[:, None] >= offs_n[None, :]` is recomputed on every
M-block, even though only the 1-2 M-blocks straddling the diagonal actually
need it.

## Inference — why this is a parallel target
The inline-delta variant differs from the base only in (a) inlining Δ via
the on-the-fly `tl.sum(o * do, axis=1)` (replacing the preprocess pass) and
(b) dropping the precomputed `delta` HBM load. The redundant per-element
causal mask is identical in both kernels. If the base patch saves N µs on
T=8192, the inline-delta patch is expected to save approximately the same
amount on T<=3072 and on T=8192 when forced via env knob.

## Proposal — apply the same SeparateMaskingIterations split
Mirror the patch from `legs/2026-04-16_whale_dkdv_early_exit/vault_patch.md`:
- Compute `m_masking_max = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)` for the
  causal case after L604.
- Split the M-loop into a masking phase (`m_start_block .. m_mask_end`) that
  applies the causal `tl.where`, and an unmasked phase (`unmasked_start ..
  m_end_block`) that drops the causal predicate.
- Use plain `range(...)` (NOT `tl.range(...)`). Triton 3.6's
  `NVGPUWarpSpecialization` pass crashes on this kernel — see the existing
  loop on L608 already uses bare `range`.

## Success criterion
On the primary shape with `WHALE_FUSED_DELTA_T_MAX=8192` and
`WHALE_BWD_VARIANT=auto` (so the inline-delta path is forced at T=8192):
fwd+bwd latency drops by >= 30 us vs the pre-patch baseline taken in the
same gate script, with numerics (max |Δ|, relative) within current
tolerance (max |Δ| dQ/dK/dV <= 1e-2 in bf16).

On the secondary shapes (T=2048, T=3072) the inline-delta path is on by
default; expected savings are smaller in absolute µs but the relative
improvement should be visible.

## Non-goals
- Do not patch `_attn_bwd_dkdv_inline_delta_tma_kernel` (the TMA variant) in
  this leg. That is a child leg if both pure-Triton patches show wins.
- No autotune-config edits. The autotune key
  `(D, IS_CAUSAL, NUM_HEADS, NUM_KV_HEADS, T_MAX)` is unchanged. Flush the
  Triton cache once before the first benched run via
  `WHALE_FLUSH_TRITON_CACHE=1` so stale configs picked for the pre-patch IR
  cannot bias the comparison.

## Sequencing
This leg is **prepped** but **must not be applied** until the base
early-exit leg shows a positive delta in its `after` bench. If the base
patch regresses or no-ops, the inline-delta patch should not land.
