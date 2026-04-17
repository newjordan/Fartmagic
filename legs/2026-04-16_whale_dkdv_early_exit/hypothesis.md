# Hypothesis — whale dkdv SeparateMaskingIterations (Lever: fine-grained early-exit)

Date: 2026-04-16
Parent: `vault/whale_kernel_triton.py` @ `_attn_bwd_dkdv_kernel` (L427-L517)
Primary shape: B=2, T=8192, H=8, KV=4, D=64 (matches Lever A bench)
Hardware: 1×H100 SXM

## Fact — what whale currently does
`vault/whale_kernel_triton.py` L470-L474:
```
if IS_CAUSAL:
    m_start_block = (pid_n * BLOCK_N) // BLOCK_M
else:
    m_start_block = 0
m_end_block = tl.cdiv(T_MAX, BLOCK_M)
```
This is the coarse early-exit: Q-rows strictly above the KV tile are skipped.
That matches FA3's `BlockMN::get_m_block_min_max` in
`third_party/flash-attention/hopper/block.h` L83-L101 exactly
(for seqlen_q == seqlen_k, no window).

## Fact — what whale currently does inside the loop
L500-L504, on **every** iteration of the M-loop:
```
if IS_CAUSAL:
    p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX) & (offs_m_cur[:, None] >= offs_n[None, :])
else:
    p_mask = row_mask[:, None] & (offs_n[None, :] < T_MAX)
p = tl.where(p_mask, p, 0.0)
```
The causal predicate `offs_m_cur[:, None] >= offs_n[None, :]` is recomputed and
applied to P on every M-block — including M-blocks that are strictly below the
diagonal, where the predicate is unconditionally true for every element.

## Fact — what FA3 does
`third_party/flash-attention/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp`
L1003-L1033 splits the M-loop into two phases when
`SeparateMaskingIterations` is true:

1. Masking phase for `m_block < min(m_block_max, m_block_masking_max)` —
   apply causal mask. `m_block_masking_max` (L1008):
   ```
   ((n_block + 1) * kBlockN - 1 + seqlen_q - seqlen_k - window_size_right) / kBlockM + 1
   ```
2. Unmasked phase for `m_block < m_block_max_before_local_mask` — no mask.
   Comment at L1003-1004: *"Not necessary for hdim 128 but for hdim 64 this
   helps quite a bit to not have to do causal masking for most of the
   iterations."* That matches our primary shape (D=64).

## Inference — why this is structural
At B=2, T=8192, BLOCK_M=BLOCK_N=128, each N-block visits ~64 M-blocks.
Only the 1–2 M-blocks straddling the diagonal actually need the mask.
The remaining ~60+ iterations unnecessarily:
  - materialize `p_mask` (three elementwise predicates and two ANDs)
  - execute `tl.where` on a BLOCK_M×BLOCK_N float tile
  - consume register pressure that may spill other reuse

For D=64 the dkdv kernel is arithmetic-light; removing redundant mask work
directly shrinks its dominant path. Kineto shows dkdv at 747us vs FA3 total
bwd 924us, so the absolute target for savings is ~50-150us.

## Proposal — exact lines to change
`vault/whale_kernel_triton.py` `_attn_bwd_dkdv_kernel`:

- Compute a scalar `m_masking_max` after line 475 that matches FA3's
  `m_block_masking_max` (for causal, seqlen_q==seqlen_k, no window):
  `m_masking_max = ((pid_n + 1) * BLOCK_N + BLOCK_M - 1) // BLOCK_M`
  (equivalently `cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)`).
- Gate the `tl.where(p_mask, p, 0.0)` by a compile-time-ish branch:
  when `IS_CAUSAL` is True AND `m_block < m_masking_max`, apply the mask.
  Otherwise, apply only `row_mask & (offs_n < T_MAX)` (the sequence-bounds
  mask, which is only non-trivial on the tail N-block / tail M-block).
- Even better: split the M-loop into two literal `for` ranges
  (`m_start_block..m_masking_max` with mask, and `m_masking_max..m_end_block`
  without), so Triton specializes each loop body and the compiler can drop
  the predicate build entirely on the tail.

Note: `row_mask` and `offs_n < T_MAX` are also compile-time trivial when
`T_MAX % BLOCK_M == 0` and `T_MAX % BLOCK_N == 0` (our primary shape: 8192
is a multiple of 128). Second follow-up: also gate the tail load masks on
`T_MAX_IS_DIVISIBLE: tl.constexpr` to let the unmasked path fully elide.

## Success criterion
Fwd+bwd latency on the primary shape (B=2, T=8192, H=8, KV=4, D=64) drops
by >= 30us vs the pre-patch baseline taken in the same gate script, with
numerics (max |Δ|, relative) within the current numerics tolerance.

## Non-goals
- No training change. This is a kernel-only micro-benchmark leg.
- No change to fwd or preprocess kernels.
- No change to autotune configs; if a config change becomes necessary after
  the mask-split, that is a child leg.
