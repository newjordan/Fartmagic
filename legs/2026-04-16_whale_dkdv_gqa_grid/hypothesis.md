# Hypothesis — whale dkdv GQA-to-Grid (Lever: parallelize Q-head over the launch grid)

Date: 2026-04-16
Parent: `vault/whale_kernel_triton.py` @ `_attn_bwd_dkdv_kernel` (L427-L544)
Primary shape: B=2, T=8192, H=8, KV=4, D=64 (group = H/KV = 2)
Hardware: 1×H100 SXM (132 SMs)

## Fact — what whale currently does (vault L447-L451, L476-L477)
```python
pid_n = tl.program_id(0)
bkv = tl.program_id(1)
b = bkv // NUM_KV_HEADS
kv_h = bkv % NUM_KV_HEADS
group = NUM_HEADS // NUM_KV_HEADS
...
for hg in range(group):
    h = kv_h * group + hg
    ... # M-loop accumulates dk, dv across all Q-heads in the group
```
Grid (vault L1396): `(triton.cdiv(T, META["BLOCK_N"]), B * KV)`.
On the primary shape with BLOCK_N=128: `(64, 8)` = **512 programs** for the
dkdv kernel. With group=2, each program walks the M-loop **twice** (once per
Q-head), accumulating into the same `dk`/`dv` registers, then stores once.

## Fact — what FA3 does (GQA path)
`third_party/flash-attention/hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp`
L456: `int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);`
Grid is launched with `bidh = full Q-head index` (NUM_HEADS, not NUM_KV_HEADS).
Each CTA produces a **per-Q-head fp32 partial** dK/dV into a workspace
(`dk_accum_ptr` of shape `[B, h_q, T, D]`). After the main kernel finishes,
`PostprocessKerneldKV` (`flash_bwd_launch_template.h` L284-L292) launches at
`(num_n_blocks, h_k, b)` and reduces the per-Q-head partials into the final
`[B, h_k, T, D]` dK/dV tensors.

This is the same structural pattern that whale already implements in
`_attn_bwd_dkdv_split_h_kernel` (vault L918-L1004) plus the post-kernel
reduce in `_whale_attn_bwd_impl` (vault L1372-L1394).

## Fact — split-H was already tested in this repo, and lost
`legs/2026-04-16_whale_pod_autoresearch/RESULTS.md` L78-L82, verbatim:
> Split-H dK/dV — tested with `tl.atomic_add` (broken: values inflated
> ~24 000× in Triton 3.6 on H100 for 2-D fp32 atomic tiles) and with
> per-Q-head partial buffer + torch reduction (correct, but ~20% slower
> than the non-split kernel due to extra fp32 traffic). Behind
> `WHALE_BWD_SPLIT_H=1` for archival.

That benchmark ran the dkdv kernel (no inline-Δ) at the previous headline
shape (B=4, T=2048, H=8, KV=4, D=64). The conclusion was: **the post-kernel
reduce of `[B, T, H, D] fp32` partials costs more than the M-loop
serialization saves at that shape.**

## Inference — why this is worth re-testing now, conditionally
- `fact`: The 2026-04-16 split-H bench used **B=4, T=2048**. Wave count =
  2 N_blocks × 4 B × 4 KV = 32 programs; with H100's 132 SMs, the kernel
  was already SM-bound by **occupancy**, not by per-program M-loop length.
  Splitting the inner `for hg` into the grid doubles the program count to
  64 — still well below 132 — so the only thing it adds is the reduction
  cost. That made the lever a net loss at that shape.
- `inference`: The current headline shape is **B=2, T=8192**. Wave count =
  64 N_blocks × 2 B × 4 KV = **512 programs** (already saturating SMs many
  times over). Splitting Q-head into the grid pushes that to 1024 programs,
  i.e. the dkdv launch becomes ~7.7 waves instead of ~3.9 waves. **This
  does not increase concurrency on a saturated grid; it only adds reduce
  cost.**
- `inference`: Therefore, on the current primary shape, the GQA-to-grid
  lever is **structurally unlikely to win** vs the existing pure dkdv path.
  The pod_autoresearch result (~20% slower) is expected to **get worse**,
  not better, at T=8192.
- `inference`: The lever could only win on a **launch-bound regime**: small
  T, small B, small KV. E.g. B=1, T=1024, H=8, KV=2 would give
  `8 × 1 × 2 = 16 programs` for the unsplit grid (well under 132 SMs);
  splitting into Q-head gives 64 programs, closer to one full wave. That
  shape is **not in this repo's training mix** (midnight III.v uses T=2048
  and longer; competition shape is T=8192).

## Proposal — fact-finding only, no expectation of headline win
Re-bench the **existing** `WHALE_BWD_SPLIT_H=1` path (no new kernel work)
on the four shapes from `legs/2026-04-16_whale_dkdv_early_exit`, plus
two artificial **launch-bound** shapes designed to expose the regime where
the lever could matter:
  - primary: B=2, T=8192, H=8, KV=4, D=64 (expected: split-H loses, ~20%)
  - secondary: B=2, T=4096, H=8, KV=4, D=64 (expected: split-H loses)
  - secondary: B=2, T=8192, H=8, KV=8, D=64 (group=1, lever is no-op,
    expect parity within noise; this is the control)
  - secondary: B=2, T=8192, H=8, KV=4, D=128 (D=128 increases compute
    per M-iter, may shift the trade)
  - launch-bound A: B=1, T=1024, H=8, KV=2, D=64 (group=4, programs per
    KV-stride small)
  - launch-bound B: B=1, T=2048, H=8, KV=2, D=64 (group=4)

If split-H wins on any production-relevant shape (the first four), promote
it to default-on at that shape range. If it only wins on the launch-bound
synthetic shapes, document the lever as **regime-bounded** and leave it
behind the existing flag.

If we want to push **further** than the existing split-H (i.e. the flagged
"expand the dK/dV output to shape [..., group, ...] and reduce in a tiny
post-kernel step" alternative): the existing `_attn_bwd_dkdv_split_h_kernel`
already does exactly that, with a per-Q-head fp32 workspace of shape
`[B, T, H, D]` reduced via `view(B, T, KV, group, D).sum(dim=3)` (vault
L1393-L1394). There is no additional layout to invent.

## Atomic_add concern (recorded for completeness)
If we tried to skip the post-kernel reduce and instead `tl.atomic_add` into
final dK/dV, we hit the documented Triton 3.6 gotcha: `maxnreg >= 224`
corrupts `tl.atomic_add` of 2-D fp32 tiles by ~19000-24000x (memory:
`triton_gotchas_atomic_add.md`; vault L177-L180). The current
`_bwd_kv_inline_configs()` uses `maxnreg=224` (vault L113-L122) — atomic
combine would force this to drop to `<= 192`, with its own latency cost,
**and** introduce serialized HBM atomics on the dK/dV tiles. Not a
candidate.

## Success criterion
- Primary: Fwd+bwd latency on B=2, T=8192, H=8, KV=4, D=64 with
  `WHALE_BWD_SPLIT_H=1` is **<= the non-split baseline** (i.e. the
  pod_autoresearch result is **no longer reproducible** on the current
  vault head, which has both the inline-Δ change and the early-exit lever
  in flight).
- If the primary criterion fails (split-H still loses on the headline
  shape), the leg succeeds as a **negation**: it confirms GQA-to-grid is
  not a headline-shape lever and rules it out, freeing budget for the
  next lever.

## Non-goals
- No new kernel work in this leg (no atomic_add, no new accumulator
  layout). Re-uses `_attn_bwd_dkdv_split_h_kernel` already in vault.
- No autotune-config change. The existing `_bwd_kv_split_h_configs()`
  list is what runs.
- No combination with the early-exit lever — that is a separate leg
  (`legs/2026-04-16_whale_dkdv_early_exit`); combining the two is a
  child leg only after both individual results are in.

## Fallback plan
- If the bench shows split-H wins on a production shape: write a follow-up
  leg that adds a shape gate to `_whale_attn_bwd_impl` (auto-pick split-H
  inside the regime, default off otherwise). DO NOT default-enable
  globally.
- If split-H loses on every shape we test: mark the lever closed. The
  `WHALE_BWD_SPLIT_H` flag remains for archival per the existing comment.
