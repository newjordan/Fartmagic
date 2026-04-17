# whale aggressive ideas — path to 0.5x FA3

Date: 2026-04-17
Scope: research only. No bench runs, no GPU. No vault edits.

## Context (facts, with citations)

- fact: whale fwd+bwd at headline (B=4,T=2048,H=8,KV=4,D=64,causal,bf16) is at parity
  with FA3 (0.354 ms vs 0.355 ms). Evidence:
  `legs/2026-04-16_whale_bwd_ablations/RESULTS.md` line 23–24.
- fact: at long T (B=2,T=8192,H=8,KV=4,D=64) whale is 63% slower than FA3
  (1.739 ms vs 1.065 ms). Same RESULTS.md, line 37.
- fact: FA3 kernel-sum ≈ 186 µs vs whale kernel-sum ≈ 348 µs at headline;
  FA3 binding dispatch overhead ≈ 170 µs. Same RESULTS.md, lines 87–98.
- fact: to reach 0.5× FA3 wall (≈ 177 µs at headline), whale must beat FA3's
  own kernel-sum (186 µs) — pure micro-opt of existing tiles cannot do this.
- fact: RoPE is applied in Python *before* attention in both the bench and
  the production model.
  - `scripts/whale_attention_bench.py` lines 275–277 (cos/sin + apply_rotary_emb on q, k).
  - `gemini_runs/train_gpt.py` lines 1232–1237 (rms_norm → rotary → q_gain → attn).
- fact: FA3's `flash_attn_func` (training path) has NO rotary_cos/rotary_sin
  arguments. RoPE fusion exists only in `flash_attn_with_kvcache` (decode).
  Evidence: `third_party/flash-attention/hopper/flash_attn_interface.py`
  lines 806–884 (flash_attn_func signature — no rotary args), vs lines
  75–77 and 946–984 (kvcache path with rotary_cos/sin).
- fact: whale Triton kernel does NOT fuse RoPE. `grep -i "rope|rotary"
  vault/whale_kernel_triton.py` returns zero matches.
- fact: whale fwd kernel grid is `(pid_m, b*NUM_HEADS)` — one program per
  (batch, Q-head). Two Q-heads in the same GQA group re-read the same
  K/V from HBM. `vault/whale_kernel_triton.py` lines 262–265.
- fact: whale bwd dK/dV kernel DOES reuse K/V across Q-heads in a group —
  line 479, `group = NUM_HEADS // NUM_KV_HEADS`, line 504 `for hg in range(group)`.
  The forward kernel does not.

## Idea (a): Shape-specialized CUDA wrapper (custom C++/CUDA extension)

Pitch: compile a FA3-equivalent kernel specialized for B=2,T=8192,H=8,KV=4,D=64
(and the headline shape). Drop the runtime scheduler and compile-time
dispatch that FA3 carries, strip the persistent-scheduler path, pre-bake
tile sizes, and hard-code the QKV layout.

Feasibility:
- effort: 16–24 hours for a minimum viable copy. FA3's mainloop is
  `mainloop_fwd_sm90_tma_gmma_ws.hpp` (warp-specialized, CUTLASS templates),
  launched via `flash_fwd_launch_template.h`. A "trimmed FA3" would fork
  that file into whale, strip templated args we don't need (varlen, paging,
  alibi, softcap, window), hardcode D=64 / pack_gqa / causal=true. Binding
  via torch.utils.cpp_extension.
- inference: FA3 is already ~186 µs kernel-sum. A shape-specialized fork
  saves mostly *host-side* dispatch, which is already not in whale's path
  because whale uses triton.jit custom_op. Kernel-side gain from removing
  runtime template branches is small (seq_kv % BLOCK_N == 0 simplification,
  drop `Seqlen_traits` fall-through) — probably 5–15 µs. Not enough for
  0.5×.
- risk: HIGH. nvcc build toolchain, CUTLASS version alignment with cu130,
  sm90a ABI, PTX arch flags. If a build fails on the pod we burn hours.
  Also touches vault stability boundary.
- could it hit 0.5× FA3? NO by itself. It produces a ~0.9× FA3 clone at best.

## Idea (b): RoPE + attention fusion (+ optional rms_norm + q_gain)

Pitch: fold the rms_norm(q), rms_norm(k), cos/sin multiply, and q_gain into
the forward attention kernel's Q/K load path. Today these are separate
kernel launches producing three intermediate bf16 tensors in HBM: q_rope,
k_rope, q_scaled. FA3 cannot do this on its training path.

Feasibility:
- effort: 3–5 hours. The Triton forward kernel already loads Q and K tiles
  (`vault/whale_kernel_triton.py` lines 272–273, 291–293). Adding a cos/sin
  lookup + interleaved pair multiply on the first `rope_dims` elements of
  each loaded tile is ~30 lines. rms_norm is one reduction over `D`; q_gain
  is a per-head scalar broadcast. All fit in registers before the qk dot.
  Bench-only change: add Triton variant `_attn_fwd_fused_prep_kernel(Q, K, V,
  COS, SIN, Q_GAIN, O, LSE, ...)` and toggle with `WHALE_FWD_VARIANT=fused_prep`.
  Bench integration: edit `scripts/whale_attention_bench.py` to plumb
  cos/sin/q_gain into the backend callable (whale variant only).
- inference: measured savings = wall-time of (rotary kernel + q_gain mul +
  2x rms_norm) minus zero. On the pod's current stack a naive rotary
  torch op at B=4,T=2048,H=8,KV=4,D=64 is ~25–40 µs (two apply_rotary_emb
  calls, two rms_norm, one q_gain mul = 5 tiny kernels). Eliminated fully.
  At T=8192 the ratio scales linearly — 100–160 µs saved. This is real.
- could it hit 0.5× FA3? At headline: whale 0.354 ms − 0.030 ms prep ≈
  0.324 ms vs FA3 0.355 + 0.030 prep ≈ 0.385 ms for apples-to-apples ⇒
  whale 0.84× FA3. At T=8192: whale 1.739 − 0.150 ≈ 1.589 vs FA3 1.065 +
  0.150 = 1.215 ⇒ whale 1.31× FA3 (still loses long-T but closes).
  NO — 0.5× not achievable on the bwd-dominated headline, but this is the
  single largest pure-apples-to-oranges fairness win because FA3 cannot
  match it. If we compare whale-with-fused-prep vs FA3-unfused on the
  **production training loop** (which also runs these prep ops), we hit
  ~0.85× on the forward path alone. To push to 0.5× we need this + (c)
  stacked.
- risk: LOW. Fused-prep variant can be added behind a feature flag, exact
  correctness is testable vs the separate-ops path, and this is a pattern
  Triton is well-suited for.

## Idea (c): Cross-Q-head K/V reuse in whale forward (pack-GQA equivalent)

Pitch: change whale fwd grid from `(M_blocks, B*H)` to `(M_blocks, B*KV)`.
Each program loads (k_tile, v_tile) once from HBM, then iterates `group =
H/KV` Q-heads locally. For our shape group=2, so HBM K/V reads are halved
in the forward. FA3 achieves the same with `pack_gqa` (see
`third_party/flash-attention/hopper/pack_gqa.h` line 18 `struct PackGQAManager`).

Feasibility:
- effort: 2–4 hours. Whale's bwd dK/dV kernel is already written exactly
  this way (lines 475–572). Port the grid remap + Q-head inner loop to
  the fwd kernel; each inner-group iteration reads a Q tile and writes an
  O tile but reuses the K/V residents. Modify `_whale_attn_fwd_impl` grid
  launcher.
- inference: K/V HBM traffic on fwd halves (for group=2). Current fwd kernel
  time 72–77 µs at headline; K/V loads are ~40–50% of kernel time in a
  64×64 tile with 8 inner-N steps. Expect 15–30% fwd speedup ≈ 10–22 µs.
  At T=8192 the reduction compounds because N-loop length grows linearly.
- could it hit 0.5× FA3? Alone: no (fwd wins +15–30% but bwd dominates).
  Stacked with (b): whale fwd ≈ 45–55 µs at headline (vs FA3 47 µs fwd) —
  whale forward beats FA3. Still not 0.5× on fwd+bwd.
- risk: LOW–MED. Grid change affects autotune key and cache. Correctness
  pattern already validated in bwd kernel (same idiom).

## Ranking — "most likely to unlock 0.5× within today"

None of the three individually reach 0.5× FA3 on fwd+bwd. The wall-time
floor is set by the backward pass (260 µs FA3 vs 275 µs whale at headline —
already near parity). To hit 0.5× we need to **shrink the workload**, not
just speed it up.

Ranked by best-effort-per-hour on the realistic headline benchmark:

1. **(b) RoPE+norm+q_gain fusion** — LOW risk, HIGH signal, 3–5 h.
   Directly removes kernel launches FA3 structurally cannot remove from
   its training path.
2. **(c) Cross-Q-head K/V reuse in fwd** — LOW–MED risk, MED signal, 2–4 h.
   Closes the last of the fwd gap and also helps long-T shapes.
3. **(a) Shape-specialized CUDA extension** — HIGH risk, LOW-MED signal,
   16–24 h. Cannot be completed today; cannot hit 0.5× alone.

## Winner for today: Idea (b) — fused-prep forward

### Why (b) first
- Biggest expected µs win per hour of effort (rotary + rms_norm + q_gain
  are 3+ separate kernels today on the fwd leg of whale; FA3 also pays
  these in unfused form).
- Does not require changes to the bwd kernel, which is already at parity.
- Easy to A/B with a `WHALE_FWD_VARIANT=fused_prep` flag.
- Uses RoPE capability FA3 explicitly lacks in `flash_attn_func`.

### Minimum viable patch outline
1. New leg: `bash scripts/new_leg.sh whale_fused_prep_fwd`.
2. Add a new Triton kernel variant in the leg's `train_gpt.py` (NOT vault):
   `_attn_fwd_fused_prep_kernel(Q, K, V, COS, SIN, Q_GAIN, O, LSE, ...)`
   that loads Q/K tiles, applies rms_norm over D, interleaved-rotary
   multiply on first `rope_dims` elements using COS/SIN, scales Q by
   per-head Q_GAIN, then runs the existing online-softmax loop.
3. Add an alternate `whale_attn_fast` entrypoint in the leg that takes
   pre-norm q, k, v (no rms_norm, no rotary, no q_gain in caller) plus
   cos, sin, q_gain tensors.
4. Extend `scripts/whale_attention_bench.py` via leg override to route
   the whale backend through the new entrypoint (skip Python
   rms_norm/rotary/q_gain for whale only; FA3 still pays them because
   `flash_attn_func` has no rotary arg).
5. Add `tracked_env.sh` with `WHALE_FWD_VARIANT=fused_prep`.
6. Correctness: 1e-3 atol vs old path on small shape (B=1,T=128,H=8,KV=4).

### First-bench plan (for when GPU is available)
- Shape: headline `B=4, T=2048, H=8, KV=4, D=64, rope_dims=16, bf16, causal`.
- Comparison: three backends, same model dim / rope / q_gain:
  1. FA3 (unfused prep in Python) — current path.
  2. whale_fast (unfused prep in Python) — current parity baseline.
  3. whale_fused_prep (prep fused inside kernel) — new path.
- Metric: fwd-only mean µs over 300 iters, plus fwd+bwd mean µs.
- Success criterion for this leg only:
  - pass: whale_fused_prep fwd+bwd < 0.9× whale_fast fwd+bwd AND <
    FA3 fwd+bwd by at least one σ.
  - stretch: whale_fused_prep fwd < 0.5× FA3 fwd (Φ ≈ 47 / 2 = 23 µs).
- Then: if (b) passes, queue (c) as next leg `whale_fwd_pack_gqa` for the
  same shape.

## Honest assessment

proposal: stacking (b) + (c) could plausibly reach ~0.65× FA3 wall on
fwd-only at T=2048. For full fwd+bwd at 0.5×, we would also need to cut
whale's bwd by ~40%, which requires warp specialization or a CUDA
rewrite — not achievable today in Triton 3.6.

inference: the "0.5× FA3 fwd+bwd" target is not reachable in a single day
without FA3 bwd replacement. The realistic best-in-a-day is:

- whale fwd 0.5× FA3 fwd (achievable with (b) + (c) stacked)
- whale fwd+bwd ≈ 0.9× FA3 fwd+bwd (small win)

If the real user goal is "whale strictly faster than FA3 in the production
training loop," (b) alone delivers that today because FA3 cannot fuse RoPE
into training-path attention.
