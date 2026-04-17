Parent: vault/train_gpt_midnight_iii_base.py
Leg: legs/2026-04-16_whale_hybrid_train_t2048

# Hypothesis

Swapping the attention forward at T=2048 from `flash_attn_func` (FA3) to
`whale_fwd_fa3_bwd` (whale Triton fwd + FA3 CUDA bwd) gives a measurable
end-to-end training step speedup, with bitwise-equivalent backward gradients
(same FA3 backward kernel is invoked).

## fact: bench evidence (whale fwd + FA3 bwd vs pure FA3)

Source bench harness: `legs/2026-04-16_whale_long_t_profile/bench_stable.py`
(defines `whale_hybrid` as `whale_fwd_fa3_bwd`, line 54-55).

Source results JSON (rounds=8 iters=100 sweep, single H100 SXM, bf16 GQA causal,
H=8 KV=4 D=64): `legs/2026-04-16_whale_pod_autoresearch/evidence/hybrid_sweep.json`

Excerpt for shape (B=4, T=2048, H=8, KV=4, D=64), forward-only mean:
- whale_hybrid fwd_mean_ms = 0.08107
- fa3         fwd_mean_ms = 0.09170
- delta = -11.6% on the forward kernel

User-cited high-precision rounds=40 iters=800 sweep on shape (B=2, H=8, KV=4, D=64):
- (B=2, T=2048): hybrid 0.325 ms vs fa3 0.384 ms = -15%
- (B=2, T=4096): hybrid 0.373 ms vs fa3 0.420 ms = -11%

(High-precision JSONs land at `legs/2026-04-17_whale_hybrid_headline/evidence/`
when the queued `legs/2026-04-16_whale_bench_queue/queue_hybrid_check.sh` run
completes; that path is not yet present locally — those numbers are
caller-reported, not yet checked into a tracked file in this repo.)

## inference

Production training is at T=2048 with bf16 GQA causal, exactly the win zone.
The backward is unchanged (still FA3's CUDA backward), so gradient quality is
identical to the FA3 baseline. The forward is the only thing that swaps.
Expected end-to-end step speedup is bounded by the fraction of step time spent
in attention forward; per-attention-call savings are ~50-60us at T=2048.

## proposal

Run this leg's `train_gpt.py` end-to-end against the same `tracked_env.sh` as
the leading midnight III variant. Compare wall-clock per step and final loss
vs the most recent FA3-baseline midnight III training run. Keep all other
hyperparameters frozen.

## risks / caveats

- fact: the swap is gated on `whale_fwd_fa3_bwd is not None`. If the import
  fails on the pod (e.g. Triton 3.6 mismatch, kernel build error), the code
  falls through to `flash_attn_3_func` -- training still runs but at FA3
  baseline speed. Check pod log for "from vault.whale_kernel_triton import"
  succeeding before claiming the hybrid path is active.
- fact: `whale_fwd_fa3_bwd` requires `flash_attn_interface._flash_attn_backward`
  importable at backward time. The pod's FA3 wheel must support that symbol.
- inference: numerics may differ slightly from pure FA3 because the Triton
  forward uses different reduction order. Expected delta is well within bf16
  noise (< 1e-3 relative). Final loss should match FA3 to within seed noise.
- inference: at T >= 8192 the whale forward loses to FA3 (see
  `evidence/triton_leg_state_20260416.md`). This leg is intentionally targeted
  at T=2048 production; do NOT extend without re-benching.
- fact: per CLAUDE.md the diff guard requires running
  `python3 scripts/leg_diff_guard.py legs/2026-04-16_whale_hybrid_train_t2048`
  before the launch step.
