# Lever C — ready.md

Status as of 2026-04-16, after Lever A edits landed in vault
(`vault/whale_kernel_triton.py` now 1533 lines; fact: `wc -l` output).

## (a) Phases that launch with NO vault patch

- **Phase A — patch-free.** Uses only `WHALE_FWD_VARIANT={default,tma}`.
  `WHALE_FWD_VARIANT` is read at `vault/whale_kernel_triton.py:1179` in
  unpatched vault. The TMA forward kernel (`_attn_fwd_tma_kernel`) and its
  autotune config list (`_fwd_tma_configs` at lines 81-94) are already present.
- **Phase B — patch-free.** Uses `WHALE_FWD_TMA_CONFIG=BM,BN,W,S` on top of
  `WHALE_FWD_VARIANT=tma`. That env name is decoded by `_env_force("FWD_TMA")`
  at line 85 via `_env_force` (lines 50-64). All already present in the vault.
- **Phase C — REQUIRES the vault patch.** `WHALE_FWD_TMA_WARPSPEC=1` is inert
  in the unpatched vault (no reader for it; the kernel signature has no
  `WARPSPEC` constexpr). Running Phase C without the patch will silently
  duplicate Phase B results — the env has no effect until the patch lands.

`run.sh` has been updated with a header comment reflecting this
(fact: line 2 of `legs/2026-04-16_whale_fwd_long_t/run.sh`).

## (b) Exact line numbers for the Lever C patch

Re-verified against current vault:

| Target | Line(s) | Status |
| --- | --- | --- |
| TMA fwd kernel def | 304 (decorator at 302-303) | shifted +2 from original patch |
| Kernel signature block | 304-319 | replace last line `IS_CAUSAL: tl.constexpr,` + add `WARPSPEC`, `NUM_STAGES_WS` |
| K/V loop to patch | 359 | unchanged line number |
| K/V loop body | 360-376 | unchanged; only loop header modified |
| Dispatch site | 1179-1192 | shifted +8 from original patch |
| `_fwd_tma_configs` | 81-94 | unchanged |
| `_env_force` | 50-64 | unchanged |
| Lever A WARPSPEC precedent | 446-447, 479-484 | reference pattern |

Naming alignment: Lever A chose `WARPSPEC: tl.constexpr = False,
NUM_STAGES_WS: tl.constexpr = 3` (line 446-447) and
`num_stages=NUM_STAGES_WS` inside `tl.range(...)` (line 481). `vault_patch.md`
has been updated to use the same names so both levers are consistent.

Env names stay as documented: `WHALE_FWD_TMA_WARPSPEC` (on/off),
`WHALE_FWD_TMA_NUM_STAGES` (int, default 3). Dispatch-site local var is
`num_stages_ws` to avoid shadowing any future `num_stages` semantics.

## (c) Phase-A prior-data check

`legs/2026-04-16_whale_fwd_warpspec_tma/RESULTS.md` exists but only tested
`SHAPE=4,2048,8,4,64` (fact: `sweep_fwd_tma.sh:6`, `sweep_fwd_configs.py:10-15`
hard-code T=2048). It reports:
- Non-TMA winner (64,64,4,3) at 74.2us GPU-time.
- TMA variant `WHALE_FWD_VARIANT=tma` at same winner config: 72.2us.
- TMA beats default by 2us at T=2048.
- NO T >= 4096 data exists anywhere in the corpus.

That evidence is not conclusive for long-T: TMA's benefit scales with the
number of K/V-loop iterations (T/BLOCK_N). Phase A at T in {4096, 8192} is
genuinely novel data, not a redundant sweep. Lower-risk launch order is
Phase A first, then Phase B, then the patch + Phase C.

## (d) Triton 3.6 TMA + warp_specialize constraint check

- `tl.range(warp_specialize=True)` plumbs to IR attribute `tt.warp_specialize`
  (`triton/compiler/code_generator.py:1224-1225`).
- On H100 (cap 9.0), that attribute is consumed by
  `nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)`
  at `triton/backends/nvidia/compiler.py:274`. Active branch is
  `capability // 10 in [8, 9]` (line 268).
- The Hopper warpspec pass receives `opt.num_stages` at build time — so the
  pipeline depth comes from the autotune Config, NOT only the `num_stages=`
  kwarg on `tl.range`. For safety, both should match (already done in patch:
  `NUM_STAGES_WS` defaults to 3 = typical Config.num_stages for this kernel).
- No TMA-specific disallow was found. The `tt.warp_specialize` attribute is
  agnostic to load source (tensor-descriptor vs pointer).
- Lever A (non-TMA kernel) uses `tl.range(..., num_stages=NUM_STAGES_WS,
  warp_specialize=True)` verbatim (line 480-482). Lever C reuses the same
  two-kwarg signature on a TMA-using kernel — same IR, same pass.
- **Stale doc caveat:** `tl.range` docstring in
  `triton/language/core.py:3281-3283` says warp_specialize is "only supported
  on Blackwell GPUs." This contradicts the Hopper pass (above). Treat the
  docstring as under-documented, not as a disallow. This is inference, not
  fact — ultimate ground truth is the Phase C run on the pod.

## (e) Remaining unknowns

1. **Autotune compile-time errors with s=4 + warpspec.** The fallback config
   list in `vault_patch.md` (narrow to s in {2,3}, warps in {8,}) is pre-staged
   as a doc-only mitigation. If autotune throws on s=4, cutting to the narrow
   list is a ~3-line vault edit away.
2. **TMA + WARPSPEC correctness at T=8192.** Not benched anywhere; Phase C
   MUST include the `max(abs(out - ref)) <= 1e-3` bf16 numerics gate named
   at the tail of `vault_patch.md`. The bench_stable.py path should already
   compare against SDPA/FA3 — confirm the correctness hook before Phase C
   launch.
3. **Overhead of `tl.range` on NON-warpspec path.** The patch changes the
   loop header unconditionally to `tl.range(..., warp_specialize=WARPSPEC)`.
   With WARPSPEC=False, Triton should compile identically to the current
   `range(...)`. Inference, not fact — Phase A after-patch should re-baseline
   the default TMA path to detect any regression from the header swap.
4. **FA3-like tile (128,128,8,2) at T=8192.** This is the Phase B hypothesis
   H9a. No prior evidence either way.

## Post-patch apply sequence (when GPU frees)

1. Apply patch by hand from `vault_patch.md` to `vault/whale_kernel_triton.py`
   (three edits: signature, loop header, dispatch).
2. Diff-sanity: `git diff vault/whale_kernel_triton.py` should show ~6 lines
   changed, no others.
3. Re-run `bash legs/2026-04-16_whale_fwd_long_t/run.sh`. Phase A and Phase B
   act as regression baselines after the patch; Phase C is the new hypothesis.
4. If Phase C WARPSPEC=1 underperforms Phase B's best, revert the vault and
   log the long-T ceiling as inference item in RESULTS.md.
