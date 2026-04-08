# Comprehensive Research Synthesis
_Date: 2026-04-02_

## Scope

This document extracts viable research from local `junkyard/` work only and promotes what is still worth carrying forward.

Excluded on purpose:
- `Medusa*`
- contaminated storm-era SLOT branches
- quarantine as an implementation source

## Executive Summary

- The clean legal neural anchor is still the Rascal II lane at `1.10986874` and about `15.44MB`.
- The strongest locally validated non-SLOT neural research remains the Rascal/Rat Rod family, not crawler/Frug recurrence.
- The strongest local zero-cost or near-zero-cost levers are loader tuning, warmdown tuning, and keeping compile on.
- The strongest local n-gram lane is already mature. It is useful as a scoring asset, but it is not the main missing neural training signal.
- Several seductive branches are already dead: Synapse, Siphon, complementary training, value residual, trigram-on-top, and shared-weight crawler recurrence as a primary SOTA path.
- QK gain, GPTQ-in-budget, and modest bigram expansion remain the best clean forward-looking neural ablation targets, but they should be treated as candidates until reconfirmed from the clean Rascal lane.

## Canonical Anchors

### Safe legal submission anchor

- Source: `junkyard/experiments/SOTA/README.md`
- Current best legal submission in local junkyard:
  - `2026-03-30_JUNKYARD_RAT_RASCAL_II_nogptq`
  - `final_sliding_window_exact val_bpb=1.10986874`
  - `Serialized model int6+zstd: 15,435,532 bytes`

### Clean launcher copy

- Source: `junkyard/experiments/Rascal_Final_Submission_LC4/README.md`
- `Rascal_Final_Submission_LC4` is documented as a byte-identical copy of the legal Rascal II trainer, with only `COPRIME_MAX_LOADED_SHARDS=4` promoted in the launcher.

### Best local neural+n-gram lane

- Source: `junkyard/experiments/Rat_Rod/PROGRESS.md`
- `Rat Rod Green v1`:
  - sliding base: `1.1129`
  - post-EMA: `1.1364`
  - legal n-gram eval: `0.4489`
  - steps: `6882`
  - ms/step: `87.20`

## Promoted Viable Research

### 1. Loader cache4 is a real local win

Strongest clean evidence:
- `Rascal_Stripper_Skipgram_2200/notes/2026-03-31_next_single_gpu_pack_seed444.md`
- baseline: `1.3112`, `786.00ms`
- `loader_cache4`: `1.3101`, `782.59ms`
- delta: `-0.0011 BPB`, `-3.41ms`

Secondary support:
- `2026-03-31_single_h100_ablation_matrix.md`
- `loader_cache4` showed slight speed and tiny quality gain there too

Promotion:
- `COPRIME_MAX_LOADED_SHARDS=4` is worth treating as part of the clean Rascal base unless disproven in a fresh multi-seed run.

### 2. Compile-on is mandatory

Source:
- `Rascal_Stripper_Skipgram_2200/notes/2026-03-31_single_h100_ablation_matrix.md`

Finding:
- `compile_off` was a catastrophic slowdown with no quality benefit.

Promotion:
- Do not burn time on non-compiled neural ablations unless the goal is pure debugging.

### 3. Rat Rod / Rascal shape choices remain the best local neural stack

Source:
- `junkyard/experiments/Rat_Rod/PROGRESS.md`

Promoted stack elements:
- `XSA_LAST_N=11`
- `BIGRAM_VOCAB_SIZE=2048`
- `ROPE_DIMS=16`
- `SWA_EVERY=50`
- `COMPLEMENT_ALPHA=0`
- `MTP_NUM_HEADS=0`
- `SKIP_GPTQ=1` for the current legal Rascal submission lane

Interpretation:
- This family is the strongest stable local base.
- New experiments should mutate this line, not replace it with crawler or biology novelty.

### 4. Warmdown shape is one of the best low-cost local hunt areas

Source:
- `junkyard/experiments/Rat_Rod/PROGRESS.md`
- `junkyard/experiments/Rat_Rod/WARMDOWN_HYPOTHESES.md`

Observed local signal:
- At 200s, `WARMDOWN_ITERS=2000` beat `3500` and `5000`
- Reported cap BPB:
  - `2000`: `1.3504`
  - `3500`: `1.3775`
  - `5000`: `1.4111`

Promotion:
- Warmdown is one of the cleanest things to keep hunting.
- The local result supports shorter warmdown as a serious candidate in this family.
- The follow-up warmdown-shape variants `swirl`, `cascade`, and `jitter` are still viable unrun research.

### 5. SWA cadence is worth minor follow-up, but it is a small edge not a breakthrough

Source:
- `junkyard/experiments/Rat_Rod/PROGRESS.md`

Signal:
- `SWA_EVERY=100` slightly beat `50` at 200s.

Promotion:
- Keep as a secondary sweep axis.
- Do not prioritize over warmdown, loader, QK gain, or GPTQ questions.

### 6. GPTQ-in-budget is still one of the highest-upside open neural questions

Sources:
- `junkyard/experiments/Rascal_AB_1p109_to_1p102/README.md`
- `junkyard/experiments/QK_SLOT_Ablation/STATUS.md`
- `junkyard/experiments/RESEARCH_REPORT_2026-03-27_racing_garage.md`

What is local and true:
- The legal Rascal submission lane currently uses `SKIP_GPTQ=1`.
- Multiple local experiment packs were explicitly organized to test GPTQ, often paired with QK gain and/or follow-on eval systems.

Promotion:
- This remains one of the main unresolved upgrade paths above the safepoint.
- It should be tested from the clean Rascal lane, not from any contaminated SLOT branch.

### 7. QK gain is a promoted candidate, but not yet a locally closed win

Source:
- `junkyard/experiments/QK_SLOT_Ablation/STATUS.md`

What the junkyard says:
- `QK_GAIN_INIT=4.0` was treated as a serious imported signal and planned as an independent cross-correlation ablation with SLOT.

Promotion:
- `QK_GAIN_INIT=4.0` belongs in the clean ablation matrix.
- It should be treated as a high-confidence candidate, not as a locally proven win yet.

### 8. Bigram expansion is still viable and underexplored in the Rascal lane

Sources:
- `Rat_Rod/PROGRESS.md`
- `Rascal_AB_1p109_to_1p102/README.md`

What is true:
- The strong local base uses `BIGRAM_VOCAB_SIZE=2048`.
- Larger bigram tables are repeatedly present in planned AB work.

Promotion:
- `2816` and `3072` are viable next ablation points.
- This is a cleaner lever than trigram, which already washed out locally.

## Viable but Not Yet Promoted to Base

### TurboMuon

Sources:
- `junkyard/experiments/Rascal_Turbo/README.md`
- `Rascal_Stripper/smoke_run1_results.md`

Current status:
- Dedicated local line exists.
- The early smoke result is not valid evidence because all four scripts were still identical at that point.
- Later stripped notes suggest `muon_ns4` was mixed or negative in the stripped setup.

Decision:
- keep as an ablation
- do not promote into the base by default

### EngramLite

Source:
- `junkyard/experiments/Rascal_AB_1p109_to_1p102/README.md`

Current status:
- clearly queued in the clean A/B lab
- not supported by a clean local winning result in the docs reviewed here

Decision:
- viable but unproven

### Skip-gram calibration

Sources:
- `Rascal_Stripper_Skipgram_2200/notes/2026-03-31_skipgram_calibration_seed444.md`

Finding:
- only `-0.0001 BPB` at best with meaningful slowdown

Decision:
- do not promote
- keep only as low-priority curiosity

## Rejected or Closed Research

### 1. Trigram-on-top is a wash

Source:
- `Rat_Rod/PROGRESS.md`

Finding:
- `TRIGRAM=1` was explicitly called a wash in `Rat Rod Green v2`

Decision:
- not worth near-term budget

### 2. Complementary training is worse

Source:
- `Rat_Rod/PROGRESS.md`

Finding:
- `COMPLEMENT_ALPHA=0.5` made both sliding and n-gram worse

Decision:
- reject for this line

### 3. Value residual is worse

Source:
- `Rat_Rod/PROGRESS.md`

Finding:
- `VALUE_RESIDUAL=1` regressed sliding

Decision:
- reject unless a very different surrounding stack appears

### 4. Synapse / HS-MTP bridge is dead

Source:
- `Rat_Rod/PROGRESS.md`

Finding:
- CPU bridge was too slow
- GPU-native version was worse on both base and n-gram

Decision:
- closed

### 5. Siphon is dead

Source:
- `Rat_Rod/PROGRESS.md`

Finding:
- severe regression on sliding and n-gram

Decision:
- closed

### 6. Shared-weight crawler recurrence is closed as a primary SOTA path

Sources:
- `RESEARCH_REPORT_crawler_signal_analysis.md`
- `FINDINGS_H_FRUG.md`

Findings:
- crawler advantage was mostly width, not recursion
- recursion showed zero measurable per-step signal
- more looping decayed over training
- shared recurrence caused gradient conflict
- Frugendorff line was explicitly marked closed

Decision:
- do not spend primary SOTA budget on crawler recurrence

### 7. Crawler add-ons that already failed

Source:
- `FINDINGS_H_FRUG.md`

Rejected:
- frozen-teacher KL distillation
- loop bottleneck gate
- naive width reinvestment at that tested scale

## Long-Horizon Research Still Worth Keeping

### Nitrust backlog

Source:
- `junkyard/Nitrust/HYPOTHESES.md`

Worth keeping as future architecture R&D:
- low-rank loop adapters
- split sharing
- adaptive loop buckets
- latent funnel recurrence
- memory tokens
- dual-rate superblocks

Interpretation:
- this is future architecture work, not immediate Rascal submission work
- it is valuable because it is explicitly n-gram-free and hardware-first

## Practical Research Map By Family

### Rascal family

Best current use:
- primary legal submission line
- primary neural ablation line

Keep:
- clean vault/safepoint source
- loader cache4
- compile on
- QK gain candidate
- GPTQ-in-budget investigation
- bigram expansion
- warmdown sweep

Avoid:
- contaminated SLOT branches
- skip-gram as a priority
- Turbomuon as default until reconfirmed

### Rat Rod family

Best current use:
- strongest local n-gram research asset
- best warmdown and training-side dead-end knowledge

Keep:
- warmdown research
- n-gram settings as a scoring reference

Avoid:
- promoting Rat Rod-specific dead lines back into Rascal

### Crawler / Frug family

Best current use:
- archive of negative evidence

Keep:
- width-vs-depth lessons

Avoid:
- using shared recurrence as the main route to neural SOTA

## Immediate Promotion List

These are the things that should actively shape current clean ablations:

1. `loader_cache4`
2. `QK_GAIN_INIT=4.0`
3. `SKIP_GPTQ=0` vs current nogptq control
4. `BIGRAM_VOCAB_SIZE=2816` and `3072`
5. warmdown experiments, especially shorter warmdown and non-linear warmdown shape
6. keep compile enabled and launcher/env clean

## Immediate Do-Not-Waste-Time List

1. trigram
2. complementary training
3. value residual
4. Synapse / HS-MTP bridge
5. Siphon
6. crawler recurrence as primary SOTA lane
7. skip-gram promotion

## Source Files Used

- `junkyard/experiments/SOTA/README.md`
- `junkyard/experiments/Rascal_Final_Submission_LC4/README.md`
- `junkyard/experiments/Rascal_AB_1p109_to_1p102/README.md`
- `junkyard/experiments/Rascal_Turbo/README.md`
- `junkyard/experiments/Rascal_Stripper/smoke_run1_results.md`
- `junkyard/experiments/Rascal_Stripper_Skipgram_2200/notes/2026-03-31_single_h100_ablation_matrix.md`
- `junkyard/experiments/Rascal_Stripper_Skipgram_2200/notes/2026-03-31_skipgram_calibration_seed444.md`
- `junkyard/experiments/Rascal_Stripper_Skipgram_2200/notes/2026-03-31_next_single_gpu_pack_seed444.md`
- `junkyard/experiments/Rat_Rod/PROGRESS.md`
- `junkyard/experiments/Rat_Rod/WARMDOWN_HYPOTHESES.md`
- `junkyard/experiments/QK_SLOT_Ablation/STATUS.md`
- `junkyard/experiments/MASTER_CHECKLIST.md`
- `junkyard/experiments/RESEARCH_REPORT_2026-03-27_racing_garage.md`
- `junkyard/experiments/RESEARCH_REPORT_crawler_signal_analysis.md`
- `junkyard/experiments/FINDINGS_H_FRUG.md`
- `junkyard/Nitrust/HYPOTHESES.md`
