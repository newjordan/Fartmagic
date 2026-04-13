# PR #1579 Concept Notes (Imported Into TON-E)

These notes capture what we took from `openai/parameter-golf#1579` and what we intentionally did not copy.

## What They Did

- Switched to crawler rhythm: `3 flat + 2 crawler` with `2` crawler loops (`3F+2Cx2`).
- Used a wider/token-richer build (`SP8192`, `d=736`, larger embedding config).
- Leaned heavily on post-quant endgame adaptation (TTT) to recover quantization loss.

## Why They Likely Did It

- Shorter unique flat path plus shared crawler reuse keeps effective depth while lowering unique parameter pressure.
- Two crawler blocks gives more middle transform capacity than `+1C` without going full `+3C/+4C`.
- TTT on shared crawler weights gives a leverage effect: one update influences repeated loop usage.

## What We Copied Into TON-E

- The layer rhythm idea itself (`3F+2Cx2`) as an overlay on our runner.
- Optional XSA coverage across flat + crawler blocks (`XSA_INCLUDE_FLAT=1` default under TON-E profile).
- Profile-style wiring so we can A/B quickly against our baseline.

## What We Did Not Copy

- Their endgame policy and crawler handling path.
- Their exact tokenizer/width stack as defaults.
- Any dependence on their script internals.

## Why Our Runner Stays In Control

- Our 10-minute harness behavior is already known and stable in-repo.
- Our quant/export path and wallclock stop behavior are battle-tested with crawler variants.
- We can isolate architecture rhythm effects without mixing in an unfamiliar endgame implementation.
