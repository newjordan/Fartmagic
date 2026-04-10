# Midnight_Black — GPTQ Bank Fix + Outlier-Preserve + Warmdown 3000

## Parent
midnight_GPTQ (GPTQ bank fix) — itself based on Midnight 12L vault source

## Variables (3, stacked)
1. GPTQ bank fix (from midnight_GPTQ) — bug fix, not a variable
2. Outlier-preserve export (QUANT_OUTLIER_PRESERVE=1) — skip percentile search, always use max
3. Warmdown 3000 (was 3500) — shorter warmdown schedule

## Why
Stack of the three best quant-gap screen results:
- Arm C (GPTQ bank): -27% quant gap — structural bug fix
- Arm A (outlier-preserve): -22% quant gap — free, no overhead
- Arm E (warmdown_3000): best absolute BPB in screens

This is aggressive because it changes 3 things. The safe version (midnight_GPTQ) isolates
the highest-confidence change alone.

## Gate target
Beat 1.10568 BPB (seed 444). Expect meaningful quant gap reduction.

## Risk
Medium. Warmdown 3000 is the untested variable here.
GPTQ fix is structural. Outlier-preserve is export-only.
If this fails but midnight_GPTQ passes, warmdown or outlier-preserve is the culprit.
