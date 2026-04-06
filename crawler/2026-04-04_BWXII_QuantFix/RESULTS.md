# BW XII Quant Fix — Results

## Status

This note records the BW XII Quant Fix results known as of April 6, 2026.
Some outputs were recovered directly from files under `/workspace/parameter-golf`
on a surviving single-GPU pod. The 2x GPU "movie test" metrics were preserved
from terminal transcript even though the dual-GPU pod was not reachable at the
time this note was written.

## 1-GPU Quant Sweep

Recovered from:

- `/workspace/parameter-golf/crawler/2026-04-04_BWXII_QuantFix/results_1gpu/quantfix_summary_s444_20260405_180122.tsv`

Best quantized result observed:

- `T4_smart_wd012_Q2_gptq_loop_int8`
- `raw_bpb=1.2634`
- `int6_sw_bpb=1.27427121`
- `quant_gap=0.0109`
- `bytes=12246411`

Relevant comparisons:

- `T4_smart_wd012_Q1_gptq_loop`: `int6_sw_bpb=1.27669537`
- `T0_wd012_Q2_gptq_loop_int8`: `int6_sw_bpb=1.27719240`
- `T2_wd020_Q2_gptq_loop_int8`: `int6_sw_bpb=1.28900083`, `quant_gap=0.0119`

Conclusions from this sweep:

- `smart_skip` was the strongest quant-fix intervention tested.
- `loop-aware GPTQ + crawler int8` was the best quantization path.
- Higher WD alone did not beat `smart_skip`.

## 2x GPU Movie Test

Recorded from the successful dual-GPU run on April 6, 2026.

Intended summary path on the dual-GPU pod:

- `/workspace/parameter-golf-lab/crawler/2026-04-04_BWXII_QuantFix/results_movie/movie_summary_s444_20260406_032806.tsv`

Known results:

| arm | raw_bpb | int6_sw_bpb | quant_gap | bytes |
| --- | ---: | ---: | ---: | ---: |
| `R1_fire_embed` | `1.2587` | `1.28120855` | `0.0225` | `10972986` |
| `R4_full_fix` | `1.2420` | `1.28033274` | `0.0383` | `10843870` |

Interpretation:

- `R4_full_fix` was the best raw result observed in this leg.
- Quantization erased most of that gain.
- `R4_full_fix` still edged `R1_fire_embed` after quantization, but only by
  `0.00087581` BPB.

## Overall Read

The leg split into two clear wins:

- Best raw model behavior: `R4_full_fix`
- Best quantized behavior: `T4_smart_wd012_Q2_gptq_loop_int8`

This means the new fire/inject/merge-cap technique improved raw BPB, but it did
not solve the quant gap. The strongest next-step hypothesis is to combine the
raw-model win with the quant-fix win:

- `R4_full_fix + smart_skip`
- likely also test `crawler_int8=1`

## Missing Artifacts To Recover Later

Minimum salvage set from the dual-GPU pod:

- `movie_summary_s444_20260406_032806.tsv`
- `R1_fire_embed_train_s444_20260406_032806.log`
- `R1_fire_embed_q1_s444_20260406_032806.log`
- `R4_full_fix_train_s444_20260406_032806.log`
- `R4_full_fix_q1_s444_20260406_032806.log`

Nice to have:

- `R1_fire_embed_s444_20260406_032806.final_model.pt`
- `R1_fire_embed_q1_s444_20260406_032806.final_model.int6.ptz`
- `R4_full_fix_s444_20260406_032806.final_model.pt`
- `R4_full_fix_q1_s444_20260406_032806.final_model.int6.ptz`
