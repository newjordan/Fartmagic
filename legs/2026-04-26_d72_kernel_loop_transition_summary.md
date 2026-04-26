# D72 Kernel Loop Transition Summary - 2026-04-26

## Training Resume Probes

All rows resume the same D72 loop-onset checkpoint:

`legs/2026-04-26_d72_loop_onset_checkpoint_builder/checkpoints/loop_onset.pt`

| test | mode | final step | train time | tok/s | result |
| --- | --- | ---: | ---: | ---: | --- |
| `2026-04-26_d72_loop_transition_fa3_clean24` | FA3 | 479 | 7.7m | 819661 | fastest control |
| `2026-04-26_d72_loop_transition_fa3_clean100` | FA3 | 555 | 10.0m | 727722 | same-horizon 100-step control |
| `2026-04-26_d72_loop_transition_x180_clean24` | hybrid x180 dynamic | 479 | 19.8m | 316782 | bad dynamic path |
| `2026-04-26_d72_loop_transition_hybrid_static_eager_cacheclean24` | hybrid static eager | 479 | 10.6m | 594829 | much better than x180, still slower than FA3 |
| `2026-04-26_d72_loop_transition_hybrid_static_compile_cacheclean24` | hybrid static compile | 479 | 10.4m | 604962 | compile recovered little speed |
| `2026-04-26_d72_split64_tail16dot_loop_transition_probe24` | hybrid tail16dot 128x64 w4 s4 | 479 | 8.1m | 770756 | first custom forward result close to FA3 |
| `2026-04-26_d72_split64_tail16dot_loop_transition_probe100` | hybrid tail16dot 128x64 w4 s4 | 555 | 9.1m | 797720 | stable 100-step confirmation |

Conclusion: the loop transition is not breaking model quality. The original custom hybrid path was correct but too slow. The D72 tail16dot forward path recovers the training-throughput gap, improving the loop probe from `604962 tok/s` to `770756 tok/s` at 24 steps and `797720 tok/s` over 100 steps. The same-horizon FA3 100-step control finished at `727722 tok/s`, so the promoted tail16dot path is faster in the matched 100-step checkpoint-resume test.

## Real-Shape Kernel Microbench

Shape: `B=48,T=2048,H=8,KV=4,D=72`, matching the observed train microbatch per rank.

| profile | avg whale/FA3 median ratio | verdict |
| --- | ---: | --- |
| `auto_current` | 1.336649 | best tested whale-forward path, still slower than FA3 |
| `fwd_128_128_w8_s4_mr224` | 1.347580 | close but worse than auto |
| `fwd_128_64_w8_s5_mr160` | 1.420775 | worse |
| `fwd_256_64_w8_s6_mr160` | 1.755875 | reject |
| `fwd_64_128_w4_s4_mr128` | 3.285443 | reject |
| `persistent_auto` | 1.571082 | reject |
| `persistent_128_128_w8_s4_mr224` | 1.579545 | reject |
| `persistent_128_64_w8_s5_mr160` | 1.568874 | reject |
| `persistent_256_64_w8_s6_mr160` | 2.192608 | reject |
| `d72split_128_64_w4_s4` | 2.256417 | reject; scalar tail math erased the padding win |
| `d72split_128_64_w8_s4` | 2.558837 | reject |
| `d72split_128_128_w4_s4` | 4.568074 | reject |
| `d72split_64_64_w4_s4` | 2.390342 | reject |
| `tail16dot_128_64_w4_s4` | 1.135430 | promote to 24-step loop-transition probe |
| `tail16dot_128_64_w4_s5` | 1.162147 | backup candidate |
| `tail16dot_128_64_w4_s4_repeat` | 1.161232 | repeat confirms winner class; not better than first run |
| `tail16dot_128_64_w4_s3` | 1.161928 | backup candidate |
| `tail16dot_128_64_w4_s2` | 1.171730 | backup candidate |
| `tail16dot_128_128_w4_s4` | 1.205415 | backup candidate |
| `tail16dot_64_64_w4_s4` | 1.236818 | better than default, not best |
| `tail16dot_128_64_w8_s4` | 1.274164 | near default |
| `tail16dot_128_32_w4_s4` | 1.252963 | reject; smaller N did not help |
| `tail16dot_64_32_w4_s4` | 1.268324 | reject; smaller M/N did not help |
| `tail16dot_64_128_w4_s4` | 1.295406 | near default |
| `tail16dot_256_64_w4_s4` | 1.547848 | reject |
| `tail16dot_128_64_w2_s3` | 2.161686 | reject; 2 warps too slow |
| `tail16dot_128_64_w2_s4` | 2.168292 | reject; 2 warps too slow |

Conclusion: do not keep tuning the old backward `qmr/dkdv` profile for `fa3_bwd`; that path delegates backward to FA3. The gap is in whale forward at D72. The current autotuned default forward is the best measured variant, but still about 1.34x slower than pure FA3 at the real train shape.

Split64+tail8 conclusion: correctness stayed in the expected BF16 tolerance band against the default forward path (`out_max_abs` about `0.007812`, `dk_max_abs` about `0.015625`), but the scalar tail correction made the best candidate about `2.25x` FA3. Do not spend loop-transition training time on the tail8 variant.

Tail16dot conclusion: replacing the scalar tail with a masked 16-wide tensor-core tail dot is the first positive D72 forward result. The best profile, `tail16dot_128_64_w4_s4`, measured `3.811811ms` versus same-GPU FA3 `3.357152ms`, ratio `1.135430`, and beat the same-run default whale reference of `4.236209ms`. It is now the right candidate for the checkpoint-resume loop-transition probe.

Refined config conclusion: the follow-up fanout did not beat the current winner. Keep `128,64,4,4` as the active config; `4` warps is required, and reducing `BLOCK_N` to `32` does not help at the real train shape.

## Artifact Paths

- `legs/2026-04-26_d72_loop_transition_resume_probe_summary.tsv`
- `legs/2026-04-26_d72_whale_fwd_config_realshape_microbench/microbench.tsv`
- `legs/2026-04-26_d72_whale_fwd_persistent_realshape_microbench/microbench.tsv`
- `legs/2026-04-26_d72_split64_tail8_kernel_microbench/microbench.tsv`
- `legs/2026-04-26_d72_split64_tail16dot_kernel_microbench/microbench.tsv`
- `legs/2026-04-26_d72_split64_tail16dot_loop_transition_probe24/results.tsv`
- `legs/2026-04-26_d72_tail16dot_refined_config_microbench/microbench.tsv`
- `legs/2026-04-26_d72_split64_tail16dot_loop_transition_probe100/results.tsv`
- `legs/2026-04-26_d72_loop_transition_fa3_clean100/results.tsv`
