# Base Model Decision - 2026-04-25

Purpose: choose the neural body worth quant/compression repair.

## Decision

Use `h10_l11_mlp50` as the provisional base neural model.

This is a provisional decision because it is raw-only evidence. It is good
enough to choose the next engineering target, not good enough to claim a final
submission candidate.

## Evidence

| Candidate | Body | Params | Step | Raw BPB | Throughput | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `h10_l11_mlp50` | 11L, MLP 5.0, xsa last 10 | `52,197,208` | `2585` | `1.0839` | `1,135,720 tok/s` | pick |
| `h08_mlp50` | 12L, MLP 5.0, xsa last 11 | `56,513,760` | `2413` | `1.0840` | `1,054,894 tok/s` | reject: tied raw, larger/slower |
| `h06_xsa12_mlp45` | 12L, MLP 4.5, xsa last 12 | `52,532,448` | `2493` | `1.0868` | `1,095,638 tok/s` | reject: weaker raw without size win |
| `h03_one_loop` | 12L, MLP 4.0, one loop | `48,548,832` | `2853` | `1.0892` | `1,247,969 tok/s` | reject: smaller but raw edge too reduced |
| PR1493 anchor | 11L, MLP 4.0 | `35,944,536` | `3636` | `1.0968` | `1,590,929 tok/s` | control |

## Why h10

- Best raw score in the trusted queue: `1.0839`.
- Essentially tied with h08, but h10 is about `4.3M` params smaller and faster.
- Still about `0.0129` BPB better than recorded PR1493 raw in the same 2x,
  1800s setup.
- The failure is not that h10 lacks neural potential; the failure is that the
  final quant/compression/eval stack is not stabilized.

## Next Gate

Do not spend on new architecture search or TTT. Next work is:

1. Tie a checkpoint/artifact to the selected h10 config, or confirm none exists.
2. Build the quant harness against the exact selected checkpoint.
3. Establish same-checkpoint global int5 baseline.
4. Test MLP-projection int6 subsets or equalization only after the baseline
   table exists.

