# whale bwd ncu profile — RESULTS

## setup
- pod: vastai/pytorch:cuda-13.0.2-auto, H100 SXM 80GB, driver 580.95.05
- Python 3.12 / venv/main, PyTorch 2.11.0+cu130, Triton 3.6, FA3 3.0.0 cu130 abi3
- shape: `B=4, T=2048, H=8, KV=4, D=64, bf16, causal`
- branch: `whale/2026-04-16_pod_autoresearch` (commit `ca25e4f`)
- WHALE_BWD_VARIANT=fused_delta

## method
- `ncu --set detailed` blocked by `ERR_NVGPUCTRPERM` (driver-level perf-counter restriction on vast pods).
- Switched to `torch.profiler` (kineto) + `torch.cuda.Event` measurements for GPU-side truth.
- Scripts: `profile_kineto.py`, `profile_v2.py`, `run_bwd.py`.

## fact — GPU kernel time at headline

| kernel | whale | FA3 | delta |
|--------|------:|----:|------:|
| forward | `_attn_fwd_kernel` 72.3us | `FlashAttnFwdSm90` 46.8us | **+25.5us (whale slower)** |
| bwd main | `_attn_bwd_dkdv_inline_delta_kernel` 193.1us, `_attn_bwd_dq_inline_delta_kernel` 76.5us | `FlashAttnBwdSm90` 102.8us | **+166.8us (whale slower)** |
| bwd helpers | — | Preprocess 11.1us + ConvertdQ 10.7us + ConvertdQ 7.8us = 29.6us | — |
| **total GPU fwd+bwd** | **345.1us** | **187.9us** | **+157.2us (whale 1.84x slower)** |

## fact — wall time vs GPU time

| measurement | whale | FA3 |
|-------------|------:|----:|
| kineto GPU total | 345us | 188us |
| event fwd+bwd    | 347us | 466us |
| CPU overhead     | ~3us  | ~278us |

## inference
- Earlier "parity on fwd+bwd wall" was correct but hid a 45% GPU gap.
- Whale's wall-time edge comes from efficient Python wrapper (`torch.library.custom_op` vs FA3's heavy Python wrapper), not kernel speed.
- FA3's 103us monolithic bwd kernel is 2.6x faster on-GPU than our 193+77=270us split-kernel bwd.
- FA3 fwd uses cutlass Sm90 with TMA + warp specialization — Triton 3.6 options we've tested (tl.async_task, tl.make_tensor_descriptor TMA) haven't closed the gap.

## proposal
Two honest paths:

### A. Wall-time victory (done)
- We already win wall time at the headline (same total, whale fwd wins, whale bwd loses).
- We win every shape on fwd wall time and on smaller shapes overall.
- This is what training loops actually experience.
- **Merge, document, move on.**

### B. Close the GPU gap (days of work, uncertain)
- Forward: -25us requires warp spec + TMA on loads, matching cutlass Sm90 fwd.
- Backward: -138us requires fusing dkdv and dq into one kernel (like FA3) or using cluster-level features.
- Triton 3.6 exposes `tl.async_task` but not full TMA+cluster combination FA3 uses.
- Risk: high; expected gain on top of "wall-time parity": visibility only, no training speedup.

## next actions (pending user choice)
- If A: tear down pod, commit this leg, tag a release.
- If B: open new leg for forward warp-spec + TMA rewrite.
