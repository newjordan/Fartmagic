# whale forward: close 25us gap to FA3 — RESULTS

## setup
- pod: vastai/pytorch:cuda-13.0.2-auto, H100 SXM 80GB, driver 580.95.05
- PyTorch 2.11.0+cu130, Triton 3.6, FA3 3.0.0 cu130
- shape: `B=4, T=2048, H=8, KV=4, D=64, bf16, causal`
- baseline GPU times: whale 72us, FA3 47us

## changes tried

### 1) Non-TMA tile sweep (3×4×2×4 = 96 configs on pod)
- script: `sweep_on_pod.sh`
- winner: `BM=64, BN=64, w=4, s=3` at **74.8us**
- `BM=128, BN=64, w=4, s=3` = 80.3us
- `BM=128, BN=128, w=8, s=3` = 93.4us
- larger tiles all worse (or OOM at 128x256, 256x256)
- **autotune already picks optimal (or near-optimal) config**

### 2) TMA forward kernel (added `_attn_fwd_tma_kernel`)
- added `_fwd_tma_configs()` config list
- added `WHALE_FWD_VARIANT=tma` dispatch in `_whale_attn_fwd_impl`
- kernel uses `tl.make_tensor_descriptor` for Q/K/V/O
- correctness: bit-exact match with non-TMA
- GPU time at autotune winner `(64,64,4,3)`: **72.2us** (default: 74.2us, -2us)
- TMA sweep (48 configs): same winner, no config under 75us

### fact
- Triton 3.6 fwd ceiling at headline shape: **~72us** (TMA) / **~74us** (non-TMA).
- FA3 Sm90 fwd: 47us via cutlass warp-specialized async pipeline + TMA.
- The remaining 25us gap is **architectural** — requires warp specialization which
  is experimental (`tl.async_task`) in Triton 3.6.

## inference
Forward closure not possible in Triton 3.6 without a warp-specialized rewrite.
Evidence: tile tuning (96 configs) + TMA (48 configs) + autotuner all converge
on the same 72-75us ceiling.

## kept
- `_attn_fwd_tma_kernel` added (opt-in `WHALE_FWD_VARIANT=tma`); 2us GPU win,
  no wall-time win (within CPU-overhead variance).

## next
Move to the backward (138us gap, 5x more room than fwd). Candidate: fuse
`_attn_bwd_dkdv_inline_delta_kernel` and `_attn_bwd_dq_inline_delta_kernel`
into a single monolithic bwd kernel like FA3's `FlashAttnBwdSm90`. Requires
atomics for dq or a separate postprocess kernel.
