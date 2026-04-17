# Results — 2026-04-16_whale_fast_kernels

Device on which these numbers were measured: **NVIDIA GB10** (local), CUDA 13.0,
PyTorch 2.11.0+cu130. The target pod is 8x H100 SXM; numbers there will differ
but the qualitative findings (correctness, fwd+bwd faster than SDPA, no Dynamo
graph breaks) are device-independent.

## 1. Baseline `vault/whale_kernel_triton.py` — before this leg

Artifact: `evidence/baseline.json`
Kernel snapshot: `kernel/whale_kernel_triton.before.py`

- fact: Forward kernel is numerically **broken**. `fwd_max_abs_err` is 3.3–4.3
  across all shapes in `evidence/baseline.json`. This is the diagonal-block
  clobbering of `m_i`/`l_i`/`acc`: the kernel discards the off-diagonal running
  state for any `q_block_id > 0`.
  Evidence: `evidence/baseline.json` lines 14, 25, 36, 47, 58, 69, 80.
- fact: Backward *numerics* look OK only because the previous implementation
  routed the backward through `F.scaled_dot_product_attention` via `torch.func.vjp`.
  Evidence: `kernel/whale_kernel_triton.before.py` lines 166–183.
- fact: `fwd+bwd` of the broken kernel is **slower** than plain SDPA: 3.36 ms
  vs 3.10 ms at `B=8, T=2048, H=8, KV=4, D=64, bf16`.
  Evidence: `evidence/baseline.json` lines 95–112.

## 2. After rewrite — Triton forward + Triton backward + autotune

Artifact: `evidence/iter2_autotune.json`
Artifact (harness): `evidence/whale_bench_custom_core.json`,
`evidence/whale_bench_sdpa_core.json`

### Correctness (bf16)

All errors below absolute scales consistent with bf16 roundoff
(mantissa 7 bits → eps ≈ 7.8e-3).

| shape (B T H KV D) | fwd_err | dq_err | dk_err | dv_err |
|---|---|---|---|---|
| 2 256 8 4 64 | 3.9e-3 | 7.8e-3 | 1.6e-2 | 3.1e-2 |
| 2 1024 8 4 64 | 3.9e-3 | 7.8e-3 | 1.6e-2 | 3.1e-2 |
| 2 2048 8 4 64 | 3.9e-3 | 3.9e-3 | 3.1e-2 | 3.1e-2 |
| 2 1024 8 8 64 (MHA) | 3.9e-3 | 3.9e-3 | 7.8e-3 | 7.8e-3 |
| 2 1024 8 2 64 (GQA 4:1) | 3.9e-3 | 3.9e-3 | 3.1e-2 | 6.3e-2 |
| 2 1024 4 4 128 | 3.9e-3 | 7.8e-3 | 7.8e-3 | 7.8e-3 |
| 2 1024 8 4 32 | 3.9e-3 | 3.9e-3 | 1.6e-2 | 1.6e-2 |

Evidence: `evidence/iter2_autotune.json` lines 8–84.

### Speed on GB10 — `B=8 T=2048 H=8 KV=4 D=64 bf16 causal`

| call | whale mean_ms | sdpa mean_ms | whale / sdpa |
|---|---|---|---|
| forward only | 0.488 | 0.682 | 0.72 (**1.40× faster**) |
| forward + backward | 2.512 | 3.089 | 0.81 (**1.23× faster**) |

Evidence:
- whale harness row: `evidence/whale_bench_custom_core.json` — steady 0.492 ms,
  33.29 M tok/s.
- sdpa harness row:  `evidence/whale_bench_sdpa_core.json`   — steady 0.687 ms,
  23.84 M tok/s.
- fwd+bwd: `evidence/iter2_autotune.json` lines 101–125 (whale), 126–150 (sdpa).

### Speed on GB10 — smaller shape `B=4 T=1024 H=8 KV=4 D=64`

Evidence: `evidence/iter2_autotune.json` lines 128–150.

| call | whale mean_ms | sdpa mean_ms |
|---|---|---|
| forward only | 0.107 | 0.129 |
| forward + backward | 1.001 | 0.753 |

- fact: For small shapes on GB10 the Triton backward is launch-overhead
  bound; SDPA's fused cuDNN backward wins by ~25 %.
- inference: On H100 with the pod's real batch size the relative picture is
  expected to match the larger shape, but that has to be confirmed on the pod
  before being claimed.

### `torch.compile` — no graph breaks, bit-exact match

Artifact: `test_compile.py`
- fact: `eager vs compiled  out max_abs_err = 0.000000`.
- fact: `eager vs compiled  dX  max_abs_err = 0.000000`.
- fact: `TORCH_LOGS=graph_breaks` emitted no break lines for the compiled
  forward+backward of a simple module wrapping `custom_whale_attn_fwd`.

## 3. What is not yet tested

- The full training gate (`legs/2026-04-16_whale_pr1493_triton_kernel/gate.sh`)
  with the new kernel. Needs multi-GPU and the pod stack.
- Pod-side H100 microbench.
- Dynamic shapes / variable seq_len (all benches use fixed T).
