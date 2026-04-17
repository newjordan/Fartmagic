# Hypothesis — whale `_attn_bwd_dkdv_kernel` TMA loads on top of early-exit

Date: 2026-04-17
Parent: `vault/whale_kernel_triton.py` `_attn_bwd_dkdv_kernel` (L455-L572, post early-exit split)
Primary shape: B=2, T=8192, H=8, KV=4, D=64 (matches Lever A bench)
Hardware: 1x H100 SXM, CUDA 13.0 / PyTorch 2.11.0+cu130 / Triton 3.6
Opt-in env: `WHALE_BWD_KV_TMA_LOADS=1` (0 = current kernel, unchanged)

## fact — current kernel uses plain pointer + mask loads
`_attn_bwd_dkdv_kernel` loads K, V, Q, DO, LSE, DELTA through raw strided pointers:

- K at L486-L488 / V at L487-L489 (outside the m-loop, once per program): `tl.load(ptrs, mask=k_mask, other=0.0)`
- Q at L517 / DO at L518 (masking phase) and L547-L548 (unmasked phase): `tl.load(ptrs, mask=q_mask, other=0.0)`
- LSE, DELTA at L522-L523 and L552-L553: 1-D masked loads

Each load emits generic `LDG.E.128 / cp.async` rather than Hopper bulk tensor copies, so the TMA unit is idle and the load path serializes through SM-issued addresses.

## fact — vault already has the TMA descriptor pattern
`_attn_bwd_dkdv_inline_delta_tma_kernel` (L686-L788) shows the exact recipe we will mirror:

- K/V descriptors (L714-L721) are built once per program (before the h/m loops), then `desc.load([pid_n * BLOCK_N, 0])` (L731-L732) pulls the tile through TMA.
- Q, O, DO descriptors (L747-L758) are built once per Q-head (inside `for hg`, before the m-loop), then `Q_desc.load([start_m, 0])` / `DO_desc.load([start_m, 0])` (L764-L766) pulls each M-block tile.
- `BLOCK_D == D` is a precondition; the descriptor's last stride is 1.
- LSE stays on plain `tl.load` (1-D, small, not worth a descriptor).
- `cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)` is already used identically in the inline-delta-tma variant (it sits on L743 as `m_end_block`; same arithmetic shape as the early-exit `m_masking_max`).

## proposal — TMA-ify the 4 bf16 bulk loads, keep early-exit structure

1. Once per program (before `for hg`): build `K_desc`, `V_desc`, replace the two `tl.load(..., mask=k_mask, ...)` calls with `K_desc.load([pid_n * BLOCK_N, 0])` and `V_desc.load([pid_n * BLOCK_N, 0])`. Same tile shape `[BLOCK_N, D]` (BLOCK_D == D enforced).
2. Once per Q-head (inside `for hg`, before both m-loops): build `Q_desc`, `DO_desc`.
3. Inside the **masking phase** m-loop: replace the Q/DO `tl.load(..., mask=q_mask, ...)` with `Q_desc.load([start_m, 0])` / `DO_desc.load([start_m, 0])`. Keep `row_mask`, `offs_m_cur`, and the P-predicate exactly as they are today — TMA only replaces the two bf16 tile loads.
4. Inside the **unmasked phase** m-loop: same substitution. The P-where drops to sequence-bounds only (already the early-exit behavior), and for `T_MAX % BLOCK_M == 0` it collapses at compile time.
5. Keep LSE and DELTA on plain `tl.load`; they are 1-D and TMA descriptors for vectors are overkill.
6. Wire `use_tma_kv_loads` in the python entrypoint (`custom_whale_attn_bwd`) so the caller can pass `use_tma_kv_loads=True` when `os.environ.get("WHALE_BWD_KV_TMA_LOADS", "0") == "1"`. If False, dispatch to the existing `_attn_bwd_dkdv_kernel` (unchanged); if True, dispatch to the new `_attn_bwd_dkdv_tma_loads_kernel`. Do NOT remove the existing kernel.

## inference — why TMA + early-exit stack multiplicatively

- Early-exit already removed the *arithmetic* on redundant M-blocks (no P-mask build, no `tl.where` on masked tiles). What's left in the unmasked phase is pure GEMM + exp + P@DO + DP + dK acc.
- In that pure-GEMM regime, HBM-load latency for Q and DO becomes a larger fraction of the budget (kineto showed the bwd gap at ~157us; commit f79d593). TMA loads cut the SM-issued address math and unblock the MMA pipe earlier — gains that compound with a tighter inner loop rather than competing with it.
- K, V are loaded exactly once; switching them to TMA costs one descriptor build per program, which amortizes across `group * m_range` m-blocks (~60+ on the primary shape). Descriptor cost is effectively free here.
- Q, DO descriptors are built once per Q-head (4 groups on primary shape), then reused across ~60 m-blocks each. Same amortization argument.

## risks

- **Descriptor build cost.** Each `tl.make_tensor_descriptor` emits a few instructions to construct the tensor-map. On short T (e.g. T=4096) the amortization window is smaller; we must verify the T=4096 shape doesn't regress.
- **16-byte alignment.** K/V/Q/DO base pointers `base + b * stride_b + head * stride_h` must be 16-byte aligned. For bf16 (2 bytes), strides on H and B come from a contiguous 4-D `[B, T, H, D]` tensor with D=64 or D=128 — both divisible by 8 elements (16 bytes). Safe for the 4 primary shapes. Add an assertion at the entrypoint.
- **Last-dim stride must be 1.** That's true for all our bwd inputs (contiguous D axis).
- **Early-exit constexpr folding.** `m_masking_max = tl.cdiv((pid_n + 1) * BLOCK_N, BLOCK_M)` is runtime (pid_n is runtime), so the `tl.minimum(m_masking_max, m_end_block)` stays a runtime `min`. TMA does not change that — same as the non-TMA early-exit kernel. No regression expected.
- **Autotune cache.** New kernel name, new cache. Flush `~/.triton/cache` once before the first after-bench.
- **BLOCK_D == D requirement.** The masked-head-dim path (D < BLOCK_D) is not supported under TMA. For D=64 and D=128 with BLOCK_D set equal to D this is fine; configs already pin `BLOCK_D = D` for the TMA config list (`_bwd_kv_inline_tma_configs`).
- **Fallback path.** Keep `WHALE_BWD_KV_TMA_LOADS=0` routing to the unchanged early-exit kernel, so a TMA regression on any shape can be escaped without a vault revert.

## success criterion

On primary shape (B=2, T=8192, H=8, KV=4, D=64):
- Fwd+bwd latency with `WHALE_BWD_KV_TMA_LOADS=1` drops by >= 15us vs the same gate run with `WHALE_BWD_KV_TMA_LOADS=0`.
- Numerics (max abs delta vs sdpa) within the existing tolerance in `legs/2026-04-16_whale_fast_kernels/bench_numerics.py` (fwd <= 5e-3, dQ/dK/dV <= 1e-2).

If the primary shape improves but T=4096 regresses, accept the leg with a note that the opt-in env defaults to 0 and the training path turns it on only at T >= 8192.

## non-goals
- No change to fwd, preprocess, or dq kernels.
- No change to autotune configs — reuse `_bwd_kv_inline_tma_configs` block list (BLOCK_D == D, maxnreg=224).
- No change to inline-delta or fused-bwd paths.
- No training change in this leg.
