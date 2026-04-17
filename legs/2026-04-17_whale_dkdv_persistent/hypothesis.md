# whale_dkdv_persistent — hypothesis (H6)

## Goal

Test whether a **persistent** variant of `_attn_bwd_dkdv_kernel` (grid sized
to `NUM_SMS`, each program loops over assigned `(b, kv_head, n_block)` tiles
via `program_id(0) += num_programs(0)`) beats the current launcher, which
uses `grid = (cdiv(T, BLOCK_N), B*KV)` and therefore schedules
`B*KV*cdiv(T, BLOCK_N)` programs on the 132 H100 SXM SMs.

This leg is **not** the H5 persistent-atomic fused leg (see
`legs/2026-04-16_whale_bwd_persistent_atomic/RESULTS.md`). That one was
persistent scheduling **plus** fp32 `atomic_add` on dQ; it was ruled out
because `maxnreg>=224` corrupts `tl.atomic_add` (memory
`triton_gotchas_atomic_add.md`) and because atomic-add bandwidth on H100
is ~1–2 GB/s aggregate vs 3 TB/s HBM plain stores. This leg keeps dK/dV
plain-`tl.store`, dQ stays on its own kernel, so the atomic path is
**never** touched.

## Fact / inference / proposal split

### fact

- `fact`: `_attn_bwd_dkdv_kernel` launcher at
  `vault/whale_kernel_triton.py:1547` uses
  `grid_kv = (triton.cdiv(T, META["BLOCK_N"]), B * KV)`.
- `fact`: For the 2x8192x8x4x64 long-T shape (BLOCK_N=128 winner config),
  this yields `grid = (64, 8) = 512 programs`.
- `fact`: H100 SXM has 132 SMs, so 512 programs / 132 SMs ≈ 3.9 waves.
- `fact`: The kernel already has the causal early-exit
  (`m_start_block = (pid_n * BLOCK_N) // BLOCK_M` at
  `vault/whale_kernel_triton.py:497` and `:629`) — this leg **does not
  revert** that.
- `fact`: `legs/2026-04-16_whale_bwd_persistent_atomic/RESULTS.md` says
  "persistent fused dkdv+dq via fp32 atomic_add" is 2× slower. The
  slowdown was traced to atomic-add bandwidth, not to persistent
  scheduling itself. The dkdv-only persistent pattern has **no
  atomic_add**; the H5 negative does not apply.
- `fact`: `legs/2026-04-16_whale_bwd_ablations` identified ~140 µs of
  non-kernel overhead and ~157 µs of GPU gap to FA3 on the dkdv leg.
  Improved tile scheduling can address part of the GPU gap.

### inference

- `inference`: Across ~3.9 waves of dkdv programs, K/V tiles
  (`BLOCK_N × D` bf16 = e.g. 128×64×2 = 16 KB each) are re-loaded from
  HBM per wave because there is no guarantee a given SM retains its K/V
  tile in L2 across separate program launches. A persistent design with
  one program per SM iterating tiles in N-major order lets the same SM
  process adjacent N-blocks for the same `(b, kv_head)`, keeping Q/O/DO
  for shared M-blocks hot in L2 across inner loops.
- `inference`: L2 on H100 is 50 MB. A single (b, kv_head) slab of Q/O/DO
  at T=8192, D=64, bf16 is `8192×64×2 = 1 MB` per tensor, 3 MB total
  per `(b, h)` — easily L2-resident. Sequential N-block processing
  within the same `(b, kv_head)` should raise L2 hit rate on the Q/O/DO
  side by a wave count ratio (~3–4×).
- `inference`: FA3's Hopper backward uses a persistent + warp-specialized
  kernel and reports ~300 µs on this shape. The remaining 157 µs gap is
  dominated by L2-residency and dispatch-overhead, both of which
  persistent directly attacks.
- `inference`: The early-exit skip in causal mode reduces useful work
  per tile non-uniformly across N-blocks (low `pid_n` blocks do much
  less work than high `pid_n` blocks). A persistent loop with
  round-robin assignment naturally **load-balances** that imbalance:
  a fast-finishing SM picks up the next tile rather than idling for the
  tail wave.

### proposal

- `proposal`: Add `_attn_bwd_dkdv_persistent_kernel` — a near-copy of
  `_attn_bwd_dkdv_kernel` wrapped in an outer `while pid < num_tiles:
  ... pid += num_programs(0)` loop. Inside the loop, derive
  `pid_n = pid % cdiv(T, BLOCK_N)`, `bkv = pid // cdiv(T, BLOCK_N)` to
  preserve N-major ordering (same SM processes adjacent N-blocks of the
  same `(b, kv_head)`).
- `proposal`: Launch with `grid = (NUM_SMS,)` where `NUM_SMS` is read
  from `torch.cuda.get_device_properties(0).multi_processor_count` (H100
  SXM = 132).
- `proposal`: Gate with `WHALE_BWD_KV_PERSISTENT=1`. Default off until
  evidence lands.
- `proposal`: Benchmark on four primary shapes:
  - `2,8192,8,4,64`  (long-T, highest wave count → biggest expected win)
  - `4,4096,8,4,64`  (also long-T, high waves)
  - `4,2048,8,4,64`  (headline — waves ≈ 0.97, smallest expected delta)
  - `4,2048,16,16,64` (MHA, waves ≈ 4.8, load-balance test)

## Expected L2 / wall-time impact

- `inference`: Wall reduction 20–60 µs on long-T shapes (`2,8192,8,4,64`
  and `4,4096,8,4,64`) where K/V reload per wave dominates.
- `inference`: Near-zero delta on the headline (`4,2048,8,4,64`) because
  waves ≈ 1 already.
- `inference`: On MHA (`4,2048,16,16,64`), load-balance win may be
  larger than L2 win — FA3 is 1.6× faster on this shape per H2 sweep.

## Success criterion

`whale_fast fwd+bwd` wall improvement ≥ 15 µs on `2,8192,8,4,64`
(current `fused_delta` = 2.065 ms, target ≤ 2.050 ms; stretch ≤ FA3's
1.030 ms is out of this leg's scope).

Correctness within bf16 tol (|grad|<5e-2) on all four shapes.

## What this does NOT test

- TMA variant (covered by H2 leg `whale_bwd_tma_dkdv`).
- dq kernel persistent (separate leg).
- Fused dkdv+dq (H5, ruled out).
- Multi-GPU.

## Why this can stack with early-exit

`fact`: The early-exit skip (`m_start_block = (pid_n * BLOCK_N) // BLOCK_M`)
reduces per-program work, it does not touch scheduling. `inference`:
Persistent scheduling changes how the **remaining** tiles are assigned
to SMs. Both apply simultaneously; neither invalidates the other. If
anything, persistent mitigates the load imbalance that early-exit
creates (low pid_n tiles finish in 1–2 inner iters, high pid_n tiles in
cdiv(T, BLOCK_M) iters — round-robin persistence keeps all SMs busy
through the tail).
