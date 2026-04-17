# whale_dkdv_autotune_expand — hypothesis

Parent leg: `legs/2026-04-16_whale_long_t_profile` (H6).
Parent finding: at T=8192,(B=2,H=8,KV=4,D=64), `_attn_bwd_dkdv_kernel`
dominates at 747 us/iter (kineto: `evidence/profile_whale_auto_20260416_231025.json`),
vs FA3's backward closing the gap. Phase-3 force-config sweep showed
autotune is picking a sensible config for the *existing search space*
(fb deltas between forced configs were within 25% of the autotune mean).
So the lever is **expanding the search space**, not forcing within it.

## Fact (evidence path)

- `fact` Phase-3 forced configs at (2,8192,8,4,64) produced fb means:
  - `128,128,8,2` -> 1.839 ms  (evidence: `p3_kv_128_128_8_2_20260416_231908.json`)
  - `128,128,8,3` -> 1.800 ms  (evidence: `p3_kv_128_128_8_3_20260416_231908.json`)
  - `256,128,8,2` -> 2.244 ms  (evidence: `p3_kv_256_128_8_2_20260416_231908.json`)
  FA3 fb at same shape = 1.045 ms.
- `fact` Autotune fb at same shape (Phase-1 `longt_baseline`) was in the
  same 1.8–1.85 ms band, i.e. autotune is already near the best in-space
  config. Gap to FA3 is 0.75 ms / 75%.
- `fact` kineto breakdown at T=8192: dkdv=747us, fwd=462us, dq=448us.
  dkdv is the single largest kernel.
- `fact` Current `_bwd_kv_configs()` space (L97 of
  `vault/whale_kernel_triton.py`): BLOCK_M∈{64,128,256}, BLOCK_N∈{64,128,256}
  with the 6 tile pairs listed in the task context; num_warps∈{4,8};
  num_stages∈{2,3,4,5}. 48 configs total.
- `fact` H100 shared-memory cap = 228 KB per SM. BF16 kernels on D=64.
- `fact` Phase-3 candidates (256,128,8,3), (128,256,8,2), (128,256,8,3),
  (64,128,4,4) did not complete (only 3 of 7 configs have evidence
  json), so we do not have force-config data for N=256 or stages=3-with-wider-M.

## Inference

- `inference` The *type* of config missing from the current space that
  is most likely to help at long T is **deeper software pipelining**
  (num_stages 6) or **wider N tiles with modest M** (amortises the inner
  dk/dv f32 accumulator across more K rows per SM launch).
- `inference` num_warps=16 is only useful at BLOCK_M*BLOCK_N >= 128*256;
  below that warps outnumber meaningful work chunks and the MMA pipeline
  can't keep them fed.
- `inference` BLOCK_M=256, BLOCK_N=256 at D=64 exceeds the register +
  shmem budget for deep pipelining: the accumulator alone is
  256*256*4 = 256 KB (f32), larger than total SM shmem. Triton would
  spill to registers/global and almost certainly lose. **Reject (c).**
- `inference` num_ctas=2 (cluster on Hopper) could cooperatively load K/V
  across 2 SMs and halve per-SM shmem pressure, but dkdv's grid already
  parallelises over (batch, kv_head, n_block) and we have plenty of
  programs for SM occupancy at T>=4096. Cluster is more useful when the
  grid is small. **Defer (d)** — mark as a follow-up leg if the
  stage/tile expansion here doesn't close >40% of the gap.
- `inference` BLOCK_N=512 with small BLOCK_M is attractive for long-T but
  the K+V tiles alone are 512*64*2*2 = 128 KB, which leaves < 100 KB for
  pipelined Q/dO — only 1 pipeline stage fits. Worth testing exactly
  one such config at the edge of the search space.

## Shared-memory bound

Conservative formula for `_attn_bwd_dkdv_kernel` with BF16 inputs, D=64:

```
shmem_bytes ≈ BLOCK_N*D*2        # K tile (loop-invariant)
            + BLOCK_N*D*2        # V tile (loop-invariant)
            + num_stages*BLOCK_M*D*2   # Q pipelined
            + num_stages*BLOCK_M*D*2   # dO pipelined
```

(The BLOCK_M*BLOCK_N f32 tile lives in registers inside the MMA path,
not shmem. Triton will reduce effective pipeline depth if this bound
would exceed 228 KB, but we keep to the explicit bound here.)

Cap = 228 * 1024 = 233472 bytes.

## Proposed new configs (<=8) and shmem check

Targeting long-T sweet spots. All at D=64 BF16.

| # | BM  | BN  | warps | stages | shmem bytes                                  | fits? |
|---|----:|----:|------:|-------:|---------------------------------------------:|:-----:|
| 1 |  64 | 128 |     4 |      6 | 2*128*64*2 + 6*2*64*64*2   = 32768 + 98304 = **131072** | YES |
| 2 |  64 | 128 |     4 |      7 | 2*128*64*2 + 7*2*64*64*2   = 32768 + 114688 = **147456** | YES |
| 3 | 128 | 128 |     8 |      6 | 2*128*64*2 + 6*2*128*64*2  = 32768 + 196608 = **229376** | borderline (98%); Triton will auto-shrink stages if over |
| 4 |  64 | 256 |     8 |      3 | 2*256*64*2 + 3*2*64*64*2   = 65536 + 49152  = **114688** | YES |
| 5 |  64 | 256 |     8 |      4 | 2*256*64*2 + 4*2*64*64*2   = 65536 + 65536  = **131072** | YES |
| 6 | 128 | 256 |    16 |      3 | 2*256*64*2 + 3*2*128*64*2  = 65536 + 98304  = **163840** | YES |
| 7 | 128 | 256 |    16 |      4 | 2*256*64*2 + 4*2*128*64*2  = 65536 + 131072 = **196608** | YES |
| 8 |  32 | 512 |     8 |      2 | 2*512*64*2 + 2*2*32*64*2   = 131072 + 16384 = **147456** | YES |

Rationale per group:

- **#1, #2 (64,128,4,{6,7})**: keep the autotune-popular tile but push
  pipeline depth further than current `stages<=5`. Long-T dkdv is mem-BW
  bound per Phase-2; deeper pipelining hides HBM latency on Q/dO loads.
- **#3 (128,128,8,6)**: FA3's hdim64 choice *with* deeper pipelining.
  Borderline shmem — listed so autotune can either pick it or Triton can
  auto-shrink to an effective stages<6. Acceptable since current (128,128,8,5)
  already passes the compile stage by the same mechanism.
- **#4, #5 (64,256,8,{3,4})**: wider N tile with small M. Amortises the
  dK/dV accumulator (register-resident) across 256 K rows per inner loop.
  Missing from current space — current space has BN=256 only paired with
  BM=128 (= 128,256) which has different occupancy.
- **#6, #7 (128,256,16,{3,4})**: the only pair of tiles in the current
  space where num_warps=16 is justifiable (BM*BN=32768, >= 8 warps per
  128-thread MMA tile). Tests whether higher warp count helps hide
  latency at long T on the widest current tile.
- **#8 (32,512,8,2)**: long-T edge case — very wide N, small M, single
  pipeline stage. Explores the extreme of "stream as many K rows per
  SM as possible."

## Proposal

- `proposal` Add the 8 configs above to `_bwd_kv_configs()`. Total
  count becomes 48 + 8 = 56. Compile-time budget roughly linear — within
  the "10-config cap is ~safe" guidance only if autotune is restricted
  to one shape at a time; the cache key hashes on shape so the 56
  configs only compile once per shape class per process. Acceptable.
- `proposal` Bench at 4 shapes (same as `queue_early_exit_headline.sh`)
  plus explicit compare vs FA3: `(2,8192,8,4,64) (2,4096,8,4,64)
  (2,2048,8,4,64) (1,16384,8,4,64)`. Focus is long-T but 2048 kept as
  regression canary.
- `proposal` `WHALE_BWD_VARIANT=baseline` — matches Phase-1 finding that
  baseline wins at long T.
- `proposal` **Do NOT auto-apply the vault patch.** Patch is staged in
  `vault_patch.md` with full BEFORE/AFTER block; user must approve
  before any vault edit. (`vault/` is a frozen-by-default path per
  CLAUDE.md.)

## Success criteria

- Primary: >=20% reduction in dkdv kernel us/iter at (2,8192,8,4,64),
  corresponding to >=0.15 ms off the fb time.
- Secondary: no regression at (2,2048,8,4,64) vs current autotune.
- Stretch: close >=40% of the FA3 fb gap at T>=4096.

## Not in scope

- TMA on dkdv (separate leg).
- num_ctas>1 / cluster cooperation (deferred; see inference above).
- BLOCK_M=256 BLOCK_N=256 (rejected above on shmem grounds).
- Touching `_bwd_kv_inline_configs` or `_bwd_q_configs` — single variable
  per leg per CLAUDE.md.
