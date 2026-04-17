# vault patch — _bwd_kv_configs() expansion

**DO NOT AUTO-APPLY.** `vault/whale_kernel_triton.py` is outside the
default scope of this leg (CLAUDE.md §Scope: "Never edit `vault/`
directly"). This file stages the patch for user review.

Target: `vault/whale_kernel_triton.py` L97-L106.

## BEFORE

```python
def _bwd_kv_configs():
    forced = _env_force("BWD_KV")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128)]:
        for w in (4, 8):
            for s in (2, 3, 4, 5):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s))
    return configs
```

## AFTER

```python
def _bwd_kv_configs():
    forced = _env_force("BWD_KV")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128)]:
        for w in (4, 8):
            for s in (2, 3, 4, 5):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s))
    # Long-T expansion (legs/2026-04-17_whale_dkdv_autotune_expand).
    # Adds deeper pipelining + wider-N tiles targeted at T>=4096. Shared-mem
    # calcs in that leg's hypothesis.md; all fit 228 KB on H100.
    for bm, bn, w, s in [
        ( 64, 128,  4, 6),   # #1  deeper pipeline, autotune-popular tile
        ( 64, 128,  4, 7),   # #2  deeper pipeline
        (128, 128,  8, 6),   # #3  FA3 tile choice + deeper pipeline
        ( 64, 256,  8, 3),   # #4  wide N, small M (new pair)
        ( 64, 256,  8, 4),   # #5  wide N, small M (new pair)
        (128, 256, 16, 3),   # #6  high warp count on widest tile
        (128, 256, 16, 4),   # #7  high warp count on widest tile
        ( 32, 512,  8, 2),   # #8  long-T edge: very wide N, single stage
    ]:
        configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                     num_warps=w, num_stages=s))
    return configs
```

## Rollback

If this patch regresses any shape in the 4-shape headline sweep, revert
to BEFORE. No other call sites change — `_bwd_kv_configs()` is consumed
only by the `@triton.autotune(configs=..., key=[...])` on
`_attn_bwd_dkdv_kernel`.

## Cache-invalidation note

Autotune cache key is by shape; adding configs changes the best-config
selection but does not invalidate already-compiled kernels for shapes
that don't touch the new configs. The `run.sh` in this leg clears
`/root/.triton/cache` before each shape to force a clean autotune.

## Verification checklist (user to run after apply)

1. `python3 -c "from vault.whale_kernel_triton import _bwd_kv_configs; print(len(_bwd_kv_configs()))"`
   should print `56`.
2. Smoke one shape with `WHALE_BWD_VARIANT=baseline` — autotune should
   complete without shmem-exceeded errors. Triton will emit a warning
   and skip any config it cannot compile; count compiled configs via
   `TRITON_PRINT_AUTOTUNING=1`.
3. Proceed with `bash legs/2026-04-17_whale_dkdv_autotune_expand/run.sh`.
