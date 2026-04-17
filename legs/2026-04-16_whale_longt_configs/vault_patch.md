# Lever D — vault patch spec (DOC ONLY; do not apply yet)

Target: `vault/whale_kernel_triton.py`. All changes are inside the eight
config-builder free functions. **No kernel rewrites.** No changes to any
`@triton.autotune(...)` decorator, no changes to call sites.

## Mechanism

`@triton.autotune` reads its `configs=` list **once at kernel-build time**;
the `key=` list (`[D, IS_CAUSAL, NUM_HEADS, NUM_KV_HEADS, T_MAX]`) only
selects *within* that fixed pool per call. We therefore cannot read `T_MAX`
inside a `_*_configs()` body and branch per-call.

The cheapest correct mechanism is to **gate the entire returned list** on a
process-level env: `os.environ.get("WHALE_LONGT_CONFIGS", "0") == "1"`.
Operationally:

- A run that targets long-T workloads exports `WHALE_LONGT_CONFIGS=1`
  *before importing the module*. The autotune pool for that whole process
  is the long-T-biased list.
- The existing `T_MAX` autotune key still does its job: within the
  long-T pool, autotune picks the best config per concrete `T_MAX` it
  sees. So the env doesn't *fix* a single config — it *narrows the pool*
  to long-T-friendly candidates.
- The `_env_force("FOO")` per-kernel single-config override remains the
  highest-precedence path (checked first, returns immediately). Lever D
  does not touch `_env_force` semantics.
- For mixed short+long-T workloads, `WHALE_LONGT_CONFIGS=0` (default)
  preserves status quo exactly.

Why not autotune-key-driven branching? Triton 3.6 evaluates the configs
list before the key is known; you'd need a separate kernel symbol per
T-bucket to do per-T pools, which is a kernel rewrite — out of scope.

## Long-T candidate pool (sample, from prompt + FA3 hopper bwd)

Tile/warp/stage tuples to share across all eight functions:
```
(BLOCK_M, BLOCK_N, num_warps, num_stages)
(128, 128, 8, 2)
(128, 128, 8, 3)
(256, 128, 8, 2)
(128, 256, 8, 2)
```

Per-function divergences:
- `_fwd_configs`, `_bwd_kv_inline_configs`, `_bwd_q_configs`: keep
  `maxnreg=160` / `maxnreg=224` exactly as today (different per func).
- `_fwd_tma_configs`, `_bwd_kv_inline_tma_configs`: TMA forces
  `BLOCK_D == D` so we drop the (256,128) and (128,256) tiles when
  `D > 64` is in scope; the four-tile sample above is fine for
  D in {64, 128} since BLOCK_D is set elsewhere.
- `_bwd_fused_configs`: keep the `WHALE_BWD_FUSED_MAXNREG >= 224` ValueError
  guard. Long-T list runs with no `maxnreg` by default, same as today.

## BEFORE / AFTER for the three highest-impact functions

### 1. `_fwd_configs` — vault/whale_kernel_triton.py:67-78

BEFORE (lines 67-78):
```python
def _fwd_configs():
    """maxnreg=160 saved ~3us on the headline fwd (legs/2026-04-16_whale_fwd_warpspec_tma)."""
    forced = _env_force("FWD")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (128, 64), (128, 128), (64, 128), (256, 64), (128, 256)]:
        for w in (4, 8):
            for s in (2, 3, 4, 5):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=160))
    return configs
```
Cardinality: 6 tiles x 2 warps x 4 stages = **48 configs**.

AFTER:
```python
def _fwd_configs():
    """maxnreg=160 saved ~3us on the headline fwd (legs/2026-04-16_whale_fwd_warpspec_tma).

    Lever D: WHALE_LONGT_CONFIGS=1 swaps the autotune pool to a long-T-biased
    set (FA3 hopper bwd uses BM=BN=128, num_stages=2, num_warps=8). See
    legs/2026-04-16_whale_longt_configs/."""
    forced = _env_force("FWD")
    if forced:
        return forced
    if os.environ.get("WHALE_LONGT_CONFIGS", "0") == "1":
        long_t = [(128, 128, 8, 2), (128, 128, 8, 3),
                  (256, 128, 8, 2), (128, 256, 8, 2)]
        return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                              num_warps=w, num_stages=s, maxnreg=160)
                for (bm, bn, w, s) in long_t]
    configs = []
    for bm, bn in [(64, 64), (128, 64), (128, 128), (64, 128), (256, 64), (128, 256)]:
        for w in (4, 8):
            for s in (2, 3, 4, 5):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=160))
    return configs
```
Cardinality (long-T branch): **4 configs**.

### 2. `_bwd_kv_configs` — vault/whale_kernel_triton.py:97-106

BEFORE (lines 97-106):
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
Cardinality: 6 tiles x 2 warps x 4 stages = **48 configs**.

AFTER:
```python
def _bwd_kv_configs():
    """Lever D: WHALE_LONGT_CONFIGS=1 narrows the pool to FA3-style long-T tiles.
    No maxnreg here (matches existing baseline-dkdv convention)."""
    forced = _env_force("BWD_KV")
    if forced:
        return forced
    if os.environ.get("WHALE_LONGT_CONFIGS", "0") == "1":
        long_t = [(128, 128, 8, 2), (128, 128, 8, 3),
                  (256, 128, 8, 2), (128, 256, 8, 2)]
        return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                              num_warps=w, num_stages=s)
                for (bm, bn, w, s) in long_t]
    configs = []
    for bm, bn in [(64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128)]:
        for w in (4, 8):
            for s in (2, 3, 4, 5):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s))
    return configs
```
Cardinality (long-T branch): **4 configs**.

### 3. `_bwd_q_configs` — vault/whale_kernel_triton.py:146-159

BEFORE (lines 146-159):
```python
def _bwd_q_configs():
    """Same tile search as the fwd kernel, but maxnreg=224 since the bwd dq
    path has heavier register pressure (see maxnreg sweep in
    legs/2026-04-16_whale_fwd_warpspec_tma)."""
    forced = _env_force("BWD_Q")
    if forced:
        return forced
    configs = []
    for bm, bn in [(64, 64), (128, 64), (128, 128), (64, 128), (256, 64), (128, 256)]:
        for w in (4, 8):
            for s in (2, 3, 4, 5):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=224))
    return configs
```
Cardinality: 6 tiles x 2 warps x 4 stages = **48 configs**.

AFTER:
```python
def _bwd_q_configs():
    """Same tile search as the fwd kernel, but maxnreg=224 since the bwd dq
    path has heavier register pressure (see maxnreg sweep in
    legs/2026-04-16_whale_fwd_warpspec_tma).

    Lever D: WHALE_LONGT_CONFIGS=1 swaps the pool to long-T-biased tiles.
    Note: this kernel does NOT use atomic_add into a scratch buffer
    (that's _bwd_fused_kernel), so maxnreg=224 is safe here."""
    forced = _env_force("BWD_Q")
    if forced:
        return forced
    if os.environ.get("WHALE_LONGT_CONFIGS", "0") == "1":
        long_t = [(128, 128, 8, 2), (128, 128, 8, 3),
                  (256, 128, 8, 2), (128, 256, 8, 2)]
        return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                              num_warps=w, num_stages=s, maxnreg=224)
                for (bm, bn, w, s) in long_t]
    configs = []
    for bm, bn in [(64, 64), (128, 64), (128, 128), (64, 128), (256, 64), (128, 256)]:
        for w in (4, 8):
            for s in (2, 3, 4, 5):
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                             num_warps=w, num_stages=s, maxnreg=224))
    return configs
```
Cardinality (long-T branch): **4 configs**.

## Remaining five functions — same pattern, abbreviated

For each, insert the same `if os.environ.get("WHALE_LONGT_CONFIGS","0") == "1":`
block immediately after the `_env_force(...)` early-return, returning the
4-tuple long-T list with that function's existing `maxnreg`/extra kwargs
preserved. Line ranges:

| Function                          | Lines     | maxnreg in long-T list | Notes                                                                 |
|-----------------------------------|-----------|------------------------|-----------------------------------------------------------------------|
| `_fwd_tma_configs`                | 81-94     | 160                    | TMA forces BLOCK_D==D; the four-tile sample is BLOCK_D-agnostic.      |
| `_bwd_kv_inline_configs`          | 109-125   | 224                    | Inline-delta dkdv; no scratch atomic_add, 224 is safe.                |
| `_bwd_kv_inline_tma_configs`      | 128-143   | 224                    | TMA + inline-delta dkdv; same as above.                               |
| `_bwd_kv_split_h_configs`         | 162-171   | (none)                 | No maxnreg today; keep none in long-T list.                           |
| `_bwd_fused_configs`              | 174-206   | `extra` (gated <224)   | Reuse `extra = {}` (or maxnreg-from-env <224) just like the legacy branch. |

## Validation gate before applying to vault

1. Run this leg's `run.sh` (Lever D off vs on) on the four shapes.
2. If on/off whale_fast fwd-or-fb mean improves >=10% at T=8192 with no
   regression >5% at T=4096, propose a tracked vault edit in a follow-up
   leg (or apply with explicit user approval; vault edits are out of
   scope here per CLAUDE.md).
3. If improvement is <10%, narrow the long-T list further or drop the
   lever; do not apply.

## Open questions

- Does `tl.constexpr T_MAX = 8192` actually compile a fresh kernel for
  T=4096 vs T=8192, or does Triton bucket on power-of-two? If the latter,
  the per-T autotune cache might already cover both with one entry — gate
  observation by reading `kernel.cache` after warmup.
- For shape `4,2048,16,16,64` (MHA, NUM_KV_HEADS == NUM_HEADS), the
  inline-delta dkdv has no group reuse — long-T pool may not help; keep an
  eye on that row.
- `_bwd_fused_configs` long-T list with no maxnreg: confirm that the
  reset_to_zero=["DQ_F32"] path stays correct under (256,128) tiles
  (atomic_add hot path).
