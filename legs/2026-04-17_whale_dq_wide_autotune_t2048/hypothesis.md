# Hypothesis — whale_dq_wide_autotune_t2048

`fact`: Profile leg 2026-04-17_whale_bwd_delta_tma_profile attributes the second-largest bwd kernel to `_attn_bwd_dq_inline_delta_kernel` at 44.46 µs/iter (total whale bwd = 150.25 µs vs FA3 bwd 64.33 µs — 2.3× gap).

`fact`: Existing `_bwd_q_configs()` (vault/whale_kernel_triton.py:177-190) autotunes BM×BN ∈ {(64,64),(128,64),(128,128),(64,128),(256,64),(128,256)} × warps ∈ {4,8} × stages ∈ {2,3,4,5}, maxnreg=224 fixed.

`hypothesis`: Existing dq grid is already wide. Quick sweep to confirm current winner is a real optimum, and test maxnreg reduction (160/192) since dq doesn't use atomic_add — the maxnreg=224 constraint from atomic_add-safety memo (`triton_gotchas_atomic_add.md`) does NOT apply here. Lower maxnreg may buy occupancy.

`proposal`: Sweep configs via `WHALE_BWD_Q_CONFIG` env override. Tighter grid (14 configs) focused on maxnreg variation and BM=256 check.

## Scope (one variable per run — the dq config)
- Shape locked: (2, 2048, 8, 4, 64), causal=True, bf16, H=8, KV=4, D=64
- WHALE_BWD_VARIANT=fused_delta_tma
- Best dkdv config from prior leg (whale_dkdv_wide_autotune_t2048) locked in via WHALE_BWD_KV_TMA_CONFIG
- Only `WHALE_BWD_Q_CONFIG` and `WHALE_BWD_Q_MAXNREG` vary

## Config grid (BM,BN,warps,stages,maxnreg)

```
128,64,4,3,224    128,64,8,3,224
128,64,8,2,224    128,64,8,4,224
128,64,8,3,192    128,64,8,3,160     # maxnreg var
128,128,8,3,224   128,128,8,3,192
256,64,4,3,224    256,64,8,3,224
64,64,4,3,224     64,128,8,3,224
128,256,8,3,192   128,64,8,5,224
```

## Success criteria
- 14 config runs complete, one JSON each
- Best config's `whale_fast.fb_mean_ms` recorded
- If any config beats the current dq kernel contribution (~44 µs GPU), apply both dkdv+dq winners in a validation leg
