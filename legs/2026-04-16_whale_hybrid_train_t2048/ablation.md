Parent: vault/train_gpt_midnight_iii_base.py
Leg: legs/2026-04-16_whale_hybrid_train_t2048

# Ablation

| variable | parent value | this leg | rationale |
|---|---|---|---|
| attention forward kernel | `flash_attn_func` (FA3) | `whale_fwd_fa3_bwd` (whale Triton fwd + FA3 CUDA bwd) | bench shows -11% to -15% on the fwd at T<=2048 |

All other knobs identical to parent; tracked_env follows
`legs/2026-04-14_midnight_iii_v_bank_gptq_fix/tracked_env.sh`.

Backward path is unchanged from FA3 baseline (the hybrid backward calls
`flash_attn_interface._flash_attn_backward` with identical arguments to
what `flash_attn_func` would call internally).
