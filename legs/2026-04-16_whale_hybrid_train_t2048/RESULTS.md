Parent: vault/train_gpt_midnight_iii_base.py
Leg: legs/2026-04-16_whale_hybrid_train_t2048

# Results

(empty -- pending pod execution by user)

Fill in after `bash legs/2026-04-16_whale_hybrid_train_t2048/run.sh`:

- log path: `logs/full_seed444_<TS>.log`
- step time mean (ms) over steps 100-end:
- final loss / val loss:
- baseline comparison (vs `legs/2026-04-14_midnight_iii_v_bank_gptq_fix` final
  log if available):
- confirmation that hybrid path actually engaged: `grep -i "whale_kernel_triton"
  logs/<log>` should NOT show ImportError.
