# Hypothesis — III_Lean

## One Variable
- Name: `NUM_LOOPS`
- New value: `0` (disable depth recurrence entirely)
- Baseline value: `2` (loop layers 3-5 twice, activate at 35% wallclock)

## Parent
- Source: `vault/train_gpt_midnight_iii_base.py`
- SHA256: `4d265579556279e3b0d652abf078fe762117227cd2408c9eca1afd81bdb15365`

## Why
Midnight III stacks 5 changes vs Midnight 12L. Four of them (QK_GAIN=5.25, MATRIX_LR=0.022,
MUON_WD=0.095, parallel residual layers 7-11) are non-architectural improvements that
shouldn't affect quantization. The fifth (depth recurrence) fundamentally changes the forward
pass and is the likely cause of the 0.45 BPB quant catastrophe.

This arm isolates the question: how much of Midnight III's pre-quant improvement comes from
the four non-loop changes alone? If the non-loop improvements beat Midnight 12L AND the quant
gap is normal, this becomes a viable submission path even if loop-aware GPTQ (III_Loop) fails.

SKIP_GPTQ=1 (naive quantization), same as baseline lane.

## Gate Pass Criteria
- 2xGPU screen (360s): final_sliding_window_exact BPB competitive with Midnight III (1.106).
- quant_roundtrip BPB shows a normal quant gap (not 0.45 BPB).
- If pre-quant BPB is worse but quant gap is small, the NET post-quant score may still beat leader.
