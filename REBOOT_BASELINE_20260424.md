# Reboot Baseline - 2026-04-24

This file is the truth anchor after context/model drift. Use logged evidence only.

## Goal

Build a legal Parameter Golf SOTA Neural submission candidate that beats the current PR1493/current-SOTA control under the same evaluator, same tokenizer/data scale, same hardware shape, and legal artifact constraints.

Primary objective is not raw BPB alone. The winning candidate must survive:

- legal 10-minute-shaped training budget
- 16MB total submission size
- EMA
- GPTQ/quantization
- sliding-window eval
- TTT eval
- same-pod PR1493/current-SOTA comparison

## Verified Same-Pod 2x Control

Control: PR1493 same-pod anchor, seed 444, nproc=2, 30-minute proxy.

Log: `logs/codex_2x_pr1493_anchor_train_only_s444_20260424_070017.txt`

- raw stop: step 3636, `val_bpb=1.0968`
- post-EMA: `1.09594768`
- quantized: `1.10645179`
- sliding: `1.08980182`
- TTT: `1.08841468`
- total size: `15,991,013` bytes

Control behavior: artifact path is controlled. Quant hurts, but sliding/TTT recovers and final TTT beats raw/post-EMA.

## Verified Candidate Results

### h10 MLP5.0

Log: `blessed_whale/logs/codex_2x_final_eval_winners_20260424_172534/h10_l11_mlp50_final.log`

Config: B31 blessed whale, 11L, `MLP_MULT=5.0`, `XSA_LAST_N=10`, `PARALLEL_RESIDUAL_START=7`.

- raw stop: step 2585, `val_bpb=1.0839`
- post-EMA: `1.08630828`
- quantized: `1.11699347`
- sliding: `1.10048540`
- TTT: `1.09521693`
- total size: `17,086,622` bytes

Read: real raw win, but illegal and artifact-brittle. Does not beat PR1493 final TTT.

### h10 MLP4.5

Log: `blessed_whale/logs/codex_2x_h10_legalize_20260424_184418/h10_mlp45_final.log`

Config: B31 blessed whale, 11L, `MLP_MULT=4.5`, `XSA_LAST_N=10`, `PARALLEL_RESIDUAL_START=7`.

Current verified lines:

- raw stop: step 2684, `val_bpb=1.0880`
- post-EMA: `1.08989372`
- quantized: `1.12236883`
- sliding: `1.10576651`
- total size with raw-code accounting: `15,997,247` bytes

Read: much closer to legal size, but quant damage is still severe. TTT result was not in the verified grep at the time this file was written; check log before making final judgment.

## Current Failure Mode

Blessed whale currently has raw power, but not artifact discipline.

Compared to PR1493:

- PR1493 final path improves from quantized to sliding/TTT and lands at `1.08841468` legally.
- h10 MLP5.0 starts better raw but quantizes poorly, remains worse after TTT, and is too large.
- h10 MLP4.5 is near legal size but still shows worse quant damage.

Therefore the active problem is artifact stability and legal byte allocation, not broad raw-model search.

## Minifier Truth

`python_minifier` is installed. The safe minifier patch compiled, but on `pr1493_decompressed.py` it saved only 81 bytes. More aggressive hoist-literal settings saved about 1079 bytes in an offline compile check. This is not enough by itself to solve h10 MLP4.5 if it is 7247 bytes over. Code-size work may still help, but it is not the main win.

## Active Runs At Reboot

2x pod: `codex_2x_h10_legalize` still running.

1x pod: `codex_1x_quant_stability_probe` still running.

Do not launch overlapping jobs on those GPUs without checking process state.

## Next Gate

1. Finish/pull the active 2x h10 legalize result, including TTT.
2. Finish/pull 1x quant-stability diagnostics.
3. Judge only against PR1493 final TTT `1.08841468` and size `15,991,013` on the same-pod proxy.
4. If h10 MLP4.5 is close but fails quant, test stabilization/winddown/EMA/clip changes that specifically reduce post-EMA-to-quant and post-EMA-to-TTT damage.
5. Do not spend 8x until a candidate has a credible same-pod path through artifact scoring, not just raw BPB.
