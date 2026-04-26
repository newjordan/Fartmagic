# Next

## 2026-04-26 Whale Donor Lock

Work from this donor:

```text
2026-04-15_whale_l12_dim576_matrixbits5
```

This is the active best-shot donor. Preserve its record and rebuild from clean
source only.

Canonical record/config:

- `legs/2026-04-15_whale_l12_dim576_matrixbits5/RESULTS.md`
- `legs/2026-04-15_whale_l12_dim576_matrixbits5/tracked_env.sh`
- `evidence/pod_pulls/instance_35002131_20260416_1113/logs/whale_l12_dim576_matrixbits5_s444.txt`

Do not use `gemini_runs/`; it was deleted by request. The next useful action is
to re-establish a clean donor source/body for the April 15 donor record, then
run new dated `run.py` legs from that clean source.

Immediate run:

```bash
cd /workspace/SOTA_FINAL/legs/2026-04-26_whale_donor_baseline_4x_full
/venv/main/bin/python3 run.py
```

This is the clean 4x mechanics baseline for the active donor.

## 2026-04-26 Whale + PR1797 Smoothing Pivot Override

Use `WHALE_1797_SMOOTHING_PIVOT_20260426.md` as the current direction.

Next work is not a CaseOps reproduction launch. The tokenizer/data legality
concern blocks that path.

Required order:

1. Treat `2026-04-15_whale_l12_dim576_matrixbits5` as the active donor record.
2. Reconstruct or re-anchor the donor from clean, non-`gemini_runs` source.
3. Only after that, prepare new dated `run.py` tests.
4. Prepare LQER asymmetric post-GPTQ correction as a separate later graft after
   a clean donor source/body is re-established.
5. Do not stack CaseOps, SmearGate, LQER, TTT warm-start, and PR1787 optimizer
   changes together.

## DANGERZONE FOCUS LOCK

Do not drift.

The mission is to make the whale donor/blessed-whale path a legal world-record
candidate. The immediate gate is not B34 and not final packaging; it is finding
or building a proper body donor that beats the current SOTA control on the
hardware we are using. Legal 16MB quantization/compression remains required
later, but only after we have a body worth compressing.

Main SOTA Neural gate: beat PR1493/current SOTA control on the same hardware,
same data, same evaluator, and same wallclock. Do not judge 1x/2x proxies
against the `1.079` 8x/public-region number. The `~1.10` raw threshold is only
for the 4k exploratory arm to decide whether it is worth discussing
eval/compression later.

Control discipline:

- The only user-facing control is PR1493 or our current best anchor, rerun on
  the same hardware shape as the candidate.
- Internal ablation controls are lab bookkeeping only. Do not mention them in
  status, and do not use them to claim progress.
- If a test is not better than PR1493/current-best on the same evaluator scale,
  keep it out of the user-facing narrative.

Current decision tree:

1. Run body candidates only when they have a credible path past the same-pod
   PR1493/current-best control.
2. Compare every promoted body candidate to PR1493 or our current best anchor on
   the same hardware, pod data, evaluator scale, seed policy, and wallclock.
3. If it does not clear that anchor, do not discuss it as progress.
4. Current main line is the edge candidate plus PR1493/current-best anchor.
   The 4k vocab leg is now promoted only as a kernel-backed architecture search:
   pinned 4k tokenizer/dataset, blessed whale fusion, and no compression spend
   until the raw curve has a credible path past the anchor.

4k-vocab status:

- The user-created original 4k tokenizer exists at
  `/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_4096_bpe.model`,
  SHA256 `6b0337698df13acdb58e5b05446ef26d9d7e4ca5545dc5e13ad5815a7ec904e0`.
- The matching local SP4096 dataset exists at
  `/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp4096/`.
- The pulled 2x-pod tokenizer is different:
  `e5b31cf3d3bd87f68998ded73e8f7f9bdaea998bee4899461ab4f4d47f7b4987`.
  Do not mix these up.
- Scripts existed: `test_4k_vocab.sh`, `test_4k_vocab_v2.sh`,
  `test_4k_vocab_v3.sh`, `test_4k_vocab_v4.sh`.
- The April 23 autoloop reruns are not decision evidence; do not discuss them as
  quality results.
- Conclusion: the original 4k vocab is still an open breakthrough candidate,
  not a rejected lane. Retest only with the pinned original tokenizer hash and a
  PR1493/SP8192 anchor on the same evaluator.

4k Kernel Architecture Search:

1. Stage the original 4k tokenizer and matching SP4096 dataset, pinned by hash.
2. Run one single-GPU smoke to verify tokenizer/dataset/trainer plumbing.
3. Queue the broad blessed-whale fusion architecture sweep from
   `blessed_whale/tmp_codex_4k_big_arch_sweep.sh`.
4. Do not assume D72 is optimal. The broad pass covers depths `8/10/12/14/16`,
   widths `384/512/576/640/704/768`, D64/D72/D88 head shapes, and MLP pressure
   from `2.75` to `6.0`.
5. Let the custom kernel follow the winning H100 shape; kernel evidence is a
   requirement, not a reason to freeze the architecture too early.
6. Do not spend final-eval/compression time unless the 4k kernel leg has a
   credible path against PR1493/current-best.
7. Midnight/Rat Rod follow-up is now prepared as
   `blessed_whale/tmp_codex_4k_midnight_wide_sweep.sh`. It keeps the pinned
   4k tokenizer and blessed-whale kernel, then tests the missing shallow/wide
   branch: D64 heads, `512-1024d`, `4-8` physical layers, high MLP, and valid
   recurrent loop spans. This is the right way to borrow from the old 1k line
   without reviving it as the primary lane.
8. 4k-arm gate is raw neural only for now. Keep 4k runs train-only with
   `SKIP_POST_TRAIN=1` and `GPTQ_RESERVE_SECONDS=0`. Do not discuss eval,
   compression, quant, sliding, or TTT for the 4k arm until a raw 4k curve gets
   around `1.10` BPB on a comparable competition-shaped scale.
9. Do not use the 2x pod for 4k runs. The 2x pod belongs to main SOTA Neural
   proxy work unless the user explicitly changes that allocation.

Midnight-line carry-forward:

- Do not restart the midnight lineage as the primary body. Its best headline
  `1.09159026` qattn6 score was not clean artifact proof because GPTQ calibrated
  `0` layers and reloaded quant was around `1.47`.
- The corrected midnight export path proved the important lesson: loop-state
  restore and reloaded-quant scoring matter. It did not produce a `~1.08`
  candidate.
- If we borrow from midnight, borrow process and byte-allocation lessons:
  enforce loop-state restore, hard-fail zero-layer GPTQ calibration, require
  quant-roundtrip/sliding/TTT from the reloaded artifact, and try aux/embed/other
  byte relief before touching attention or MLP.
- Rat Rod / GreenRod architecture lesson: the remembered strong 1k-vocab line
  was real and useful as hypothesis fuel. Rat Rod Green v1 used `512d`, `11L`,
  `8` heads, `4` KV, `MLP_MULT=3.0`, all-layer XSA, `ROPE_DIMS=16`,
  BigramHash `2048x128`, EMA/SWA, and fast Muon-style training. Its failed
  additions were Synapse/HS-MTP bridge, complement loss, value residual, and
  shared trigram. For 4k blessed whale, test the shallow/wide geometry first;
  only port BigramHash/local-context code if geometry produces a credible raw
  curve.

## Live pod order - 2026-04-24

Keep pods hot only for anchor-relevant work. Do not launch overlapping
torchruns on the same GPU set. User-facing status should mention only:

- PR1493/current-best anchor runs.
- 4k pressure/body candidates with a credible path past the anchor.
- Final artifact candidates that clear the anchor.

Current 2x order:

1. Finish the active same-pod PR1493/current-SOTA anchor on the 2x H100 pod.
2. Compare it directly against
   `codex_2x_main_raw_b31_mom95_mlp45_20260424_062205`, which reached
   step `2504`, `val_bpb=1.0866` under the same 30-minute train-only proxy.
   The same-pod PR1493 anchor stopped raw at step `3636`,
   `val_bpb=1.0968`, so the candidate currently has a raw proxy edge of about
   `0.0102` BPB.
3. Let `codex_2x_stalemate_breaker_after_anchor` start only after the anchor
   torchrun exits. It runs
   `blessed_whale/tmp_codex_2x_stalemate_breaker_queue.sh`: 15 sequential
   train-only hypotheses around loop onset/span, XSA depth, MLP pressure,
   layer/width allocation, Muon momentum/warmup, warmdown, and QK gain. The
   script is unattended-safe: it keeps going after failed candidates and uses
   a `2700` second per-candidate timeout above the `1800` second train cap.
4. Promote only candidates that beat the same-pod anchor curve. Everything
   else is lab bookkeeping.

Next serious compression move, only after the raw body gate:

1. Stop global clip sweeps.
2. Attack non-uniform byte allocation by tensor group/layer, or reduce code
   bytes enough to buy a targeted quant relief.
3. Re-run only on corrected tokenizer hash
   `0518df93315eaff6bc75ad4d9711b01c1b5e9ac32e8b3da603c040d8653f7f03`.

## RED HOT NEXT ACTION

The pod is refreshed and the stale-tokenizer BPB scale bug is fixed. `B34` is
the first corrected-scale final-eval result for the B31/mom95 signal.

Latest corrected-scale result:

- `B34`: donor/no-kernel + `MUON_MOMENTUM=0.95`, seed `444`
- train stop: step `2350`, `val_bpb=1.0938`
- post-EMA: `val_bpb=1.09713879`
- quantized: `val_bpb=1.12773448`
- quantized sliding: `val_bpb=1.11096944`
- quantized TTT: `val_bpb=1.10557300`
- total submission size: `16008038` bytes, over the 16MB cap by `8038` bytes
- local evidence: `evidence/pod_pulls/35372056_2026-04-23_b34_refresh_result/`

This is not the winner yet, but it is now a real comparison on the correct
competition scale.

Immediate work order:

1. Do not spend more 8x time on broad B26-B35 sweeps.
2. Make the B34 artifact legal: remove at least `8038` bytes, preferably by
   reducing code/package overhead instead of weakening the model.
3. Recover training time. B34 reserved `180s` for GPTQ/post-eval and only
   reached step `2350`; the original B31 scout reached step `3358`.
4. Run a narrow corrected-scale promotion lane after size/time fixes, not before.
5. Only revisit kernel fusion after donor mom95 final-eval is legal and
   competitive.

Historical signal to preserve:

- 8x scout result: `B31` donor + `MUON_MOMENTUM=0.95`, step `3358`, `val_bpb=1.1384`.
- It beat `1.14` and beat the same-run donor control `B26=1.1406`.
- It is not yet a legal final submission score because it skipped post-train quant/sliding/TTT.
- It was captured before the tokenizer refresh, so treat it as a mom95 delta
  signal, not a final score.

Sanity evidence to preserve:

- Upstream PR1493 record trainer copied to the pod as `/workspace/SOTA_FINAL/pr1493_upstream_record.py`.
- SHA256 is `477f432097ce619d4080af0a249d29e6e0bf31b7a747ec4e6b509b5be2e202ee`, same as local `pr1493_raw.py`.
- PR1493 run log: `/workspace/SOTA_FINAL/logs/pr1493_upstream_sanity_seed42_20260423.log`.
- It reproduced training loss (`val_loss=2.8112`, close to published `2.8119`) but not PR1493's submitted BPB while the stale tokenizer was present (`1.1499` raw stop and `1.14153786` quantized TTT vs submitted about `1.0886` raw stop and `1.08079352` TTT).
- After refreshing the tokenizer, B34 step 0 returned to the expected scale:
  `val_bpb=3.4864`.

## Current priority

The active objective is not to re-prove the old `1.09159026` lane. It is to turn the whale donor plus blessed whale kernel path into a legal record candidate against the current public world-record region around `1.08`.

Known truth to preserve:
- raw whale donor/non-kernel build has reached about `1.079` BPB
- quantization/compression currently pulls the final submission-facing result back to about `1.09`
- blessed whale is the donor/kernel fusion foundation for the world-leading lane
- the active order is tame it, optimize it, then add compression/final eval
- full 20k-step runs are unlimited-compute science only; legal validation must be 10-minute, 8xH100-shaped, artifact-size-aware, and post-train scored

## Immediate sequence

1. Freeze `legs/2026-04-13_midnight_iii_v_qattn6` as the old promoted local reference, not the active ceiling.
2. Tame blessed whale:
   - make the live exact-donor lane clear startup, step-0 eval, loop enable, and real post-loop training
   - keep checkpoint/resume probes diagnostic only
   - do not score broken or partial lanes
   - executable scout: `bash blessed_whale/run_foundation_sequence.sh` runs `B26` donor control, `B27` fusion, then `B28` x180 fusion
   - narrow reruns use `FOUNDATION_BENCHES="B27" bash blessed_whale/run_foundation_sequence.sh`
3. Establish kernel/donor infusion baselines:
   - donor/no-kernel legal-shaped control
   - blessed whale kernel legal-shaped lane
   - same seed, same wallclock, same post-train scoring, exact byte accounting
4. Optimize blessed whale:
   - close or reverse any throughput deficit versus donor/no-kernel control
   - preserve the raw donor quality signal
   - avoid compression/eval changes until the kernel foundation is stable
   - first recovered neural knob to test: `MUON_MOMENTUM=0.95` from `blessed_whale/RECOVERED_NEURAL_ABLATIONS.md`
   - second recovered knob only if momentum survives: `MLP_MULT=3.5`, with size and final-score accounting
5. Add compression/final eval only after the fusion lane is stable and competitive.
   - explicit final-eval mode: `FOUNDATION_INCLUDE_FINAL_EVAL=1 bash blessed_whale/run_foundation_sequence.sh` appends `B29` donor final-eval and `B30` fusion final-eval
   - B34 has now run once after tokenizer refresh; next corrected-scale run should include the size/time fixes above
6. Only then lock final whale donor calibrations from the raw `~1.079` donor lineage.
7. Run merge-safe eval strategy first (see `legs/2026-04-13_midnight_iii_v_postquant_ctrl/EVAL_STRATEGY.md`):
   - Promotion lane uses `final_sliding_window_exact` + `final_quant_roundtrip_exact` + artifact bytes.
   - Keep `NGRAM_EVAL_ORDER=0` for promotion decisions.
8. Keep all sweep changes in each leg's `tracked_env.sh` or `train_gpt.py` (one variable per leg).
9. First merged-lineage expansion test from `legs/2026-04-13_midnight_iii_v_qattn6`:
   - child leg variable: `BIGRAM_VOCAB_SIZE=2816`.
10. If the bigram-vocab leg is competitive, make a child with one additional variable:
   - `BIGRAM_DIM=160`.
11. Run `python3 scripts/leg_diff_guard.py legs/<leg>` before each gate/run.
12. Record each sweep in that leg's `ablation.md` and `RESULTS.md` with exact log line citations.

## 2x queue current leader - 2026-04-24

Current queue leader is `h08_mlp50`: raw `1.0840`, beating prior 2x edge
`1.0866` and same-pod PR1493 anchor `1.0968`. Next action after the queue
finishes: inspect h08 config/log first, then decide whether to run a narrow
MLP pressure refinement around `MLP_MULT=5.0` rather than broad drifting.

## Next narrow lane from 2x queue

If no later queue candidate beats `h10_l11_mlp50`, promote a narrow MLP-pressure
refinement: compare 12L/11L around `MLP_MULT=4.75/5.0/5.25`, same seed and
same wallclock, then run final-eval only for the winner. Treat `h10` over `h08`
as a tie until repeated; the robust signal is high FFN capacity, not exact
layer count.

## Post-2x-queue promotion order

When `codex_2x_stalemate_breaker_20260424_075754` finishes:

1. Pull the complete queue directory and rebuild a clean results table.
2. Audit the top logs first: `h10_l11_mlp50` and `h08_mlp50`.
3. Treat `h10` vs `h08` as a tie unless the final queue produces a clear later winner. The robust signal is MLP pressure around `5.0`, not exact 11L vs 12L yet.
4. Run a narrow confirmation/refinement before broad compression: repeat/compare the MLP-pressure winners under the same anchor discipline.
5. Only after a raw winner survives confirmation, run final-eval/compression lockdown on that shape: EMA, GPTQ, sliding, TTT, artifact bytes, and code-size accounting.
6. If no 8x is available, 2x final eval is only a mechanics smoke. The real promotion must be 8x/legal-shaped before any record claim.

## 4k next from adaptive bracket

4k bracket selected the low-MLP side: `MLP3.5` beat `4.25` and `5.0` on
512d/11L. Let the low-side refinements finish (`MLP3.0`, `3.75`, and 10L/3.5).
If low-side remains best but still far from useful, the next real 4k move is not
more width; it is porting the 1k Rat Rod local-context/BigramHash mechanism.
