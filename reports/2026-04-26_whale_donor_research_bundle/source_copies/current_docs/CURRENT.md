# Current

## 2026-04-26 Whale Donor Lock

Active donor is now locked to:

```text
2026-04-15_whale_l12_dim576_matrixbits5
```

This is the best current shot and the baseline record to preserve.

Canonical record/config:

- `legs/2026-04-15_whale_l12_dim576_matrixbits5/RESULTS.md`
- `legs/2026-04-15_whale_l12_dim576_matrixbits5/tracked_env.sh`
- evidence log:
  `evidence/pod_pulls/instance_35002131_20260416_1113/logs/whale_l12_dim576_matrixbits5_s444.txt`

Record metrics:

- raw stop: `3295/20000`
- raw bpb: `1.0802`
- post-EMA raw bpb: `1.07952782`
- quantized bpb: `1.11723031`
- quantized sliding bpb: `1.10039777`
- quantized TTT bpb: `1.09561039`
- artifact bytes: `15,995,671`

Important containment:

- `gemini_runs/` has been deleted locally and on the pod. Do not restore it.
- The actual preserved binary for `2026-04-15_whale_l12_dim576_matrixbits5`
  has not been found in non-vault repo paths; treat the record/config/logs as
  source of truth until a clean donor source/body is re-established.
- All next work should protect and reconstruct this donor line, not pivot away
  from it.

Current clean 4x baseline runner:

- `legs/2026-04-26_whale_donor_baseline_4x_full/run.py`
- source body: `pr1493_decompressed.py` copied into the leg as `train_gpt.py`
- trainer SHA256:
  `f583502d8dbc23c37c0c81a2d309e14391b74dd128b7a1259fd4512604d2cfb7`
- pod path:
  `/workspace/SOTA_FINAL/legs/2026-04-26_whale_donor_baseline_4x_full`
- launch:
  `cd /workspace/SOTA_FINAL/legs/2026-04-26_whale_donor_baseline_4x_full && /venv/main/bin/python3 run.py`

## 2026-04-26 Whale + PR1797 Smoothing Pivot Override

Current direction is now recorded in `WHALE_1797_SMOOTHING_PIVOT_20260426.md`.

- CaseOps/tokenizer adoption is blocked until explicitly cleared.
- PR #1797 uses a different 8192-token model than PR #1493:
  `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`, with CaseOps
  operator IDs and byte-sidecar BPB accounting.
- Keep the whale donor body and standard tokenizer/data path.
- Mine PR #1797 only for portable smoothing or packaging techniques.
- Active donor is `2026-04-15_whale_l12_dim576_matrixbits5`.
- No active staged SmearGate launch exists.
- First likely portable target remains LQER asymmetric post-GPTQ correction,
  but only after the donor source/body is cleanly re-established.
- No CaseOps data prep or launch while legality is unresolved.

## DANGERZONE FOCUS LOCK - Do Not Drift

- Mission: turn the whale donor/blessed-whale path into a legal world-record
  candidate.
- Current record-region target: about `1.08` BPB final submission score.
- We have historical raw whale-donor evidence around `1.079`, but that number
  only applies on comparable 8x/legal-scale evidence. On 1x/2x pods, do not ask
  whether a candidate hits `1.079`; ask whether it beats PR1493/current SOTA
  control on the same hardware, same data, same evaluator, same wallclock.
- Main SOTA Neural gate: every proxy result must be reported as a delta versus
  the same-pod SOTA control. A candidate becomes 8x-worthy only if it clearly
  beats that relative control curve. The `~1.10` raw threshold applies only to
  the exploratory 4k line as a "worth discussing" signal, not to the main SOTA
  Neural objective.
- Quantization/compression is a known later bottleneck, but do not let it
  distract from the current body-donor breakthrough gate.
- `B34` is not the north star. It only matters if it teaches something that
  helps build the body donor; otherwise do not orbit it.
- Control discipline: the only control worth surfacing is PR1493 or our current
  best anchor run on the same hardware/time/data/evaluator as the test. Internal
  ablation controls are lab bookkeeping only. Do not promote a result unless it
  clears that same-pod SOTA control curve.
- Do not narrate failed probes or non-promotable test results. Archive them in
  logs if needed; keep user-facing focus on candidates with a credible path past
  PR1493/current-best.
- The next real engineering target after mom95 is decided is a body-donor jump,
  not final packaging. Legal byte allocation comes after a raw candidate is
  worth compressing.
- Important 4k-vocab truth: the user did create and test a 4k tokenizer before
  the broken April 23 reruns. The original asset is in
  `/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_4096_bpe.model`,
  SHA256 `6b0337698df13acdb58e5b05446ef26d9d7e4ca5545dc5e13ad5815a7ec904e0`.
  The matching local dataset also exists under
  `/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp4096/`.
  It is not the same tokenizer as the later pulled-pod copy
  `e5b31cf3d3bd87f68998ded73e8f7f9bdaea998bee4899461ab4f4d47f7b4987`.
- The April 23 `test_4k_vocab*.sh` reruns are not trustworthy quality evidence:
  the autoloop/operator was interpreting high BPB as good, and the reruns either
  OOMed or failed before useful validation. Treat the user's original 4k vocab
  as an unvalidated breakthrough candidate, not as rejected.
- 4k thesis to preserve: 8k vocab may be overfitting/over-spending capacity in a
  small model. The active 4k leg is now a broad custom-kernel architecture
  search, not a tokenizer distraction: pinned original 4k assets, blessed whale
  fusion, mom95 donor setting, and no assumption that D72 is optimal. The sweep
  tests whether 4k plus reallocated body budget wants shallow/wide, lean/deep,
  D64 power-of-two heads, the known D72 shape, or a non-power-two custom-kernel
  target.
- 4k-arm gate is raw neural only for now. Use train-only raw curves as the
  decision signal. Do not spend 4k time on eval/compression/quant/sliding/TTT
  unless a raw 4k curve gets around `1.10` BPB on a comparable
  competition-shaped scale.
- 4k kernel search plan is tracked in
  `blessed_whale/FOUR_K_KERNEL_ARCH_SEARCH.md`; runnable sweep wrapper is
  `blessed_whale/tmp_codex_4k_big_arch_sweep.sh`.
- 4k Midnight/Rat Rod follow-up is prepared in
  `blessed_whale/tmp_codex_4k_midnight_wide_sweep.sh`. The old Rat Rod Green v1
  signal was `512d / 11L / 8 heads / 4 KV / MLP 3.0`, all-layer XSA,
  `ROPE_DIMS=16`, BigramHash `2048x128`, EMA/SWA, and fast Muon mechanics.
  Since blessed whale's PR1493-family trainer does not expose BigramHash as an
  env-only knob, the immediate 4k test is the geometry lesson: shallow/wide,
  D64 heads, high MLP, and valid short loop spans. Port BigramHash/local-context
  only if that geometry earns it.
- Midnight-line truth: do not revive midnight as the primary world-record body.
  The beloved `midnight_iii_v_qattn6` headline was `final_sliding_window_exact
  =1.09159026` at `15,576,180` bytes, but it had `gptq:calibrated 0 layers`
  and poor reloaded-quant scores around `1.47`. The corrected export path fixed
  GPTQ/loop restore but landed around `1.105` sliding / `1.122` quant-roundtrip.
  Keep midnight as a lesson source, not the main lane.
- Midnight lessons to port forward: loop-state restore before compression/eval,
  require reloaded-quant artifact scoring, enforce GPTQ calibrated-layer count
  as a hard gate, and try aux/embed/other byte relief before weakening
  attention or MLP.

## Live small-pod takeover - 2026-04-24

- Active Vast pods taken over:
  - 2x H100: `34.150.151.125:9588`, `/workspace/SOTA_FINAL`
  - 1x H100 full-data pod: `219.86.90.204:40060`, `/workspace/SOTA_FINAL`
  - 1x H100 small-data pod: `219.86.90.208:40017`, `/workspace/SOTA_FINAL`
- Current 2x edge candidate:
  `codex_2x_main_raw_b31_mom95_mlp45_20260424_062205`.
  Exact config: B31 blessed-whale donor/kernel, seed `444`,
  train-only, `nproc=2`, `MAX_WALLCLOCK_SECONDS=1800`,
  `MLP_MULT=4.5`, `MUON_MOMENTUM=0.95`, `NUM_LAYERS=12`,
  `MODEL_DIM=576`, `XSA_LAST_N=11`, `PARALLEL_RESIDUAL_START=8`,
  loop onset `0.35`, layers `3..5`, `NUM_LOOPS=2`. Result:
  step `2504`, `val_bpb=1.0866`. Treat as a strong candidate signal,
  but do not call it a win until the same 2x pod PR1493/current-SOTA anchor
  finishes on the same data/evaluator/wallclock.
- Same-pod 2x PR1493 anchor raw result is now in:
  `codex_2x_pr1493_anchor_train_only_20260424_070017` stopped at step
  `3636`, `val_bpb=1.0968` under the same 30-minute proxy. Raw delta:
  B31 mom95 MLP4.5 candidate beats this anchor by about `0.0102` BPB. The
  anchor was still in post-train cleanup when noted; the raw train-stop delta
  is the important proxy signal.
- 2x queue discipline: the PR1493 anchor is running now, and
  `blessed_whale/tmp_codex_2x_stalemate_breaker_queue.sh` is staged on the
  2x pod in tmux session `codex_2x_stalemate_breaker_after_anchor`. It waits
  for the active torchrun to exit before launching 15 sequential train-only
  body hypotheses. Each candidate has a `2700` second hard timeout above the
  `1800` second training cap, and the queue continues even if one candidate
  fails. Do not start another 2x torchrun over it.
- Single-GPU shutdown pull:
  local evidence directory
  `evidence/pod_pulls/2026-04-24_single_gpu_shutdown_pull/`.
  The 4k pod logs/scripts were pulled after the final wide sweep finished.
  The 1x80 pod logs/scripts were pulled after raw stop and quantized eval;
  a local watcher `codex_pull_1x80_final_after_exit` will re-pull once more if
  that final scorer exits before the pod is terminated.
- Correct tokenizer hash to use everywhere:
  `0518df93315eaff6bc75ad4d9711b01c1b5e9ac32e8b3da603c040d8653f7f03`.
- The local workstation tokenizer was stale (`4e979...`) and was replaced from
  the clean 1x pod. Do not resurrect the stale tokenizer.
- Small-pod probes that were intentionally killed after useful signal:
  - SmearGate + attn-out gate: trained to step `412`, train-stop `1.2465`,
    but post-EMA `1.73551145`, quantized `1.75190733`, size `16037747`.
    Current tracked-env form is not promotable.
  - 2x B34 mom95 `MATRIX_CLIP_SIGMAS=10.5`: trained to step `602`,
    train-stop `1.2000`, post-EMA `1.36021732`, quantized `1.37226651`,
    sliding `1.35622167`, size `17465459`. Global clip relief is dead:
    it worsens bytes and BPB.
  - 1x B34 mom95 baseline on 4 shards: trained to step `305`, train-stop
    `1.3094`, post-EMA `2.46014365`, quantized `2.46708187`, size
    `16068035`. Short 4-shard post-EMA/final-eval is not a trustworthy
    final-score proxy.
- Current live pod jobs:
  - 1x full-data B31 mom95 train-only:
    `codex_1xa_full_b31_mom95_20260424_011314`, `80` train shards,
    `MAX_WALLCLOCK_SECONDS=1800`, `SKIP_POST_TRAIN=1`.
  - 1x small-data B31 mom95 sentinel:
    `codex_1xb_4shard_b31_mom95_20260424_011314`, `4` train shards,
    `MAX_WALLCLOCK_SECONDS=1200`, `SKIP_POST_TRAIN=1`.
  - 2x B31 mom95 train-only:
    `codex_2x_20shard_b31_mom95_20260424_011609`, `20` train shards,
    `MAX_WALLCLOCK_SECONDS=1800`, `SKIP_POST_TRAIN=1`.
- Internal ablation bookkeeping exists in pod logs, but do not use it as a
  user-facing decision gate. Only PR1493/current-best anchor comparisons matter.
- Operational conclusion: keep final-eval decisions on full, corrected-scale,
  competition-shaped runs. Use small pods for raw train curves, smoke tests,
  and compression mechanics only.

## Refreshed pod truth - B34 ran on the corrected scale

- Date/time: `2026-04-23` on the 8x H100 Vast pod.
- The pod BPB mismatch was traced to a stale/bad SP8192 tokenizer on the pod.
- Before refresh:
  - tokenizer hash: `4e979c92754cc9d7bdebd9dfe5edf3fc0b41b38714e4297fa851e3afd516`
  - exact upstream PR1493 step-0 BPB: `3.6849`, wrong/high versus the competition scale.
- After rerunning the frozen startup and replacing the suspect tokenizer:
  - tokenizer hash: `0518df93315eaff6bc75ad4d9711b01c1b5e9ac32e8b3da603c040d8653f7f03`
  - validation shard hash stayed `c4fd1d379ff3e99f7cb58398a73aba45da6434fa852b51212c744f3a55cac01b`
  - byte reconstruction: `40540802` tokens, `151230888` bytes, `3.7303378458` bytes/token
  - B34 step-0 check: `val_loss=9.0057`, `val_bpb=3.4864`, back on the expected competition scale.
- We then ran our best immediate promotion lane, `B34`:
  - lane: donor/no-kernel + `MUON_MOMENTUM=0.95`, seed `444`, post-train/final eval enabled
  - log: `/workspace/SOTA_FINAL/blessed_whale/logs/b34_seed444_20260423_225512.log`
  - local evidence: `evidence/pod_pulls/35372056_2026-04-23_b34_refresh_result/`
  - train-stop scout: step `2350`, `val_bpb=1.0938`
  - post-EMA: `val_bpb=1.09713879`
  - quantized: `val_bpb=1.12773448`
  - quantized sliding: `val_bpb=1.11096944`
  - quantized TTT: `val_bpb=1.10557300`
  - total submission size: `16008038` bytes, over the 16MB cap by `8038` bytes.
- Conclusion: refresh fixed the pod/evaluator scale. `B34` is the first corrected-scale final-eval result, and it is not the winner yet.

## RED HOT ALARM - B31 was the signal that led to B34

- Date/time of finding: `2026-04-23` on the 8x H100 Vast pod.
- The important blessed-whale result is `B31`:
  - lane: donor/no-kernel + recovered neural knob `MUON_MOMENTUM=0.95`
  - seed: `444`
  - run log: `/workspace/SOTA_FINAL/blessed_whale/logs/b31_seed444_20260423_214934.log`
  - train-stop scout result: step `3358`, `val_loss=2.7833`, `val_bpb=1.1384`
- Yes, this beats `1.14` on the current raw scout metric. Do not lose this.
- This is not yet a final submission score: it used `SKIP_POST_TRAIN=1`, `GPTQ_RESERVE_SECONDS=0`, and did not run quantized sliding/TTT/final artifact scoring.
- Important correction: this was captured before the SP8192 tokenizer refresh, so do not compare `1.1384` directly against refreshed B34. Its value is as a same-batch mom95 signal over B26, not as a final legal score.

## Pod sanity alarm - resolved by tokenizer refresh

- We fetched upstream `openai/parameter-golf` PR `#1493` into `/tmp/parameter-golf-pr1493`.
- Full audit note: `blessed_whale/POD_AUDIT_20260423.md`.
- Exact upstream record trainer:
  - `/tmp/parameter-golf-pr1493/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py`
  - remote copy: `/workspace/SOTA_FINAL/pr1493_upstream_record.py`
  - SHA256: `477f432097ce619d4080af0a249d29e6e0bf31b7a747ec4e6b509b5be2e202ee`
  - local `pr1493_raw.py` has the same hash.
- Upstream PR1493 sanity run on this pod:
  - log: `/workspace/SOTA_FINAL/logs/pr1493_upstream_sanity_seed42_20260423.log`
  - command used exact upstream trainer, seed `42`, `QK_GAIN_INIT=5.25`, `TTT_ENABLED=1`, `TTT_LR=0.005`, `TTT_EPOCHS=3`, 8 GPUs.
  - training loss curve and timing are sane: stop step `4617`, `val_loss=2.8112`, close to published seed-42 `val_loss=2.8119`.
  - but BPB is wrong/high: pod reports raw stop `val_bpb=1.1499`, pre-quant post-EMA `1.1486`, quantized sliding `1.1429`, quantized TTT `1.14153786`.
  - published PR1493 seed-42 for nearly same loss reports raw stop about `1.0886`, pre-quant post-EMA about `1.0873`, quantized sliding `1.0829`, TTT `1.0808`.
- Corrected interpretation: older local whale/PR1493-faithful evidence with `80` train shards and `val_tokens=40540160` reported step-0 `val_bpb~=3.4870`, matching PR1493's submitted scale. The exact upstream merged PR1493 file on the stale pod reported step-0 `3.6849`. This was SP8192 tokenizer byte-count drift, not ordinary training variance.
- After rerunning the frozen pod startup and replacing the stale tokenizer, B34 step 0 returned to `val_bpb=3.4864`; the pod is now scoring on the expected scale again.
- Additional audit finding: the pod has `80` SP8192 train shards, while PR1493 submitted logs used `128`; the PR1493 submitted seed-42 log also includes `ttt_hash_buckets/ttt_hash_embed`, which are not present in the exact merged record trainer.
- The old B31 scale translation is now only historical context. Use refreshed competition-shaped runs as source of truth.
- Use the competition evaluator as source of truth for final runs. Judge B31 first by same-pod, same-evaluator deltas, then by final competition-shaped artifact scoring.

## Active truth for blessed whale work

- Current public world-record target to beat: about `1.08` BPB, not the older README leaderboard value previously cited.
- Our whale donor lineage has evidence of a raw/non-kernel model build around `1.079` BPB.
- The blocker is not raw donor quality; the blocker is that quantization and compression currently degrade the final submission-facing score back to about `1.09`.
- Blessed whale is the donor/kernel fusion foundation intended to carry the world-leading lane, not a side experiment.
- The active sequence is: tame blessed whale, optimize blessed whale, then add compression/final evaluation.
- We are now in kernel/donor infusion: establish the correct baselines, make the fusion stable and fast, then decide whether the kernel path can preserve or improve the donor while staying legal under the 10-minute/16MB competition rules.

## Latest blessed whale foundation scout

- Pod: Vast 8x H100, run dir `/workspace/SOTA_FINAL/blessed_whale/logs/foundation_foundation_8x_truth_20260423_160030/`.
- `B26` donor/no-kernel control: step `3339`, `val_bpb=1.1406`, `tok/s=4383233`.
- `B27` fusion (`hybrid_backend=fa3_bwd`): step `3111`, `val_bpb=1.1438`, `tok/s=4079555`.
- `B28` x180 fusion (`hybrid_backend=fa3_bwd`): step `2838`, `val_bpb=1.1496`, `tok/s=3726476`.
- `B31` donor mom95: step `3358`, `val_bpb=1.1384`, `tok/s=4412888`.
- `B32` was interrupted at step `400` and is invalid.
- Judgment: in this 8x scout batch, plain fusion and x180 fusion did not beat donor control. `B31` did beat donor control and crossed below `1.14`; it is the mom95 signal that led to B34. The pod BPB accounting issue is now resolved by tokenizer refresh, and B34 is the corrected-scale final-eval result to beat.

## Recovered neural ablation signal

- The old "neural" tests the team remembered are recovered in `blessed_whale/RECOVERED_NEURAL_ABLATIONS.md`.
- The exact remembered result is from the April 23 MLINT focused sweep: control `MUON_MOMENTUM=0.99` hit step `843`, `val_bpb=3.1532`; `mom95` hit step `793`, `val_bpb=3.0543`.
- Treat `MUON_MOMENTUM=0.95` as the first neural knob to promote into the current foundation sequence.
- Runnable matrix lanes are now `B31` donor mom95 scout, `B32` fusion mom95 scout, `B33` fusion mom95 x180 scout, plus final-eval lanes `B34/B35`.
- Caveat: those runs were 2x proxy, `SKIP_POST_TRAIN=1`, train-stop only, and used the older ad hoc `BLESSED_WHALE_HYBRID_BACKEND=fa3` convention. They are a signal, not legal proof.

## Last promoted local reference

- Promoted legal local reference lane: `legs/2026-04-13_midnight_iii_v_qattn6`
- Promoted metric (seed 444): `final_sliding_window_exact val_bpb: 1.09159026`
- Promoted size: `15576180` bytes mixed+brotli
- Corpus: `legs/2026-04-13_midnight_iii_v_qattn6/logs/full_seed444_20260413_044438.log` lines `111` and `107`
- Rule remains: no ad-hoc env experiments from the shell

Update this file when the active target changes.

## 2x overnight queue update - 2026-04-24

- Same-pod anchor remains PR1493 raw stop `1.0968` from
  `codex_2x_pr1493_anchor_train_only_20260424_070017`.
- Previous 2x edge candidate was
  `codex_2x_main_raw_b31_mom95_mlp45_20260424_062205`, raw stop `1.0866`.
- New queue leader so far: `h08_mlp50` from
  `codex_2x_stalemate_breaker_20260424_075754`, raw stop `1.0840` at step
  `2413`. This is about `0.0026` BPB better than the prior 2x edge candidate
  and about `0.0128` BPB better than the same-pod PR1493 anchor.
- Interpretation: MLP pressure is the first clear overnight improvement signal.
  Do not promote final claims until the queue finishes and logs are pulled, but
  this is the current best body-donor direction.

## 2x overnight queue leader update - h10 edged h08

- Queue run: `codex_2x_stalemate_breaker_20260424_075754`.
- Current queue leader after `h10`: `h10_l11_mlp50`, raw stop `1.0839` at
  step `2585`.
- Previous queue leader: `h08_mlp50`, raw stop `1.0840` at step `2413`.
- Same-pod PR1493 anchor: raw stop `1.0968`.
- Interpretation: high MLP pressure is confirmed by two adjacent winners
  (`MLP_MULT=5.0` at 12L and 11L). The `h10` delta over `h08` is tiny, so do
  not overfit the exact layer count yet. The next narrow lane should test MLP
  pressure around `5.0` with 11L/12L shapes and final artifact risk in mind.

## 4k adaptive bracket signal - 2026-04-24

- Run: `codex_1x_4k_adaptive_bracket_20260424_161011`.
- Bracket tested 512d/11L at low/mid/high MLP pressure.
- Results so far:
  - `low_mlp35_l11`: best `1.4356`
  - `mid_mlp425_l11`: best `1.4430`
  - `high_mlp50_l11`: best `1.4663`
- Decision: bracket moved toward low MLP. This is evidence that 4k does not want
  the same high-MLP pressure as the 8k whale lane; it is leaning back toward
  Rat Rod-style body allocation. Continue low-side refinements before judging
  4k as dead.
