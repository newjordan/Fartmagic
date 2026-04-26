# Blessed Whale Crack-It Attack Plan - 2026-04-25

Purpose: turn the four-step objective into a bounded ablation campaign that can
keep the pod useful without drifting, broad-sweeping, or trusting handoff
summaries.

Primary goal: produce a legal SOTA Neural submission candidate that beats
PR1493/current SOTA under comparable evaluator, tokenizer/data scale, wallclock,
artifact bytes, quantization, sliding eval, and TTT.

## Operating Rules

- Raw logs beat handoff notes.
- `iterations: 20000` is only the configured horizon. Completed steps and
  wallclock come from `stopping_early` or final train lines.
- No PR1493/control rerun unless Jordan explicitly authorizes it.
- No pod stop/destroy/cancel unless Jordan explicitly authorizes it.
- No direct SSH unless Jordan explicitly asks for it.
- No full-run or broad queue before the previous gate has a raw-log-backed pass.
- A run that fails its gate is archived and excluded from user-facing progress.
- Any paid launch must print: instance id, repo path, branch, trainer checksum,
  tokenizer hash, `world_size`, `max_wallclock_seconds`, run id, and exact gate.

## Current Evidence Anchors

Trusted:

- PR1493 same-pod 2x anchor raw log:
  `evidence/pod_pulls/2026-04-24_2x_overnight_watch/codex_2x_pr1493_anchor_train_only_20260424_070017/run.log`
  - raw stop `1.0968`
  - post-EMA `1.09594768`
  - quantized `1.10645179`
  - sliding `1.08980182`
  - TTT `1.08841468`
  - total size `15,991,013`

- 2x raw body queue:
  `evidence/pod_pulls/2026-04-24_2x_stalemate_final/codex_2x_stalemate_breaker_20260424_075754/summary.tsv`
  - `h10_l11_mlp50`: raw `1.0839`
  - `h08_mlp50`: raw `1.0840`

Untrusted until raw log is recovered:

- `h10_mlp45` final artifact baseline from summary notes:
  raw `1.0880`, TTT `1.09999677`, size `15,997,247`.

Diagnostic only:

- `mlp_proj_int6_calib128`: 2x, wallclock-capped at `2681/20000`,
  pre-quant EMA `1.08989891`, quantized TTT `1.09374447`, size `18,050,347`.
  This is a clue about MLP projection quant pressure, not a final score.

## Phase 1 - Reconfirm Blessed Whale Potential Versus SOTA

Question: does blessed whale carry a real neural advantage over PR1493/current
SOTA on the same hardware/evaluator, before artifact failure?

Use existing logs first. Launch only if the required evidence is missing.

### Phase 1A: Evidence Audit, No Spend

Build a table from pulled logs:

| Lane | Hardware | Steps | Wallclock | Raw | Post-EMA | Quant | Sliding | TTT | Size | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PR1493 2x anchor | 2x | known | 1800s | `1.0968` | `1.09594768` | `1.10645179` | `1.08980182` | `1.08841468` | `15,991,013` | control |
| h08 2x raw | 2x | `2413` | 1800s | `1.0840` | NA | NA | NA | NA | params `56,513,760` | raw candidate |
| h10 2x raw | 2x | `2585` | 1800s | `1.0839` | NA | NA | NA | NA | params `52,197,208` | raw candidate |
| mlp-proj-int6 diag | 2x | `2681` | 1800s | `1.0880` | `1.08989891` | `1.11442694` | `1.09794194` | `1.09374447` | `18,050,347` | illegal diagnostic |

Pass gate:

- h08/h10 raw edge over PR1493 is confirmed from raw logs.
- No claim is made about h08/h10 final artifacts unless the raw final logs exist.

Fail gate:

- If the raw edge cannot be traced to logs with `world_size`, wallclock, and
  tokenizer scale, blessed whale is not cleared for more pod spend.

### Phase 1B: Minimal Reconfirmation Run, Only If Needed

Run one candidate and one recorded-control comparison only if existing evidence
is insufficient. Since PR1493 control reruns need approval, default is to use the
recorded PR1493 anchor.

Candidates:

| ID | Purpose | Config |
| --- | --- | --- |
| `bw_reconfirm_h10_raw` | confirm best 2x raw shape | h10, 11L, MLP 5.0, 1800s, train-only |
| `bw_reconfirm_h08_raw` | confirm tied 12L raw shape | h08, 12L, MLP 5.0, 1800s, train-only |

Pass gate:

- Candidate raw is at least `0.006` BPB better than recorded PR1493 raw under
  the same shape, or both h08/h10 are within noise and still clearly better than
  control.

Kill gate:

- Step-0 BPB/tokenizer scale mismatch.
- `world_size` not as declared.
- No parseable raw val before wallclock.
- Candidate does not beat recorded control curve by a meaningful margin.

## Phase 2 - Pick The Best Potential Base Neural Model

Question: which base body is worth spending artifact repair on?

No compression work until a base is chosen.

### Base Selection Ablations

Use short, comparable train-only runs or existing logs:

| ID | Body | What It Tests | Gate |
| --- | --- | --- | --- |
| `base_h10_l11_mlp50` | h10 | best raw seen, fewer params than h08 | raw curve, params, throughput |
| `base_h08_l12_mlp50` | h08 | tied raw, larger body | raw curve vs size risk |
| `base_h06_l12_mlp45_xsa12` | h06 | lower MLP pressure, lower artifact risk | raw loss for size savings |
| `base_h03_one_loop` | h03 | lower parameter pressure baseline | whether loop simplification preserves enough raw |

Pick rule:

1. Prefer the best raw candidate that is not obviously impossible to legalize.
2. If raw tie is within `0.0015` BPB, choose the smaller/faster/lower-byte-risk
   model.
3. If h10/h08 raw advantage disappears on reconfirmation, stop and do not enter
   quant repair.

Output artifact:

- `BASE_MODEL_DECISION_20260425.md`
  - exact chosen body
  - raw log path and line refs
  - params
  - hardware shape
  - why rejected bodies were rejected

## Phase 3 - Stabilize Quantization And Compression

Question: why does the chosen body lose after EMA/GPTQ/compression, and what is
the cheapest legal fix?

This phase should start from a checkpoint/artifact where possible. If retraining
is required, keep it short and controlled.

### Phase 3A: Post-Training Quant Harness

Build or use a harness that loads the same post-EMA checkpoint and emits one
parseable table:

`candidate, bits_by_family, layers, post_ema, quant, quant_delta, model_bytes, total_bytes, decision`

Required ablations:

| ID | Ablation | Purpose | Pass Gate |
| --- | --- | --- | --- |
| `q_global_int5` | normal int5/global clip baseline | real baseline from same checkpoint | establishes quant delta |
| `q_mlp_proj_all_int6` | all MLP proj int6 | upper-bound quality repair | confirms MLP proj pressure |
| `q_mlp_proj_late_int6` | late half MLP proj int6 | legal approximation | quant improves with lower byte cost |
| `q_mlp_proj_loop_int6` | loop layers only int6 | target recurrent-sensitive layers | most gain per byte |
| `q_mlp_proj_binary_subset` | binary search MLP proj layer subset | smallest useful int6 set | monotonic quality/byte table |
| `q_attn_proj_int6` | attn proj int6 | rule out attention projection pressure | must beat int5 baseline |
| `q_qkv_int6` | QKV int6 | rule out QKV pressure | must beat int5 baseline |
| `q_mlp_fc_int6` | MLP fc int6 | rule out FC pressure | must beat int5 baseline |

Pass gate:

- Quant improves over same-checkpoint int5 baseline.
- Total bytes are legal or have a measured path to legal.

Kill gate:

- Any candidate over the exact legal cap (`16,000,000` bytes) with no
  subset/byte path is diagnostic only.
- No sliding/TTT until quant and size gates pass.

### Phase 3B: Equalization / Constraint Ablations

Only after Phase 3A identifies the responsible tensor family.

| ID | Ablation | Purpose | Pass Gate |
| --- | --- | --- | --- |
| `eq_mlp_proj_row_rms` | row RMS cap on MLP proj before GPTQ | reduce outliers | quant delta improves |
| `eq_mlp_proj_awq` | SmoothQuant/AWQ-style scale migration | make int5 mimic int6 | quant improves without byte cost |
| `eq_mlp_proj_layerwise_clip` | per-layer clip sigmas | avoid global blunt clip | improves quant/bytes |
| `repair_fake_quant_short` | short fake-quant repair | train into int5-stable basin | quant improves without raw collapse |
| `ema_swa_quant_pick` | choose checkpoint by quant eval | avoid raw-only EMA choice | quant/sliding improves |

Pass gate:

- Quant is better than same-checkpoint int5 baseline and size is legal.
- Prefer candidates that reduce post-EMA to quant delta, not candidates that
  only shrink bytes.

## Phase 4 - Stabilize Final TTT / Eval

Question: after quant is legal, can sliding/TTT recover like PR1493 rather than
degrade like h10 summaries suggest?

Only run this phase for candidates that pass Phase 3.

### TTT / Eval Ablations

| ID | Ablation | Purpose | Gate |
| --- | --- | --- | --- |
| `ttt_baseline` | existing TTT settings | final reference | compare PR1493 TTT `1.08841468` |
| `ttt_lr_low` | lower TTT LR | avoid over-adapting brittle quant | TTT improves |
| `ttt_lr_high` | higher TTT LR | test under-adaptation | TTT improves |
| `ttt_epochs_2` | fewer TTT epochs | lower damage/time | TTT improves or same |
| `ttt_epochs_4` | more TTT epochs | recover more | TTT improves enough to justify time |
| `sliding_stride_check` | exact promotion eval settings | prevent eval mismatch | line refs match expected evaluator |

Pass gate:

- Legal total bytes.
- Quant does not fail.
- Sliding and TTT move toward or beat PR1493 final TTT.

Final promotion gate:

- raw-log-backed artifact metrics beat PR1493/current SOTA on the final
  submission-facing metric under comparable conditions.

## Pod Custody Loop

Keep the pod useful without letting it drift:

1. Every live action starts with `vastai show instances`.
2. If GPU util is `0%`, inspect logs/processes before launching anything.
3. If a run is active and matches an approved gate, monitor only.
4. If a run is active but mismatched, report exact mismatch and ask before kill.
5. If no approved next ablation exists, do not improvise. Prepare files and ask.
6. After every run, pull logs before starting the next run.
7. Append a one-row result to the phase table before any follow-on launch.

Minimum monitor output:

- instance id and status
- GPU util/memory
- active process command
- latest log path
- latest step and wallclock
- current gate pass/fail status

## Immediate Next Actions

1. Build `BASE_MODEL_DECISION_20260425.md` from existing raw logs.
2. Recover or disprove the missing `h10_mlp45_final.log`.
3. If h10/h08 raw potential is still supported, implement the post-training
   quant harness before launching more paid training.
4. Run only Phase 3A ablations first: tensor-family/layer-subset quant tests.
5. Do not spend on TTT until quant and size pass.
