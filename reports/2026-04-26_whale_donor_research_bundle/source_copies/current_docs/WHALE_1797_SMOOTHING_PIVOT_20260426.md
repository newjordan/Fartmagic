# Whale + PR1797 Smoothing Pivot - 2026-04-26

## Active Donor Lock

The active donor is:

```text
2026-04-15_whale_l12_dim576_matrixbits5
```

This is the best-shot record to protect and rebuild around:

- post-EMA raw bpb: `1.07952782`
- quantized bpb: `1.11723031`
- quantized sliding bpb: `1.10039777`
- quantized TTT bpb: `1.09561039`
- artifact bytes: `15,995,671`

Source-of-truth files:

- `legs/2026-04-15_whale_l12_dim576_matrixbits5/RESULTS.md`
- `legs/2026-04-15_whale_l12_dim576_matrixbits5/tracked_env.sh`
- `evidence/pod_pulls/instance_35002131_20260416_1113/logs/whale_l12_dim576_matrixbits5_s444.txt`

`gemini_runs/` has been deleted locally and on the pod. Do not restore it.

## Direction

CaseOps/tokenizer adoption is blocked until explicitly cleared. PR #1797 does
not use the standard PR #1493 tokenizer file: it uses
`fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`, still with
`vocab_size=8192`, but with CaseOps operator symbols reserved at IDs 4..7 and
with validation byte sidecars for BPB accounting.

The active research angle is now:

1. Keep the whale donor body and standard tokenizer/data path.
2. Mine PR #1797 only for portable smoothing or packaging techniques.
3. Do not inherit CaseOps shards, CaseOps tokenizer, or byte-sidecar scoring
   until legality is reviewed and explicitly approved.
4. Re-establish any new test from clean donor source, not from deleted
   `gemini_runs` material.

## Portable Candidates

- LQER asymmetric rank-4 post-GPTQ correction:
  likely the best first transplant. It is a compression/artifact repair layer,
  not a tokenizer change. Test only after a clean whale donor final-eval control
  exists, because it changes quantized artifact bytes and quant BPB.
- TTT warm-start A / phased score-first TTT:
  potentially portable to whale final eval. Treat as eval/packaging, not body
  evidence.
- Fused CE:
  possibly portable as a training-speed path if it compiles cleanly on cu130.
  It should be validated as mechanics, not as a quality claim.
- Smear gate:
  a train-time model-body modification. It may work on the whale donor, but it
  must be tested only after clean donor source is re-established.
- SparseAttnGate / min-LR / PR1787 base optimizer choices:
  potentially useful, but they are coupled to the PR1787 body stack. Do not
  stack them blindly into the first whale transplant.

## Stop Rules

- No CaseOps data prep or launch while tokenizer legality is unresolved.
- No claim that PR #1797 score transfers to whale; the reported lift includes
  CaseOps and PR1787 stack interactions.
- No model-body change before one clean whale donor control for the current
  hardware/data/evaluator is recorded.
- First graft should be isolated and source-of-truthed in a dated `run.py`.
- No `gemini_runs` restore or launch.
