# Quant Checkpoint Inventory - 2026-04-25

Purpose: identify whether a ready h10 checkpoint exists for no-spend
post-training quant ablations.

## Local Inventory

A local artifact scan found checkpoints and final models, but no pulled
`final_model.pt` directly tied to the selected `h10_l11_mlp50` raw candidate or
the missing `h10_mlp45_final.log`.

Relevant local findings:

- `pod_pulls/2x_h10_20260425/final_extract/...` contains h10 selective logs and
  summaries, but the scan did not find a colocated `final_model.pt`.
- `pod_pulls/4x_dealer_34405931/SOTA_FINAL/final_model.int6.ptz` and
  `final_model.mixed.ptz` exist, but are from the later 4x Dealer state, not a
  proven h10 selected-base checkpoint.
- `evidence/pod_pulls/.../artifacts/blessed_whale/final_model_blessed_whale_b34.*`
  exists, but it is b34 evidence, not the selected h10 base.

## Remote Inventory

The live pod `/workspace/SOTA_FINAL` has:

- `./final_model.pt`, modified `2026-04-25 14:17`, about `177M`
- `./final_model.int6.ptz`, modified `2026-04-25 14:18`, about `17M`
- `./final_model.mixed.ptz`, modified `2026-04-17 04:36`, about `11M`
- `./artifacts/blessed_whale/final_model_blessed_whale_b24.pt`, modified
  `2026-04-25 15:52`, about `177M`
- `./artifacts/blessed_whale/final_model_blessed_whale_b24.int6.ptz`, modified
  `2026-04-25 15:52`, about `17M`

These are not accepted as h10 checkpoints until a matching log/config proves
which run produced them.

## Decision

No ready, provenance-clean h10 checkpoint is available locally yet.

The next ready action is not a quant ablation launch. It is quant-harness and
provenance implementation:

- parse a selected checkpoint path;
- print trainer checksum and tokenizer hash;
- emit a table with post-EMA, quant, size, sliding, TTT fields;
- refuse to run if checkpoint provenance does not match the selected base.

