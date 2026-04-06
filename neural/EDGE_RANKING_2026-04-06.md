# Ranked Edge Queue (2026-04-06)

Ranked by expected BPB upside x feasibility on the current Rascal stack.

1. MTP heads on Rascal II mixed-int base (`MTP_NUM_HEADS>0`) - 9.0/10
2. Arm B E2E fast-MLP TTT (`TTT_E2E=1`) - 8.4/10
3. Arm A long-context training (`TRAIN_SEQ_LEN=4096`) - 7.6/10
4. Query-only TTT at eval (qTTT-style) - 7.4/10
5. DAQ objective for quant scale search in export path - 7.0/10
6. TTQ activation-aware test-time quantization - 6.5/10
7. SCORE contractive recurrent depth - 6.1/10
8. Tokenizer-free hierarchy (HAT/HoloByte/ByteFlow) - long-term 9.5/10, near-term 3.0/10

## Immediate 3-run queue

1. Gate `MTP_NUM_HEADS=1` on Rascal II mixed-int base (single variable).
2. Run existing concept Arm B script (`test_lab/concept_arm_b_e2e_ttt.sh`).
3. Run existing concept Arm A script (`test_lab/concept_arm_a_longctx.sh`).

## Notes

- Dead hypotheses remain excluded per `PIPELINE.md`.
- This list is based on local corpus analyses and current repo state as of 2026-04-06.
