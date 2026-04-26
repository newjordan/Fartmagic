# Trainer Selective-Bits Audit - 2026-04-25

Purpose: prove which trainer can actually run selective quant/compression tests
before any new spend.

## Result

The pod and local checkout do not currently point at the same trainer.

| Location | Path | Branch | SHA256 | Selective family bits |
| --- | --- | --- | --- | --- |
| Local | `/home/frosty40/SOTA_FINAL/pr1493_decompressed.py` | `wh_kern_int` | `f583502d8dbc23c37c0c81a2d309e14391b74dd128b7a1259fd4512604d2cfb7` | not present in this file |
| Pod | `/workspace/SOTA_FINAL/pr1493_decompressed.py` | `main` | `84534e46598baa08c2f359f54f52078aee584fdda483f9f2c52869544b526f4e` | present |

Pod-side grep confirmed:

- `_selective_bits` at `pr1493_decompressed.py:270`
- `MLP_FC_BITS` / `MLP_FC_CLIP_SIGMAS`
- `MLP_PROJ_BITS` / `MLP_PROJ_CLIP_SIGMAS`
- `ATTN_CQ_BITS`, `ATTN_CK_BITS`, `ATTN_CV_BITS`
- `ATTN_PROJ_BITS`
- selective dispatch at `pr1493_decompressed.py:299`

Local grep confirmed only global `MATRIX_BITS`, `EMBED_BITS`, and
`GPTQ_CALIBRATION_BATCHES` in root `pr1493_decompressed.py`.

## Gate

Any paid launch must print:

- repo path
- branch
- trainer checksum
- whether `_selective_bits` exists in the exact trainer being executed
- env names used for family bits

If the active trainer checksum is not `84534e46598baa08c2f359f54f52078aee584fdda483f9f2c52869544b526f4e`
or an explicitly reviewed successor, do not launch selective quant ablations.

## Implication

The pod has the selective quant path; the local branch does not expose it in the
root trainer. That makes sync/provenance a hard gate. A launch from the wrong
checkout can silently turn a selective quant test into a global int5/int6 test.

