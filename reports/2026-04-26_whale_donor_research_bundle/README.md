# Whale Donor Research Bundle - 2026-04-26

This bundle is for agents doing deep research on the whale / whale donor /
12L SOTA-breaking line in `SOTA_FINAL`.

## Current Lock

Active donor record:

```text
2026-04-15_whale_l12_dim576_matrixbits5
```

Treat this as the primary record to protect and rebuild around. The preserved
record is a config + log + metrics lock, not a preserved model-binary lock.

## Mission

Find or rebuild a legal SOTA Neural candidate from the whale donor line:

- preserve the 12L / dim576 / matrixbits5 donor evidence;
- establish clean source provenance before launching new work;
- compare candidates only against same-hardware, same-data, same-evaluator
  PR1493/current-best controls;
- solve final artifact legality under the `16,000,000` byte cap only after
  the neural body is worth packaging.

## What Is Included

- `manifest.md`: inventory, hashes, and exclusions.
- `key_metrics.md`: donor metrics and comparison anchors.
- `source_index.md`: where each source came from and how to use it.
- `theory_index.md`: working theories, gates, and research questions.
- `source_copies/`: small text-only source copies and logs.

## What Is Excluded

- No binary model checkpoints.
- No tokenizer or dataset binaries.
- No `vault/` content.
- No pod launch wrappers or run-control shell scripts.
- No deleted or unsafe legacy source directories.

## First Read Order

1. `key_metrics.md`
2. `source_index.md`
3. `theory_index.md`
4. `source_copies/donor_record/RESULTS.md`
5. `source_copies/evidence_logs/whale_l12_dim576_matrixbits5_s444.txt`

## Branch Suggestion

Suggested branch name for publication:

```text
research/2026-04-26-whale-donor-bundle
```

