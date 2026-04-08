# Parameter Golf PR Mergeability Template

Use this template for record and non-record submissions to maximize reviewer trust and merge speed.

## 1) Scope

- **Track:** `track_10min_16mb` or `track_non_record_16mb`
- **Type:** Record / Non-record
- **Single primary claim:** one core result per PR
- **Supersedes:** `#<old_pr>` (if applicable)

## 2) Results (3 seeds)

| Seed | Primary metric (BPB) | Steps | Train time (s) | Eval time (s) | Artifact bytes |
|---|---:|---:|---:|---:|---:|
| 444 |  |  |  |  |  |
| 300 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| **mean** |  |  |  |  |  |
| **std** |  |  |  |  |  |

## 3) Compliance Statement

- Training <= 600s on 8xH100 SXM: **Yes/No**
- Eval <= 600s on 8xH100 SXM: **Yes/No**
- Total artifact <= 16,000,000 bytes (if record track): **Yes/No**
- No validation leakage during training: **Yes/No**
- No pre-eval adaptation on unseen validation tokens: **Yes/No**
- Score-first / backward-looking eval logic (if applicable): **Yes/No**

## 4) Exact Reproduction

```bash
# exact command used for seed 444
```

- Dependencies:
- Hardware:
- Expected final log lines (paste exact metric lines):

## 5) Diff Hygiene

- [ ] PR contains only one record directory (plus minimal required metadata)
- [ ] No unrelated file edits
- [ ] No rolling branch noise from prior experiments
- [ ] `submission.json` matches README + logs

## 6) Reviewer Notes (Short)

- What is new in one sentence:
- Why it should merge now in one sentence:
- Known limitations:

