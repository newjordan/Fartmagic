# Evidence Protocol (Mandatory)

This repository follows an evidence-first operating protocol for analysis and audit tasks.

## Rules

1. No claim about logs without corpus evidence.
   - Cite the extracted corpus path or raw transcript path.
   - Point to exact line references or exact `grep`/`sed` output.

2. No claim about code history without git evidence.
   - Cite commit IDs or diff output.

3. Do not use these words unless the supporting artifact is named:
   - `checked`
   - `verified`
   - `saved`
   - `logged`
   - `validated`
   - `root cause`
   - `ready`

4. No implementation before evidence on evidence-heavy tasks.
   - For log-audit tasks, the first output must be a corpus summary, not a solution scaffold.

5. Separate every statement into one of three classes:
   - `fact`
   - `inference`
   - `proposal`

6. Treat plausible-but-unverified as failure, not partial success.

## Reporting Contract

- Every evidence-backed claim must include a direct artifact reference.
- If evidence is missing, explicitly mark the item as unverified.
