# Crawler Science Board

Track: Bandit_Wagon lineage · Goal: best int6_sw_bpb + smallest artifact ≤ 16MB
Champion: **1.18672385 BPB** (seed 444) · **8.61MB** · `crawler/2026-03-29_BW5/`

Legend: → PROMOTED · ✓ PASS · ✗ FAIL · ⏳ PENDING · — n/a

---

## Thread: Baseline — Crawler loops=3 / mlp=6.0

Established in Leg 3. This is the root config all BW legs descend from.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-29 | Leg 3 (CL3-01) | loops=3, mlp=6.0, SKIP_GPTQ=1, 600s wallclock | — | — | 1.18720 | 8.84MB | → PROMOTED | Extra time > GPTQ tricks. Quant gap nearly closed. 7MB headroom. |
| 2026-03-29 | BW4 | unknown | — | — | 1.18731 | 8.97MB | → PROMOTED | +0.00011 vs Leg 3 seed=444. Reference parent for BW5. |

---

## Thread: Compile / Fullgraph

Hypothesis: `torch.compile(fullgraph=True)` eliminates graph breaks → faster step → more steps in budget → lower BPB.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-29 | **BW5** (CHAMPION) | BW4 + COMPILE_FULLGRAPH=1 | — | ✓ 74.52ms | **1.18672385** | **8.61MB** | → PROMOTED | −0.00058 vs BW4. 0 graph breaks. Roundtrip eval 2.77× faster. Seed=300 ⚠️ +0.00012 vs Leg 3 (mean still better). |

Notes: BW5 seed=300 does NOT individually confirm vs Leg 3. Mean is better (1.18715 vs 1.18743). A future leg should try to close this seed disparity.

---

## Thread: MLP Choke Architecture

Hypothesis: pyramid-shaped MLP bottleneck (CRAWLER_MLP_CHOKE_DIM=512) gives the loop
more representational pressure at each recurrence → better quality even at lower BPB.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-31 | BW5_Pyramid | BW5 + CHOKE_DIM=512, shape=pyramid, groups=8 | ✓ −0.00987 int6_sw_bpb | ⏳ PENDING | — | — | ⏳ PENDING | 1GPU signal one of strongest in series. +1.57M params, +747KB. Speed +3.4ms/microbatch — needs 8GPU confirm. |

Gate 1GPU detail (500 steps, seed=444, grad_accum=8):
- BWVP-00 (flat): step_avg 583.99ms · int6_sw_bpb 1.44668780 · 6,750,039 bytes
- BWVP-01 (pyramid): step_avg 611.21ms · int6_sw_bpb **1.43681894** · 7,497,734 bytes
- Delta: +27.22ms (÷8 ≈ +3.4ms real) · **−0.00987** bpb · +747KB

---

## Thread: Cannon (gate feed-forward type)

Hypothesis: scalar cannon feed-forward gives the crawler loop a faster compiled path.
NOTE: BW5_Cannon must be individually gated for speed BEFORE combining with Pyramid.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| — | BW5_Cannon | BW5 + CRAWLER_CANNON_TYPE=scalar | — | — | — | — | NOT STARTED | Speed gate required first. Must not exceed 74.68ms/step. |

---

## Thread: Combined — Pyramid + Cannon

Run ONLY after BW5_Cannon (speed) AND BW5_Pyramid (quality) both individually confirm on 8GPU.
This is a 2-variable test by design — combines validated Pyramid + validated Cannon.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-31 | BW5_PyramidCannon | BW5 + CHOKE_DIM=512 + CANNON_TYPE=scalar | BLOCKED | BLOCKED | — | — | BLOCKED | Waiting on: Pyramid 8GPU + Cannon 8GPU |

---

## Planned Hypotheses

| Priority | Hypothesis | Thread | Prerequisite | Rationale |
|----------|-----------|--------|-------------|-----------|
| 1 | BW5_Cannon (speed gate) | Cannon | BW5_Pyramid 8GPU result | Need cannon individual speed confirm before combo |
| 2 | BW5_Pyramid (8GPU gate) | MLP Choke | Pod run | 1GPU signal is strong; confirm step_avg on 8GPU |
| 3 | BW5_PyramidCannon | Combined | Pyramid 8GPU ✓ + Cannon 8GPU ✓ | Combine both if individually validated |
| 4 | Seed 300 alignment | Baseline | Any | BW5 seed=300 ⚠️ — does a warmdown tweak help? |

---

## All-Time Reference

| Leg | BPB (seed 444) | Size | Mean BPB | Status |
|-----|----------------|------|----------|--------|
| Leg 3 | 1.18720 | 8.84MB | 1.18743 (3-seed) | Former champion |
| BW4 | 1.18731 | 8.97MB | — | Superseded |
| **BW5** | **1.18672** | **8.61MB** | **1.18715** | **CHAMPION** |
