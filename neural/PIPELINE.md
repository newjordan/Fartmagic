# Neural Pipeline — Ranked Hypothesis Queue

**Updated:** 2026-04-06
**Champion:** Rascal II · 1.1099 mean val_bpb · 15.55MB max artifact
**Frozen record:** `/home/frosty40/sota_nueral/records/track_10min_16mb/2026-03-30_Rascal_8xH100`
**Working copy:** `/home/frosty40/sota_nueral/experiments/Rascal_II_homebase`
**Active lab:** `/home/frosty40/sota_nueral/experiments/Rascal_II_mixed_int_lab`
**Launch surface:** `/home/frosty40/sota_nueral/test_lab`
**Current campaign:** stronger base model first, no eval hacks

---

## ACTIVE — In Progress

### Megarascal: Kernel Fusion Speed Ablation
**Status:** Committed, pending pod run. 4×GPU, 2000 steps each.
**Location:** `experiments/megarascal/`
**Arms:**
- Control (baseline compile)
- Arm A: COMPILE_MODE=reduce-overhead (CUDA graphs)
- Arm B: MLP_KERNEL_MODE=triton_act (existing Triton activation)
- Arm C: MLP_KERNEL_MODE=fused_norm_act (fused RMSNorm+MLP kernel)
**Goal:** More steps/sec = lower BPB at same wallclock. Not a per-step improvement.

### Mixed-Int Promotion to 8×GPU
**Status:** Confirmed positive on 4×GPU. Needs 8×GPU full run + seed confirmation.
**Variable:** QUANT_ATTN_BITS=5 (attn=5, mlp=6, embed=8)
**Delta:** -0.0028 sw BPB, -0.54 MB saved.

### Ouroboros Improvement (Crawler Track)
**Status:** 4×GPU sweep complete. All 3 arms positive. Needs seed 444 confirmation.
**Location:** `parameter-golf-lab/crawler/2026-04-06_Ouroboros_ablation/`
**Results:**
- crawler_int8: -0.00315 sw BPB (best quality, 15.5MB size concern)
- noisy_qat: -0.00296 sw BPB (best size/quality tradeoff)
- contractive dt: -0.00227 sw BPB (weakest, fastest step time)
**Next:** Seed 444 confirmation, then decide 4-hour run allocation (Ouroboros vs Helix).

---

## COMPLETED — Rascal II Research Ablation Sweep (2026-04-06)

Full results: `experiments/Rascal_II_mixed_int_lab/SWEEP_RESULTS.md`

| Arm | Variable | Result | Verdict |
|-----|----------|--------|---------|
| Mixed-int | QUANT_ATTN_BITS=5 | -0.0028 sw BPB, -0.54 MB | **POSITIVE** |
| Trigram | TRIGRAM=1 | — | **DEAD** (user pre-tested) |
| mu-centering | Code change | +0.0097 sw BPB | **DEAD** |
| Gated attention | GATED_ATTENTION=1 | -0.0003 sw BPB (noise) | **DEAD** |
| HEQ | Code change | +0.0040 sw BPB | **DEAD** |
| DDL | Code change | +0.0505 sw BPB, 27% slower | **DEAD** |

---

## DEAD (do not retry)

| Hypothesis | Result | Date |
|-----------|--------|------|
| Bigram 3072 | 0.0000 at proxy | 03-31 |
| Bigram 4096 | +0.0006 (hurts) | 03-31 |
| Warmdown 4000 | +0.0034 (hurts) | 03-31 |
| QAT early (0.25) | +0.0004 (hurts) | 03-31 |
| QAT late (0.05) | +0.0004 (hurts) | 03-31 |
| SWA dense (every=10) | +0.0010 (hurts) | 03-31 |
| RoPE 32 | -0.0004 (noise floor) | 03-31 |
| QK_GAIN_INIT=4.0 | DEAD, extensively tested | 03-31+ |
| GPTQ post-train | torch.compile hook bug | 03-31 |
| Trigram on Rascal II | User pre-tested, null | 04-06 |
| mu-centering | +0.0097 (hurts) | 04-06 |
| HEQ entropy scales | +0.0040 (overhead) | 04-06 |
| DDL rank-1 delta | +0.0505, 27% slower | 04-06 |
| Gated attention | -0.0003 (noise) | 04-06 |

---

## Organizer-Requested Concept PRs (2026-04-06)

Each concept has a folder under `experiments/` with research findings and implementation plan.

| Concept | Folder | Lead Paper | Maturity |
|---------|--------|-----------|----------|
| JEPA | `concept_jepa/` | JEPA-Reasoner (2512.19171) | Research only |
| Text diffusion | `concept_text_diffusion/` | GDDS (2603.21342) | Research only |
| H-net tokenization | `concept_hnet_tokenization/` | HAT (2603.15953) | Promising |
| Universal transformer | `concept_universal_transformer/` | **Ouroboros (PR #1308, 1.1364 BPB)** | 50+ ablations |
| Megakernels | `concept_megakernels/` | FlashMHF (2512.06989) | **Active: megarascal** |
| SSM + E2E TTT | `concept_ssm_ttt/` | TTT-E2E (2512.23675) | Research only |
| Random map adapters | `concept_random_map_adapters/` | Persistent Memory (2603.22329) | Novel |

---

## Workspace Rules

- New neural work starts from `experiments/Rascal_II_homebase/`
- Mixed-bit export and capacity work starts from `experiments/Rascal_II_mixed_int_lab/`
- Speed work starts from `experiments/megarascal/`
- Frozen records stay under `records/`
