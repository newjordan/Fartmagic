# Source Index

## Donor Record

| Bundle path | Original path | Purpose |
| --- | --- | --- |
| `source_copies/donor_record/RESULTS.md` | `legs/2026-04-15_whale_l12_dim576_matrixbits5/RESULTS.md` | canonical donor metrics |
| `source_copies/donor_record/tracked_env.sh` | `legs/2026-04-15_whale_l12_dim576_matrixbits5/tracked_env.sh` | donor config lock |

Note: the historical leg `hypothesis.md` was not copied because the useful
source of truth is the result/config/log lock. Rebuild from clean reviewed
source before launching new tests.

## Evidence Logs

| Bundle path | Original path | Purpose |
| --- | --- | --- |
| `source_copies/evidence_logs/whale_l12_dim576_matrixbits5_s444.txt` | `evidence/pod_pulls/instance_35002131_20260416_1113/logs/whale_l12_dim576_matrixbits5_s444.txt` | full donor training/eval metrics |
| `source_copies/evidence_logs/full_seed444_20260416_030758.log` | `evidence/pod_pulls/instance_35002131_20260416_1113/legs/2026-04-15_whale_l12_dim576_matrixbits5/logs/full_seed444_20260416_030758.log` | run wrapper log with environment evidence |

## Current Direction Docs

| Bundle path | Original path | Purpose |
| --- | --- | --- |
| `source_copies/current_docs/CURRENT.md` | `CURRENT.md` | latest local active-state notes |
| `source_copies/current_docs/NEXT.md` | `NEXT.md` | latest local next-step notes |
| `source_copies/current_docs/WHALE_1797_SMOOTHING_PIVOT_20260426.md` | `WHALE_1797_SMOOTHING_PIVOT_20260426.md` | CaseOps blocked, portable-only PR1797 ideas |
| `source_copies/current_docs/NEURAL_MODEL_PIVOT_20260426.md` | `NEURAL_MODEL_PIVOT_20260426.md` | body-first pivot notes |
| `source_copies/current_docs/BASE_MODEL_DECISION_20260425.md` | `BASE_MODEL_DECISION_20260425.md` | h10 provisional base decision |
| `source_copies/current_docs/BLESSED_WHALE_CRACK_IT_ATTACK_PLAN_20260425.md` | `BLESSED_WHALE_CRACK_IT_ATTACK_PLAN_20260425.md` | bounded ablation plan |
| `source_copies/current_docs/FINISHLINE_RESULTS_20260425.tsv` | `FINISHLINE_RESULTS_20260425.tsv` | quant/finishline result table |
| `source_copies/current_docs/QUANT_CHECKPOINT_INVENTORY_20260425.md` | `QUANT_CHECKPOINT_INVENTORY_20260425.md` | checkpoint provenance status |
| `source_copies/current_docs/TRAINER_SELECTIVE_BITS_AUDIT_20260425.md` | `TRAINER_SELECTIVE_BITS_AUDIT_20260425.md` | local vs pod selective-quant trainer audit |

## Research/Theory Docs

| Bundle path | Original path | Purpose |
| --- | --- | --- |
| `source_copies/research_docs/whale_oob_experiment_shotgun.md` | `docs/whale_oob_experiment_shotgun.md` | big-swing concept bank |
| `source_copies/research_docs/agent_handoff_help_needed.md` | `docs/agent_handoff_help_needed.md` | agent operating style and focus |
| `source_copies/research_docs/whale_t2048_research_focus_20260418.md` | `docs/whale_t2048_research_focus_20260418.md` | kernel/fusion focus notes |
| `source_copies/research_docs/whale_t2048_breakthrough_20260418.md` | `docs/whale_t2048_breakthrough_20260418.md` | T=2048 breakthrough notes |
| `source_copies/research_docs/whale_t2048_next_gates_20260419.md` | `docs/whale_t2048_next_gates_20260419.md` | T=2048 follow-up gates |
| `source_copies/research_docs/GPTQ_MEGAKERNEL_BUG.md` | `docs/GPTQ_MEGAKERNEL_BUG.md` | quant/kernel bug notes |

## Excluded From Bundle

- `vault/`
- `data/`
- binary model files: `*.pt`, `*.ptz`, `*.bin`, `*.safetensors`
- pod launch wrappers and queue scripts
- large tarballs and pod snapshots
- deleted/unsafe legacy source directories

