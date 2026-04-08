# Crawler System Architecture Matrix

Updated: 2026-04-08  
Branch baseline: `housekeeping/testlab-sync-20260408`  
Leader baseline: `records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/` (`1.13867894`, `15,239,617` bytes, seed 444)

## Snapshot: Where We Are

- In-tree promotion baseline remains BWX 9F (`crawler/LEADER.md`).
- Recent hard fail: `crawler/2026-04-06_Helix_ab_3/RESULTS.md` (`+0.13954768` int6_sw_bpb vs control).
- Ouroboros stacked full run regressed: `crawler/2026-04-06_Ouroboros_III/RESULTS.md` (`+0.00593790` vs BWX 9F).
- BW22/Katta decision quality is currently limited by unfilled result tables:
  - `crawler/2026-04-07_BW22_LoopDepth_9F/ablation.md`
  - `legs/2026-04-07_Crawler_Katta/ablation.md`

## Matrix: Where We Are vs Where We Can Go

| System Axis | Current In-Tree State | Gap / Constraint | Corpus-Mined Opportunities | Next Gate Candidate (One Variable) |
|---|---|---|---|---|
| Recurrence cadence and solver | Fixed loop depth dominates (`CRAWLER_LOOPS=3` baseline). BW22 loop-depth and Katta RK solver sweeps are staged but incomplete. | Compute is spent uniformly; no difficulty-aware depth policy. | Dynamic compute allocation for recurrent transformers (`2602.08864`), depth-recurrent stability objectives (`2603.21676`), adaptive level signals (`2502.13181`). | `DYNAMIC_LOOP_HALTING=1` (eval-only halt policy with fixed max loops). |
| Inter-loop state transfer | Leader is tap-off + anchor-off (`CRAWLER_TAP_DIM=0`, `ANCHOR_DIM=0`). Prior tap/anchor signals did not survive stacked promotions. | Loop differentiation mostly via static rope scales; no persistent compact memory carrier. | Persistent memory bridge (`2505.00929`), adaptive per-step conditioning (`2502.13181`). | `LEVEL_SIGNAL_DIM=8` (shared low-rank loop-level conditioning). |
| Attention/routing | XSA-heavy flat backbone works; Helix cross-stream redesign failed at gate scale. | No uncertainty-aware routing of extra recurrence compute. | Difficulty-aware routing and controller policies (`spot_analyses/technique_extraction/adaptive_compute_allocation*.md`). | `LOOP_ROUTER_MODE=uncertainty_gate_v1` (route extra loop only on high-uncertainty tokens). |
| Quantization path (train + export) | Baseline still favors naive int6 path (`SKIP_GPTQ=1`). Noisy QAT / crawler_int8 show isolated wins but stacked run failed. | Current QAT/PTQ policy is not sensitivity-driven and may overfit isolated arms. | Continuous QAT surrogates (`2410.10849`), StableQAT (`2601.19320`), MatGPTQ (`2602.03537`), LeanQuant (`2407.10032`), layer-wise impact PTQ (`2511.17801`). | `QAT_SURROGATE=softclamp_sigmoidste` (replace STE surrogate only). |
| Bit-allocation and artifact budget | Size stays legal but some arms approach cap; mixed-bit stacks lacked reliable promotion behavior. | No canonical per-block sensitivity-based bit map under 16MB cap. | Training-free MPQ policy search (`2512.07419`), Hessian-style mixed precision (`2509.02512`), single-pass PTQ (`2511.22316`). | `QUANT_POLICY=sensitivity_map_v1` (export-only bit-allocation policy). |
| Runtime throughput and wallclock efficiency | `COMPILE_FULLGRAPH=1` is stable baseline; large architecture swings often regress step time. | Quality gains that increase step_ms can lose total 10-minute value. | Adaptive compute allocation and verifier-triggered extra compute (`test_time_compute_scaling.md`). | `ADAPTIVE_EXTRA_PASS=margin_triggered` (extra pass only when confidence margin is low). |
| Test-time adaptation | SLOT/TTT concepts exist in pipeline notes, but no trusted in-tree promotion line yet. | Eval-time adaptation risks wallclock blow-up and unstable gains. | Adaptive TTT gating (`2601.00894`), taxonomy and guardrails (`test_time_adaptation*.md`, `continual_online_tta*.md`). | `TTT_TRIGGER=high_reconstruction_error` (strict budgeted trigger). |
| Experiment governance and evidence quality | One-variable and gate discipline exists in protocol, but several recent ablation docs are incomplete. | Missing tables degrade ranking confidence and can mis-prioritize expensive runs. | Decision-rule checklists and methodology controls (`analysis_outputs/lm_council_methodology/decision_rules_checklist.md`). | `ABLATION_COMPLETENESS_GUARD=1` (process gate: block promotion if summary fields are missing). |

## Immediate Integration Notes

- Use adaptive-compute ideas first in **eval-only** mode to keep training stack stable while measuring wallclock tradeoffs.
- Prioritize quantization-path improvements that are **surgical** (single surrogate or policy change) before adding new architecture branches.
- Do not run any 8x full production candidate until BW22/Katta tables are backfilled and comparable against BWX 9F controls.

## Corpus Evidence Used (Primary)

- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.08864_understanding-dynamic-compute-allocation-in-recurrent-transformers_20260403_111041.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2603.21676_thinking-deeper-not-longer-depth-recurrent-transformers-for-compositional-generalization_20260404_104013.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2025/2502.13181_ringformer-rethinking-recurrent-transformer-with-adaptive-level-signals_20260210_143925.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2025/2505.00929_compact-recurrent-transformer-with-persistent-memory_20260210_143543.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2024/2410.10849_continuous-approximations-for-improving-quantization-aware-training-of-llms_20260212_082344.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.03537_matgptq-accurate-and-efficient-post-training-matryoshka-quantization_20260402_005940.md`
- `/home/frosty40/ml-research-analysis/archive/_archived_research_analysis_v1/2601.19320_stableqat-stable-quantization-aware-training-at-ultra-low-bitwidths_20260203_202937.md`
- `/home/frosty40/ml-research-analysis/archive/_archived_research_analysis_v1/2511.17801_layer-wise-high-impact-parameter-ratio-optimization-in-post-training-quantization-for-large-language-models_20260202_062929.md`
- `/home/frosty40/ml-research-analysis/archive/_archived_research_analysis_v1/2512.07419_revolutionizing-mixed-precision-quantization-towards-training-free-automatic-proxy-discovery-via-large-language-models_20260203_183054.md`
- `/home/frosty40/ml-research-analysis/archive/_archived_research_analysis_v1/2407.10032_leanquant-accurate-and-scalable-large-language-model-quantization-with-loss-error-aware-grid_20260201_150754.md`
- `/home/frosty40/ml-research-analysis/spot_analyses/technique_extraction/adaptive_compute_allocation.md`
- `/home/frosty40/ml-research-analysis/spot_analyses/technique_extraction/test_time_compute_scaling.md`
- `/home/frosty40/ml-research-analysis/spot_analyses/technique_extraction/test_time_adaptation.md`
- `/home/frosty40/ml-research-analysis/spot_analyses/technique_extraction/continual_online_tta.md`
