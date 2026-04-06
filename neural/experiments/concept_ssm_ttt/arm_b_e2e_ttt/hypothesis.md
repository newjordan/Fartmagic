# Arm B: E2E Test-Time Training

**Variable:** TTT_E2E=1, TTT_HIDDEN=64
**Base:** Rascal III
**Paper:** TTT-E2E (2512.23675)
**Mechanism:** Add tiny fast MLPs (dim→64→dim) to the last 2 transformer layers. These are part of the model architecture, trained during pre-training. The fast MLPs learn to refine hidden states. At export, they're included in the artifact (small: 2×(512×64 + 64×512) = ~131K params = ~100KB at int6).
**Hypothesis:** The fast MLPs act as a learned refinement layer that improves token prediction. Because they're small and trained E2E, they add minimal size/compute while improving representation quality.
**Note:** This is NOT eval-time optimization. The TTT layers are part of the model from step 0.
**Gate:** 2000 steps, 4×GPU, seed=300. Compare sw BPB and artifact size vs Rascal III control.
