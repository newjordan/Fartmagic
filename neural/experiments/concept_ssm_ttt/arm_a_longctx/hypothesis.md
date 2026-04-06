# Arm A: Long Context Training (4096)

**Variable:** TRAIN_SEQ_LEN=4096 (default 2048)
**Base:** Rascal III
**Mechanism:** Train on 4096-token sequences instead of 2048. Each step sees 2x the context but processes half as many sequences per batch (same total tokens). RoPE NTK-aware scaling already handles the extended positions.
**Hypothesis:** Longer context during training improves BPB because the model learns longer-range dependencies. The sliding-window eval (stride=64) particularly benefits from models trained on longer context.
**Risk:** Fewer sequences per batch may hurt diversity. Attention cost scales O(n²) so steps may be slower. May need to reduce batch size.
**Gate:** 2000 steps, 4×GPU, seed=300. Compare sw BPB and ms/step vs Rascal III control.
