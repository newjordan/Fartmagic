# B-WING III — LoRA TTT + Cubric + All #809 N-gram

## What's new vs bwing_II
- REPLACED our slow full-weight SGD TTT (410s, -0.0025 BPB) with
  PR #809's fast LoRA TTT (53s, -0.015 BPB)
- LoRA adapters on Q, V, LM head (rank 8)
- Per-document batched (64 docs), AdamW, Polyak averaging
- No cross-GPU sync needed (each rank processes independent docs)

## Full stack
1. Train: complementary training (alpha=0.5)
2. Export: GPTQ int6+zstd
3. TTT: LoRA adapters, ~53s, adapts model before n-gram
4. N-gram: cubric 3D + entropy shift + alpha 0.05-0.60 clip 0.95
