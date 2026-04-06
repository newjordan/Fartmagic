# Arm C: SSM Hybrid (Mamba-style early layers)

**Variable:** SSM_HYBRID=1, SSM_LAYERS=4, SSM_STATE_DIM=64
**Base:** Rascal III
**Mechanism:** Replace attention in the first 4 layers with a diagonal state-space model. Early layers typically learn local patterns that don't need full O(n^2) attention. The last 7 layers keep full attention for complex long-range dependencies. The SSM carries context forward in a fixed-size state vector, providing O(n) context processing.
**Hypothesis:** Early layers waste attention capacity on local patterns. SSM handles these efficiently, freeing compute for the attention-heavy deep layers. May also improve training speed (SSM is cheaper than attention).
**Size impact:** SSM adds ~4x(512x128 + 64 + 512x64) = ~400K params. Attention banks for those layers still exist but are unused -- could be pruned for size savings.
**Risk:** Sequential SSM scan is slow in pure PyTorch. Quality may suffer if early layers need attention. The unused attention bank weights waste size budget.
**Note:** torch.compile fullgraph=1 is incompatible with the sequential scan loop. Gate uses COMPILE_FULLGRAPH=0.
**Gate:** 2000 steps, 4xGPU, seed=300. Compare sw BPB, ms/step, and artifact size vs Rascal III control.
