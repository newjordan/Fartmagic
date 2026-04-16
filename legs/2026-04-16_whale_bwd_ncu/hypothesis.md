# whale bwd ncu profile — hypothesis

## fact
- Headline (B=4,T=2048,H=8,KV=4,D=64, bf16, causal): whale fwd+bwd wall = 352us, GPU-only event = 352us.
- CPU overhead (165us) fully hidden behind GPU work. Launcher bypass saves ~0us here.
- Remaining ~10us gap to FA3 at headline must come from on-device work.

## goal
Identify the real hotspot in the pure-Triton bwd via ncu --set full:
- Is dkdv_inline memory-bound or compute-bound?
- Are we stalled on smem banks, MIO throttle, LG throttle, or math?
- What is the register spill count? Occupancy?
- How does FA3 bwd compare on the same metrics?

## plan
1. Minimal runner that calls whale bwd kernels N times.
2. ncu --set full --target-processes all --launch-count K for whale.
3. Same for FA3 bwd via flash_attn_3 path.
4. Diff the two reports on SM/tensor utilization, MIO throttle, issued IPC, stall reasons.
5. RESULTS.md with specific actionable next step or "no further on-device win".
