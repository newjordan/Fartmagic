# Whale OOB Experiment Shotgun

Purpose: keep a working list of big-swing experiment concepts that could move the needle more than scalar knob tuning.

## Principles

- Prefer structural or policy shifts over single global knob sweeps.
- Treat the `12L + 576` donor as a source of untapped truth.
- Focus on where quantization destroys that truth, not on tiny global tweaks.
- Use this as a concept bank; it is not a ranked execution plan yet.

## 30 Concepts

1. `Late-Layer Sanctuary`
   - Keep the last 2-4 layers at looser quant or higher bits while the rest stays aggressive.

2. `MLP Sanctuary`
   - Protect only `mlp_up` and `mlp_down` in the top layers to see if the new truth lives in depth-MLP capacity.

3. `KV Sanctuary`
   - Protect only `kv` banks in late layers and leave everything else harshly compressed.

4. `QO Sanctuary`
   - Protect only `q/o` banks in late layers and see if attention output is the sensitive path.

5. `Control Sanctuary`
   - Keep `q_gain`, scales, gates, `resid_mix`, and skip tensors in high precision no matter what.

6. `Late-Clip Relief`
   - Apply looser clipping only to the last layers instead of one global clip value.

7. `Early-Hard / Late-Soft`
   - Quantize early layers brutally and give late layers more room under the same byte budget.

8. `Layer-Band Quant`
   - Split model into `early / middle / late` bands with separate bit and clip policies.

9. `Tensor-Group Quant`
   - Split policy by `qo`, `kv`, `mlp_up`, `mlp_down`, `embed/head`, and controls.

10. `Per-Layer Hessian Budget`
    - Spend most GPTQ calibration on the layers that matter most instead of evenly everywhere.

11. `Fake-Quant Warmdown`
    - Train normally, then do the final warmdown with fake quant pressure so the model learns to survive compression.

12. `STE Top-Layer Retune`
    - Post-train, retune only the last 2-4 layers with straight-through fake quant.

13. `Adapter Rescue`
    - Add tiny FP16 residual adapters only on the late layers after quant to recover truth cheaply.

14. `Low-Rank Error Patch`
    - Store quantization error for sensitive layers as small low-rank sidecars instead of fully lifting their bits.

15. `Post-Quant Distillation`
    - Force the quantized model to match the float donor's logits on a calibration stream.

16. `Disagreement Curriculum`
    - Train or retune only on tokens where float donor and quantized model disagree most.

17. `TTT Top-Only`
    - During TTT, update only the late layers instead of the full model.

18. `TTT Controls-Only`
    - During TTT, update only scales, gates, and control tensors.

19. `Channel Equalization`
    - Rescale in/out channels before GPTQ so outliers stop wrecking the global quant scale.

20. `Outlier Row Escape Hatch`
    - Pull the worst outlier rows or columns into a tiny higher-precision side path.

21. `Depth-Only Extension`
    - Try `13L` or `14L` with less width to see if more serial thought beats more channels.

22. `Synthetic Depth`
    - Replace a full extra layer with extra recurrent passes in the late stack.

23. `Late Expert Branch`
    - Add a tiny gated expert only in the last 2 layers so extra capacity is concentrated where it matters.

24. `Aux Truth Head`
    - Train with an auxiliary late-layer prediction head, then remove it at inference.

25. `Stochastic Continuation`
    - Launch several structural branches for `1k-2k` steps, then continue only the best one.

26. `Hard-Example Curriculum`
    - Overweight hard shards or hard spans instead of uniform training once the model is competent.

27. `Context Pressure Schedule`
    - Shorter context early, longer context late, so late training spends more compute on abstraction.

28. `Quant-Noise Immunization`
    - Inject quant-like weight noise late in training so the model becomes robust before GPTQ.

29. `Constrained Policy Search`
    - Search mixed layer/group quant policies under a strict byte cap instead of hand-picking one global rule.

30. `Throughput Moonshot`
    - If Triton or systems work buys materially more steps in the same wallclock, spend that budget on bigger structural bets.

## Immediate Read

- The strongest legal donor still looks like a `12L + 576` branch.
- The evidence so far says uniform quant policy is too blunt.
- The most likely real edge is in non-uniform quant by layer band or tensor group.
- If bigger structural bets resume, they should be paired with a non-uniform compression plan from the start.
