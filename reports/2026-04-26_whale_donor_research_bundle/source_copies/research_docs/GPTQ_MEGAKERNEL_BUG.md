# GPTQ silently no-ops on whale megakernel blocks

One-line summary: GPTQ calibration's forward-hook registration filters on `isinstance(module, (nn.Linear, CastedLinear))`, but whale's `CausalSelfAttention` and `MLP` blocks execute projections via raw `F.linear(x, bank_slice)` — no `nn.Linear` submodule exists for the hook to attach to, so **zero layers are calibrated** and GPTQ falls back to uncalibrated per-row clipping.

---

## Root cause (file:line)

All references are to `/home/frosty40/SOTA_FINAL/vault/train_gpt_midnight_iii_base.py` at commit `2e07aea` (HEAD of `whale/2026-04-16_pod_autoresearch`).

### GPTQ layer discovery (the filter)
`fact` L684-686:
```python
for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, CastedLinear)):
        hooks.append(module.register_forward_hook(make_hook(name)))
```
Only `nn.Linear` / `CastedLinear` modules receive hooks. `F.linear` is a functional call — there is no module to register against.

### The megakernel projections that go uncalibrated
`fact` L1109 (comment inside `CausalSelfAttention`): `# No CastedLinear -- weights come from banks`
`fact` L1135-1137: Q/K/V projections use raw functional linear on bank slices:
```python
q = F.linear(x, q_w.to(x.dtype)).reshape(...)
k = F.linear(x, k_w.to(x.dtype)).reshape(...)
v = F.linear(x, v_w.to(x.dtype))
```
`fact` L1174: attention output projection is also `F.linear(y, out_w.to(x.dtype))`.
`fact` L1239 (comment inside `MLP`): `# No CastedLinear -- weights come from banks`
`fact` L1242-1244: MLP up/down projections use `F.linear(x, up_w.to(x.dtype))` / `F.linear(x, down_w.to(x.dtype))`.

The banks live as 3D parameter tensors (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`, L1370-1373) — they are sliced per-layer at forward time and handed to `F.linear`. Because no `nn.Linear` wraps them, GPTQ's `named_modules()` walk never reaches them.

### What survives
`fact` L1117 `attn_gate = nn.Linear(dim, num_heads, bias=True)` — hooked iff `gated_attention=True`
`fact` L1192 `BigramHashEmbedding.proj = CastedLinear(...)` — hooked iff `bigram_dim != model_dim`
`fact` L1226 `ValueEmbedding.proj = CastedLinear(...)` — hooked iff `ve_dim != model_dim`
`fact` L1409 `lm_head` (CastedLinear) — hooked iff `tie_embeddings=False`
`fact` L1413 `mtp_heads` — hooked

With `tie_embeddings=True` and the other small modules absent, **zero modules qualify** → `gptq:calibrated 0 layers`.

### The fallback path (why the model still "runs" with +0.24 BPB)
`fact` L762-775 in `mixed_quantize_gptq`:
```python
if t.ndim == 2:
    H = hessians.get(module_name)
    if H is not None and H.shape[0] == t.shape[1]:
        # GPTQ path
    else:
        # naive quantize_lowbit_per_row
elif t.ndim >= 1:
    quantize_lowbit_per_row(t_2d, bits)  # 3D banks reshape here
```
Banks are 3D → they go to `quantize_lowbit_per_row` (L701-717) regardless. With zero Hessians collected, the "GPTQ" branch is never entered for any 2D tensor either. Everything gets naive per-row scale-sweep clipping. This is why the collapse is silent — the run completes but quality craters.

---

## Detection

### The log line to watch for
`fact` L2462:
```python
log0(f"gptq:calibrated {len(gptq_hessians)} layers in {time.perf_counter()-t_gptq:.1f}s")
```

### Observed evidence of this bug firing on whale runs
`fact` — corpus matches for the exact `gptq:calibrated 0 layers` string:
- `evidence/raw_transcripts/2026-04-13_midnight_iii_v_qattn6_seed300_pod_run_raw_transcript.txt:107` — `gptq:calibrated 0 layers in 3.0s` (2026-04-13)
- `evidence/pod_pulls/instance_34775495_20260413_011337/extracted/metrics_summary.txt:11` cites `.../full_seed444_20260413_044438.log:93: gptq:calibrated 0 layers in 2.9s`
- `evidence/pod_pulls/instance_34775495_20260413_011337/extracted/metrics_summary.txt:17` cites `.../full_seed300_20260413_050254.log:93: gptq:calibrated 0 layers in 3.0s`
- `evidence/pod_pulls/instance_34775495_20260413_011337/extracted/metrics_summary.txt:101,120,139` — three additional pod logs all show `gptq:calibrated 0 layers in ~3s`
- `legs/2026-04-13_midnight_iii_v_qattn6/RESULTS.md:12,18` documents the failure
- `legs/2026-04-14_midnight_iii_v_bank_gptq_fix/hypothesis.md:10` identifies the symptom: "the export path reports `gptq:calibrated 0 layers`, falls back to naive bank quantization, and collapses `final_quant_roundtrip_exact`"
- `legs/2026-04-17_whale_recursion_prep/scratch/RESULTS.md:8` — same failure reproduced: `gptq:calibrated 0 layers` and `mixed_quantize: 0 GPTQ tensors, 4 naive tensors int6:4 int8:1`

Contrast (`fact`): `evidence/pod_pulls/instance_34775495_20260413_011337/extracted/metrics_summary.txt:22` — a legacy pre-whale awing_green1 run shows `gptq:calibrated 68 layers in 3.6s`, confirming the calibrator itself works on `nn.Linear`-heavy architectures.

### Quick detection rule
If `gptq:calibrated N layers` prints with `N < expected_count`, the run is compromised. For the current whale config with `tie_embeddings=True`, expected count is approximately 1-2 (bigram/ve proj only) — NOT the 48-72 that matters. For any config that intends to quantize attention/MLP banks at `bits in {4,5,6,7}`, the calibrator MUST have covered them, and currently cannot.

---

## How to fix properly

Two independent options; pick one or both.

### Option A — Unwrap bank projections into hookable modules (cleanest)
Replace each `F.linear(x, bank_slice)` call site with a thin wrapper (e.g., `CastedLinear` that reads its weight from a provided slice), so that `named_modules()` sees it and GPTQ's hook registers. Keep the bank storage, but route the forward through a per-layer module that GPTQ can observe.

Locations to edit (in a new leg copy, not vault/):
- `CausalSelfAttention.forward` — L1135-1137, L1174
- `MLP.forward` — L1242, L1244

Each of the four call sites becomes a per-block per-projection submodule. Cost: +4 modules per block × 12 blocks = 48 modules, sufficient for per-slice Hessians.

### Option B — Bank-aware custom hook pass
Instead of hooking `nn.Linear`, register manual pre-hooks on each `Block` that capture the input `x` before the Q/K/V/out/up/down calls, and accumulate `H = x.t() @ x` keyed by the bank slice name. Then `mixed_quantize_gptq` L762 lookup (`hessians.get(module_name)`) succeeds for 3D bank tensors keyed by `{bank_name}[layer_idx]`.

This is the approach `legs/2026-04-14_midnight_iii_v_bank_gptq_fix/` started. See `legs/2026-04-17_whale_recursion_prep/patches_drafts.md` for concrete patch sketches (patch #1 + #3).

### Option C (NOT recommended) — Monkeypatch `F.linear`
Fragile; interacts badly with flash-attn and torch.compile. Mentioned only to explicitly reject it.

---

## Scope: vortex all-int8 per-channel path is NOT affected

`fact` L590, L753-755: the `bits == 8` branch in `mixed_quantize_gptq` calls `quantize_float_tensor` (L568-583), which computes a per-row clip via `torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)` directly on the weight tensor — **no Hessian lookup, no hooks needed**. The int8 per-channel path operates on whatever the state_dict contains (2D or 3D reshaped) and is correct regardless of how the module is wrapped.

`inference`: if a run configures all quant categories to `bits=8`, the whole `gptq_calibrate` call is skipped by the `_needs_lowbit_gptq` guard at L2455-2457 (`log0("gptq:SKIPPED (all quant categories are int8 or float passthrough)")`), so the bug cannot bite.

The bug fires only when at least one category is in `{4,5,6,7}` AND the quantized tensor is a whale megakernel bank.

---

## Proposed assert (not implemented)

One line, drop-in after L2462 in a new leg. Compute expected count from the model's actual hookable-module count and from the quant policy before the calibrator runs, then fail loudly:

```python
# proposal — replaces the bare log line:
assert len(gptq_hessians) >= expected_gptq_layers(base_model, args), (
    f"gptq:calibrated {len(gptq_hessians)} layers — expected >= {expected_gptq_layers(base_model, args)}. "
    f"Megakernel bank projections are unhooked (F.linear on bank slices). "
    f"See docs/GPTQ_MEGAKERNEL_BUG.md. Set SKIP_GPTQ=1 only if this is intentional."
)
```

Where `expected_gptq_layers(model, args)` is a helper that:
1. Walks `named_modules()` and counts `nn.Linear`/`CastedLinear` instances the hook WOULD catch.
2. Adds the count of bank-slice projections that SHOULD be calibrated given the quant policy (attn Q/K/V/out/up/down × num_layers when any of those categories is in `{4,5,6,7}`).
3. Returns the sum.

Minimal version (strictly asserting the bug symptom): `assert len(gptq_hessians) > 0, "gptq no-op — see docs/GPTQ_MEGAKERNEL_BUG.md"`. This catches the exact case observed in the 2026-04-13 logs but misses partial-hook scenarios where e.g. only `attn_gate` gets calibrated.

---

## Related artifacts

- `legs/2026-04-17_whale_recursion_prep/quant_audit.md` — full calibration-path audit with per-line references; independently reached the same conclusion
- `legs/2026-04-17_whale_recursion_prep/patches_drafts.md` — draft patches for option B (bank-aware hooks)
- `legs/2026-04-14_midnight_iii_v_bank_gptq_fix/` — first attempt leg that identified the symptom on 2026-04-14
- `legs/2026-04-13_midnight_iii_v_qattn6/RESULTS.md` — original +0.24 BPB impact data
