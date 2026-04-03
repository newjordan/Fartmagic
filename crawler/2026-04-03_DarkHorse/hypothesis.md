# Dark Horse — Hypothesis

## Concept
Port Helix (bidirectional cross-stream co-firing) onto the PR #1296 codebase
(SP4096 + depth recurrence + parallel residuals + MuonEq-R + QK5 + GPTQ + brotli).

Their 1.0897 BPP platform + our Helix architecture = a genuinely differentiated submission
on a competitive baseline.

## Parent
PR #1296 (aryanbhosale): 1.0897 BPB, 3-seed mean, ~15.99MB
- SP4096 vocab, MLP 4x, WD 0.090
- Depth recurrence layers 4,5 (start step 3000)
- Parallel residuals from layer 7
- MuonEq-R optimizer
- QK-Gain 5.0
- Full GPTQ int6 + brotli compressed wrapper

## What we add
1. **HELIX=1** — dual-stream co-firing with linear projection bridge
2. **HELIX_DIM=~192** — fat pipe (~37% of model_dim=512), based on micro findings
3. **HELIX_STRIDE=1** — frequent firing (confirmed best at fat dims)
4. **Loop-aware GPTQ** — our 2-phase Hessian recalibration (shields shared-weight quant damage)
5. **CRAWLER_LOOPS=1** — no sequential loops (helix is the recurrence)

## Why this should work
- Their depth recurrence (layers 4,5 fired twice) amplifies quant damage
  (we proved: 0.095 BPP quant gap without helix)
- Our helix shields against this quant damage (drops gap to 0.004)
- Their parallel residuals from layer 7 are orthogonal to helix cross-injection
- MuonEq-R optimizer benefits all architectures equally
- SP4096 tokenizer lifts the entire baseline

## Architecture mapping
Their GPT class uses a U-Net encoder/decoder with virtual layer mapping for recurrence.
Helix integration points:
1. Add crawler block (shared weights, fires at each virtual layer step)
2. Add cross-injection projections (flat↔crawler bidirectional)
3. Add merge gate at output
4. Their `_get_virtual_layers()` already handles recurrence — helix fires at each virtual step

## Risk
- Their codebase is different from ours — integration may have bugs
- MuonEq-R + helix cross-injection gradients are untested
- SP4096 may change the optimal helix_dim ratio
- Code size with helix additions may exceed brotli compression headroom

## Plan
1. First: micro test their base code unmodified (verify we reproduce ~1.09)
2. Then: add helix modules to their GPT class
3. Micro test: helix on vs off on their base
4. If signal: full 8×H100 600s production run
