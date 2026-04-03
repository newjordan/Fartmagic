# Helix FlatServe — Hypothesis

## Core Idea
The flat layers serve the crawler. Modify flat layer behavior so they produce
representations that the crawler's shared weights can maximally denoise.

## What We Know
- Crawler shared weights learn a universal denoising function
- Early flat layers produce noisy, half-formed representations
- Helix cross-injection creates co-adaptation pressure between flat and crawler
- The crawler fires at every flat layer — it sees the full noise→signal trajectory
- Fat bridge (dim=192) works because it gives flat layers room to express STRUCTURED noise

## Modifications to Test

### 1. Residual Scaling (FLAT_RESIDUAL_SCALE)
Standard: x = x + block(x)
Modified: x = x + alpha * block(x)  where alpha < 1.0

Make flat layers produce SMALLER updates. Forces the model to take many small
steps instead of few large ones. The crawler sees a smoother trajectory with
more correctable noise per step. The crawler's job gets easier — many small
corrections instead of few large ones.

Test: alpha = 0.5, 0.7, 0.9, 1.0 (control)

### 2. Noise Injection (FLAT_NOISE_STD)
Standard: x = block(x)
Modified: x = block(x) + noise * std

Deliberately inject noise into flat layer outputs during training. Forces the
crawler to learn robust denoising. The flat layers learn to produce representations
that are noise-tolerant because the crawler will clean up. At eval time, no noise —
the crawler's learned denoising still fires but on already-clean signal = bonus.

This is dropout's cousin applied to the residual stream, but the crawler is
explicitly trained to undo it.

Test: std = 0.0 (control), 0.01, 0.05, 0.1

### 3. Progressive Delegation (FLAT_DELEGATE_SCHEDULE)
Standard: all flat layers have full capacity
Modified: early flat layers are WEAKER (smaller MLP mult or fewer heads),
         later flat layers are STRONGER

Force early layers to produce incomplete representations that NEED the crawler.
The crawler handles early-stage denoising (its specialty via shared weights).
Later layers get full capacity to build on the crawler-cleaned signal.

Test: early_mlp_mult=2.0 for layers 0-2, normal 3.0 for rest

### 4. Crawler-Aligned Output Projection (FLAT_CRAWL_PROJ)
Standard: flat layer output goes directly to residual
Modified: flat layer output goes through a learned projection that aligns
         it with what the crawler expects

Each flat layer gets a small projection: Linear(model_dim, model_dim, bias=False)
initialized to identity. The projection learns to rotate the flat output into
a basis the crawler can efficiently process. The flat layers learn to "speak
the crawler's language."

Test: FLAT_CRAWL_PROJ=1 vs 0

### 5. Asymmetric Skip Connections (FLAT_SKIP_TO_CRAWL)
Standard: U-Net skips go flat→flat (encoder→decoder)
Modified: some skips route through the crawler instead

Encoder skip outputs go to the crawler stream. The crawler processes them
and injects the result into the decoder. This forces ALL information in the
U-Net to pass through the crawler's shared weights at least once.

Test: FLAT_SKIP_TO_CRAWL=1 vs 0 (standard)

## Priority Order
1. Residual scaling — simplest, one env var, tests the "many small steps" theory
2. Noise injection — tests whether crawler learns robust denoising
3. Asymmetric skips — tests whether routing everything through crawler helps
4. Crawler-aligned projection — tests whether flat→crawler alignment matters
5. Progressive delegation — most complex, tests capacity redistribution
