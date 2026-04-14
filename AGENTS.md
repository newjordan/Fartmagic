# Agent Rules (Codex / Claude / all agents)

All rules in CLAUDE.md apply here. This file exists because Codex reads AGENTS.md.

## FROZEN FILES — do not edit without explicit user approval

- `scripts/Im_sorry_pod_setup.sh` — pod bootstrap script, validated against real hardware.
- `scripts/pod_stack_guard.sh` — hash guard for the above.
- `scripts/pod_stack.lock` — hash lock for the above.
- `vault/` — all files in vault are frozen.

## Pod Environment (do not guess, do not change)

The pod runs:
- Python 3.12.13 (conda-forge, at /venv/main/bin/python3)
- PyTorch 2.11.0+cu130
- CUDA 13.0 (driver 580.126.09)
- Image: vastai/pytorch:cuda-13.0.2-auto
- 8x H100 SXM (81 GB each)
- FA3: flash_attn_3-3.0.0 abi3 wheel for **cu130**

The CUDA tag is cu130. Not cu124. Not cu128. Do not change the FA3 wheel URL.
