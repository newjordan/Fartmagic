# Hypothesis

If we separate ops from search logic and drive the existing remote runtime through a strict per-minute watchdog, the whale 2x pod can stay saturated for long stretches while the queue head is rewritten around the real frontier instead of blindly draining historical blocks.

Operational assumptions:
- The pod should be treated as **busy** when direct process checks show active `bench_stable_scratch.py` or `run_single_case.sh` work.
- To avoid minute-boundary races between sequential cases in the same chain, the watchdog may also respect the dedicated remote chain-launcher process as an ops sentinel.
- Search logic stays frozen except for **queue ordering**.
- Heavy validation only: `rounds=10`, `iters=200`.
- DQ stays fixed to `128,64,8,5` for the main queue; a small late fallback block for `128,64,8,2` is acceptable only as a sanity tail.

Queue policy encoded here:
1. Rank by absolute whale time first, not by delta alone.
2. Exploit the strongest current family first: `DQ=128,64,8,5 / DKDV=128,128,8,3 / DKDV_MAXNREG=256`.
3. Use the second GPU for controlled exploration of adjacent horizons and orthogonal interaction planes.
4. Keep challenger families in the deeper queue, but do not let them steal the live head when a better frontier has already emerged.
5. Press farther in the winning direction than normal, then wind down around the last winner with tighter spacing.
6. Never overlap work on the same GPU.
