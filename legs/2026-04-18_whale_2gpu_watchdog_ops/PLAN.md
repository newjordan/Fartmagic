# PLAN

## Scope
Ops automation only for the whale 2x pod.

- Local repo: `/home/frosty40/SOTA_FINAL`
- Pod: `root@34.150.151.125 -p 9588`
- SSH key: `/home/frosty40/.ssh/id_ed25519_apollo`
- Remote runtime: `/workspace/SOTA_FINAL/legs/2026-04-17_whale_2gpu_pruned_search_runtime`
- Frozen files remain untouched.

## Design

### 1. Local watchdog leg
Create a new local ops leg that owns:
- staged queue
- queue state
- heartbeat/status snapshots
- cron entry
- remote launcher payload

### 2. Remote execution model
Do **not** modify the live runtime scripts.
Instead, create a separate remote ops subdir under the runtime:
- `/workspace/SOTA_FINAL/legs/2026-04-17_whale_2gpu_pruned_search_runtime/ops_watchdog`

Remote launcher responsibilities:
- run one serial chain per GPU
- call the existing absolute-path `run_single_case.sh`
- continue through the chain even if one case fails
- write remote chain logs and action log

### 3. Cron tick behavior
Every minute, under `flock`:
1. refresh/deploy the remote launcher
2. query remote active processes
3. write local heartbeat/status
4. if active: log heartbeat and exit
5. if idle: pop the next queue pair and launch one chain on GPU 0 and one on GPU 1

### 4. Status model
Local truth files:
- `status/heartbeat.json`
- `status/heartbeat.log`
- `status/last_launch.json`
- `status/queue_state.json`

Remote truth files:
- `ops_watchdog/action.log`
- `ops_watchdog/status/heartbeat.txt`
- `ops_watchdog/status/gpu0.current`
- `ops_watchdog/status/gpu1.current`
- `ops_watchdog/status/gpu0.last`
- `ops_watchdog/status/gpu1.last`

### 5. Queue layout
Exploit-first heavy queue now has two layers:

1. historical staged backlog
2. live-head exploit/explore insertions when the frontier shifts

Current lead family:
- `DQ=128,64,8,5`
- `DKDV=128,128,8,3`
- `DKDV_MAXNREG=256`

Current live queue policy:
- GPU 0 exploits the best known Hybrid B basin floor
- GPU 1 explores adjacent horizons and orthogonal interaction planes on the same family
- challenger families remain in the deeper staged backlog, but the live head can be rewritten to preserve the current search policy

Examples of controlled exploration:
- lower horizon below `1664`
- upper horizon above `2304`
- `DKDV_MR` checks at strong Hybrid B points

Queue is organized in **pairs**.
Each pair contains:
- one serial chain for GPU 0
- one serial chain for GPU 1

This matches the requested launch model exactly.

### 6. Downstream readiness
Keep the queue and notes ready for the next gates:

1. multi-GPU signal
2. whale-donor integration

Those gates stay downstream until the single-GPU frontier is bracketed tightly enough to choose a winner with confidence.

### 7. Safe disable path
Two safe stop levers:
1. touch the local disable flag so cron stops launching new work but does not kill active chains
2. remove the cron line entirely once the active chains finish
