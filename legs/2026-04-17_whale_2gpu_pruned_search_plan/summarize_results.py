#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parent / 'evidence'
rows = []
for path in sorted(root.glob('*.json')):
    payload = json.loads(path.read_text(encoding='utf-8'))
    for entry in payload.get('results', []):
        backends = entry.get('backends', {})
        whale = backends.get('whale_fast', {})
        fa3 = backends.get('fa3', {})
        if not whale or not fa3:
            continue
        rows.append({
            'file': path.name,
            'shape': tuple(entry.get('shape', [])),
            'whale_fb_ms': whale.get('fb_mean_ms'),
            'fa3_fb_ms': fa3.get('fb_mean_ms'),
            'delta_ms': whale.get('fb_mean_ms') - fa3.get('fb_mean_ms'),
            'delta_pct': ((whale.get('fb_mean_ms') / fa3.get('fb_mean_ms')) - 1.0) * 100.0,
            'env_bkv_tma': payload.get('env_bkv_tma'),
            'env_bkv_tma_maxnreg': payload.get('env_bkv_tma_maxnreg'),
            'env_bq': payload.get('env_bq'),
            'env_bq_maxnreg': payload.get('env_bq_maxnreg'),
            'label': payload.get('label'),
        })

rows.sort(key=lambda row: (row['whale_fb_ms'], row['delta_ms']))
for idx, row in enumerate(rows, start=1):
    print(
        f"{idx:02d} {row['file']} whale={row['whale_fb_ms']:.6f} fa3={row['fa3_fb_ms']:.6f} "
        f"delta_ms={row['delta_ms']:+.6f} delta_pct={row['delta_pct']:+.2f}% "
        f"dkdv={row['env_bkv_tma']} dkdv_mr={row['env_bkv_tma_maxnreg']} "
        f"dq={row['env_bq']} dq_mr={row['env_bq_maxnreg']}"
    )
