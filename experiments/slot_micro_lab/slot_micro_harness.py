from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import zstandard
    COMPRESSOR = "zstd"
except ImportError:  # pragma: no cover
    import zlib
    zstandard = None
    COMPRESSOR = "zlib"


@dataclass(frozen=True)
class CaseResult:
    name: str
    raw_bytes: int
    compressed_bytes: int
    notes: str


class TinyModel(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden, bias=False)
        self.mid_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, dim, bias=False)
        self.slot_bias = nn.Parameter(torch.zeros(1, 1, hidden), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = F.gelu(self.mid_proj(h))
        return self.out_proj(h)

    def slot_helper(self, x: torch.Tensor, steps: int = 2, lr: float = 1e-3) -> torch.Tensor:
        delta = torch.zeros(1, 1, x.shape[-1], device=x.device, requires_grad=True)
        opt = torch.optim.AdamW([delta], lr=lr, weight_decay=1e-8, eps=1e-5)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            y = self.forward(x + delta)
            loss = y.square().mean()
            loss.backward()
            opt.step()
        return delta.detach()


def serialize_state_dict(sd: dict[str, torch.Tensor]) -> tuple[int, int]:
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw = buf.getvalue()
    if COMPRESSOR == "zstd":
        comp = zstandard.ZstdCompressor(level=22).compress(raw)
    else:  # pragma: no cover
        comp = zlib.compress(raw, 9)
    return len(raw), len(comp)


def train_tiny(case: str, steps: int = 10, seed: int = 1337) -> CaseResult:
    torch.manual_seed(seed)
    model = TinyModel()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randn(4, 8, 64)
    target = torch.randn(4, 8, 64)

    slot_delta = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()
        if case == "contam_pre_export":
            with torch.no_grad():
                model.slot_bias.add_(0.001)

    export_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if case == "contam_state_dict":
        export_sd["slot_delta"] = torch.zeros(1, 1, 64)

    raw_bytes, compressed_bytes = serialize_state_dict(export_sd)

    if case == "slot_post_export":
        slot_delta = model.slot_helper(x[:, :1, :], steps=2, lr=1e-3)
        _ = slot_delta.sum().item()

    if case == "slot_helper_unused":
        def _unused_slot_helper(y: torch.Tensor) -> torch.Tensor:
            return y.mean(dim=-1, keepdim=True)
        _ = _unused_slot_helper(x)

    note = "baseline"
    if case == "slot_helper_unused":
        note = "slot helper defined but never affects export"
    elif case == "slot_post_export":
        note = "slot helper runs after export bytes are fixed"
    elif case == "contam_state_dict":
        note = "extra tensor enters exported state dict"
    elif case == "contam_pre_export":
        note = "persistent model tensor mutated before export"

    return CaseResult(case, raw_bytes, compressed_bytes, note)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cases = args.cases or [
        "ctrl",
        "slot_helper_unused",
        "slot_post_export",
        "contam_state_dict",
        "contam_pre_export",
    ]

    results: list[CaseResult] = []
    for case in cases:
        results.append(train_tiny(case, steps=args.steps, seed=args.seed))

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
        return 0

    print(f"compressor={COMPRESSOR} steps={args.steps} seed={args.seed}")
    print("| case | raw bytes | compressed bytes | note |")
    print("|---|---:|---:|---|")
    for r in results:
        print(f"| {r.name} | {r.raw_bytes} | {r.compressed_bytes} | {r.notes} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
