#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
from pathlib import Path


def _last_float_pair(text: str, pattern: str) -> tuple[float, float] | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    a, b = matches[-1]
    return float(a), float(b)


def _last_int(text: str, pattern: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return int(matches[-1])


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    seed_match = re.search(r"RESULT .* seed=(\d+)", text)
    if seed_match is None:
        seed_match = re.search(r"^seed:(\d+)$", text, flags=re.MULTILINE)
    if seed_match is None:
        raise ValueError(f"Could not find seed in {path}")
    seed = int(seed_match.group(1))

    diag = _last_float_pair(
        text,
        r"DIAGNOSTIC post_ema val_loss:([0-9.eE+-]+) val_bpb:([0-9.eE+-]+)",
    )
    if diag is None:
        raise ValueError(f"Could not find DIAGNOSTIC val_bpb in {path}")
    _, raw_bpb = diag

    sw = _last_float_pair(
        text,
        r"final_int6_sliding_window_exact val_loss:([0-9.eE+-]+) val_bpb:([0-9.eE+-]+)",
    )
    rt = _last_float_pair(
        text,
        r"final_int6_roundtrip_exact val_loss:([0-9.eE+-]+) val_bpb:([0-9.eE+-]+)",
    )
    if sw is not None:
        val_loss, val_bpb = sw
    elif rt is not None:
        val_loss, val_bpb = rt
    else:
        raise ValueError(f"Could not find final int6 metrics in {path}")

    steps = _last_int(text, r"^\s*steps:\s+(\d+)\s*$")
    train_time_s = _last_int(text, r"^\s*train_time_s:\s+(\d+)\s*$")
    bytes_total = _last_int(text, r"^\s*bytes_total:\s+(\d+)\s+\(limit 16000000\)\s*$")
    bytes_code = _last_int(text, r"^\s*bytes_code:\s+(\d+)\s*$")
    if bytes_total is None:
        bytes_total = _last_int(text, r"Total submission size int6\+\w+:\s+(\d+)\s+bytes")

    legal_match = re.findall(r"^\s*artifact_legal:(yes|no)\s*$", text, flags=re.MULTILINE)
    artifact_legal = legal_match[-1] == "yes" if legal_match else (
        (bytes_total is not None and bytes_total <= 16_000_000)
    )

    out = {
        "seed": seed,
        "val_bpb": round(val_bpb, 4),
        "val_bpb_exact": val_bpb,
        "val_loss_exact": val_loss,
        "int6_sw_bpb": val_bpb,
        "raw_bpb": raw_bpb,
        "artifact_legal": artifact_legal,
    }
    if steps is not None:
        out["steps"] = steps
    if train_time_s is not None:
        out["train_time_s"] = train_time_s
    if bytes_total is not None:
        out["bytes_total"] = bytes_total
    if bytes_code is not None:
        out["bytes_code"] = bytes_code
    out["log_file"] = str(path)
    out["_log_name"] = path.name.lower()
    return out


def _pick_seed_entry(entries: list[dict]) -> dict:
    def score(e: dict) -> tuple:
        name = e.get("_log_name", "")
        has_steps = 1 if "steps" in e else 0
        train_time = e.get("train_time_s")
        ten_min_like = 1 if (train_time is not None and train_time <= 900) else 0
        not_4h = 1 if "4h" not in name else 0
        legal = 1 if e.get("artifact_legal", False) else 0
        # Primary: choose competition-like logs; secondary: best bpb.
        return (not_4h, ten_min_like, has_steps, legal, -float(e["val_bpb_exact"]))
    return max(entries, key=score)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build submission.json from TON-E log files.")
    ap.add_argument("--log-glob", default="logs/tone_comp_s*.txt")
    ap.add_argument("--output", default="submission.json")
    ap.add_argument("--name", default="TON-E Rhythm Crawler")
    ap.add_argument("--author", default="Frosty40")
    ap.add_argument("--github-id", default="newjordan")
    ap.add_argument("--hardware", default="8xH100 SXM")
    ap.add_argument(
        "--blurb",
        default=(
            "Nightcrawler Cubed runner with TON-E rhythm overlay "
            "(3 flat + 2 crawler x2), loop-aware GPTQ, selective pruning"
        ),
    )
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.log_glob))
    if not paths:
        raise SystemExit(f"No logs matched: {args.log_glob}")

    parsed_all = [parse_log(p) for p in paths]
    by_seed: dict[int, list[dict]] = {}
    for item in parsed_all:
        by_seed.setdefault(item["seed"], []).append(item)
    parsed = [_pick_seed_entry(items) for _, items in sorted(by_seed.items())]
    mean_bpb = sum(x["val_bpb_exact"] for x in parsed) / len(parsed)

    out: dict[str, object] = {
        "author": args.author,
        "github_id": args.github_id,
        "name": args.name,
        "blurb": args.blurb,
        "date": f"{dt.datetime.now(dt.UTC).date().isoformat()}T00:00:00Z",
    }
    for item in parsed:
        out[f"seed_{item['seed']}"] = {
            "val_bpb": item["val_bpb"],
            "val_bpb_exact": item["val_bpb_exact"],
            "val_loss_exact": item["val_loss_exact"],
            "int6_sw_bpb": item["int6_sw_bpb"],
            **({"steps": item["steps"]} if "steps" in item else {}),
            **({"train_time_s": item["train_time_s"]} if "train_time_s" in item else {}),
            **({"bytes_total": item["bytes_total"]} if "bytes_total" in item else {}),
            "artifact_legal": item["artifact_legal"],
            "log_file": item["log_file"],
        }
    out["val_bpb"] = round(mean_bpb, 4)
    bytes_totals = [x.get("bytes_total") for x in parsed if x.get("bytes_total") is not None]
    if bytes_totals:
        out["bytes_total"] = int(max(bytes_totals))
    bytes_codes = [x.get("bytes_code") for x in parsed if x.get("bytes_code") is not None]
    if bytes_codes:
        out["bytes_code"] = int(max(bytes_codes))
    out["hardware"] = args.hardware

    output_path = Path(args.output)
    output_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
