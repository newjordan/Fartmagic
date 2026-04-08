#!/usr/bin/env python3
"""Generate a Midnight 12L architecture point cloud for the Shroud viewer.

The output is a compact, fully labeled architecture visualization derived from
actual model facts:
- 12 layers
- GQA 8/4
- attn int5, mlp int6, aux int6, embed int8, other int8
- RoPE 16
- Bigram 2048
- XSA active on the last 11 layers

It produces a viewer-ready points JSON plus a companion architecture-flow JSON.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True, help="Path to midnight_12l_spec.json")
    p.add_argument("--output-points", required=True, help="Output points JSON path")
    p.add_argument("--output-flow", required=True, help="Output architecture flow JSON path")
    return p.parse_args()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, clamp(l, 0.0, 1.0), clamp(s, 0.0, 1.0))
    return float(r), float(g), float(b)


def rgb_from_palette(palette: dict[str, list[float]], key: str, *, lightness_boost: float = 0.0, hue_shift: float = 0.0) -> tuple[float, float, float]:
    h, s, l = palette[key]
    return hsl_to_rgb(h + hue_shift, s, clamp(l + lightness_boost, 0.0, 1.0))


class Builder:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.facts = spec["facts"]
        self.layout = spec["layout"]
        self.palette = spec["palette"]
        self.points: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.flow_nodes: dict[str, int] = {}
        self.flow_edges: list[dict[str, Any]] = []

    def add_point(self, key: str, *, x: float, y: float, z: float, color: tuple[float, float, float], label: str, **meta: Any) -> int:
        idx = len(self.points)
        point = {
            "x": x,
            "y": y,
            "z": z,
            "r": color[0],
            "g": color[1],
            "b": color[2],
            "label": label,
            **meta,
        }
        self.points.append(point)
        self.flow_nodes[key] = idx
        return idx

    def add_edge(self, src_key: str, dst_key: str, *, kind: str, magnitude: float = 1.0, color: tuple[float, float, float] | None = None, **meta: Any) -> None:
        if src_key not in self.flow_nodes or dst_key not in self.flow_nodes:
            raise KeyError(f"missing node for edge {src_key!r} -> {dst_key!r}")
        src = self.flow_nodes[src_key]
        dst = self.flow_nodes[dst_key]
        p_src = self.points[src]
        p_dst = self.points[dst]
        if color is None:
            color = (
                clamp(0.5 * (float(p_src["r"]) + float(p_dst["r"])), 0.0, 1.0),
                clamp(0.5 * (float(p_src["g"]) + float(p_dst["g"])), 0.0, 1.0),
                clamp(0.5 * (float(p_src["b"]) + float(p_dst["b"])), 0.0, 1.0),
            )
        edge = {
            "i": src,
            "j": dst,
            "r": color[0],
            "g": color[1],
            "b": color[2],
            "alpha": 0.38,
            "flow": float(magnitude),
            "phase": 0.0,
            "loop": int(meta.get("loop", 0)),
            "block": int(meta.get("block", 0)),
            "head": int(meta.get("head", -1)),
            "kv_head": int(meta.get("kv_head", -1)),
            "step": int(meta.get("step", 0)),
            "qk_align": float(meta.get("qk_align", 0.0)),
            "transfer": float(meta.get("transfer", 0.0)),
            "attn_entropy": float(meta.get("attn_entropy", 0.0)),
            "attn_lag": float(meta.get("attn_lag", 0.0)),
            "recent_mass": float(meta.get("recent_mass", 0.0)),
            "attn_peak": float(meta.get("attn_peak", 0.0)),
            "out_rms": float(meta.get("out_rms", 0.0)),
            "token_count": int(meta.get("token_count", 0)),
            "flow_kind": kind,
            "src_stage": str(meta.get("src_stage", self.points[src].get("stage", ""))),
            "dst_stage": str(meta.get("dst_stage", self.points[dst].get("stage", ""))),
            "src_key": src_key,
            "dst_key": dst_key,
            "magnitude": float(magnitude),
        }
        self.edges.append(edge)
        self.flow_edges.append({
            "src": src_key,
            "dst": dst_key,
            "kind": kind,
            "weight": float(magnitude),
        })

    def layer_center(self, layer_index: int) -> tuple[float, float, float]:
        layer_spacing = float(self.layout["layer_spacing"])
        base_y = float(self.layout["base_y"])
        theta = 0.42 * (layer_index - 1)
        radius = float(self.layout["core_radius"])
        x = radius * math.cos(theta)
        y = base_y + (layer_index - 1) * layer_spacing
        z = radius * math.sin(theta)
        return x, y, z

    def build_rope_ring(self) -> None:
        radius = float(self.layout["rope_radius"])
        y = float(self.layout["embed_y"]) - 0.04
        for tick in range(16):
            angle = (2.0 * math.pi * tick) / 16.0
            hue = 0.53 + 0.01 * tick
            color = hsl_to_rgb(hue, 0.74, 0.58)
            self.add_point(
                f"rope_{tick:02d}",
                x=radius * math.cos(angle),
                y=y,
                z=radius * math.sin(angle),
                color=color,
                label=f"RoPE tick {tick + 1:02d}/16 | rope_dims=16",
                stage="rope",
                role="rope",
                component="rope",
                layer_index=0,
                bitwidth=0,
                q_heads=0,
                kv_heads=0,
                gqa_ratio=0.0,
                rope_dims=16,
                bigram_dim=0,
                xsa_active=False,
                quant_name="rope-16",
                bit_index=tick + 1,
                bit_total=16,
                head_group=0,
                step=1,
                micro_step=tick,
            )

    def build_bigram_and_embed(self) -> None:
        bigram_color = rgb_from_palette(self.palette, "bigram")
        self.add_point(
            "bigram",
            x=0.0,
            y=float(self.layout["bigram_y"]),
            z=0.0,
            color=bigram_color,
            label="Bigram basis | bigram_dim=2048",
            stage="bigram",
            role="bigram",
            component="input_basis",
            layer_index=0,
            bitwidth=0,
            q_heads=0,
            kv_heads=0,
            gqa_ratio=0.0,
            rope_dims=0,
            bigram_dim=2048,
            xsa_active=False,
            quant_name="bigram-2048",
            bit_index=0,
            bit_total=0,
            head_group=0,
            step=0,
            micro_step=0,
        )

        embed_color = rgb_from_palette(self.palette, "embed")
        embed_bits = int(self.facts["quant_bits"]["embed"])
        for i in range(embed_bits):
            y = float(self.layout["embed_y"]) + i * float(self.layout["bar_point_spacing"])
            self.add_point(
                f"embed_tick_{i:02d}",
                x=-5.25,
                y=y,
                z=-1.15,
                color=embed_color,
                label=f"Embed int8 | bit {i + 1}/{embed_bits}",
                stage="embed",
                role="embed",
                component="embed",
                layer_index=0,
                bitwidth=8,
                q_heads=0,
                kv_heads=0,
                gqa_ratio=0.0,
                rope_dims=0,
                bigram_dim=2048,
                xsa_active=False,
                quant_name="embed=int8",
                bit_index=i + 1,
                bit_total=embed_bits,
                head_group=0,
                step=0,
                micro_step=i,
            )
        self.add_edge(
            "bigram",
            "embed_tick_00",
            kind="tokenize",
            magnitude=1.0,
            color=bigram_color,
            src_stage="bigram",
            dst_stage="embed",
        )

    def build_global_quant_bars(self) -> None:
        bars = [
            ("attn", "attn_q", -2.45, 2.25, 5, "attn_q"),
            ("mlp", "mlp", -0.15, 2.55, 6, "mlp"),
            ("aux", "aux", 2.15, 2.25, 6, "aux"),
            ("other", "other", 4.35, 1.95, 8, "other"),
        ]
        for key, stage, x, z, bits, palette_key in bars:
            color = rgb_from_palette(self.palette, palette_key)
            anchor_key = f"{key}_anchor"
            self.add_point(
                anchor_key,
                x=x,
                y=float(self.layout["base_y"]) + 0.05,
                z=z,
                color=color,
                label=f"{stage.upper()} int{bits}",
                stage=stage,
                role=stage,
                component=key,
                layer_index=0,
                bitwidth=bits,
                q_heads=self.facts["query_heads"] if stage == "attn_q" else 0,
                kv_heads=self.facts["kv_heads"] if stage == "attn_q" else 0,
                gqa_ratio=self.facts["gqa_ratio"] if stage == "attn_q" else 0.0,
                rope_dims=0,
                bigram_dim=0,
                xsa_active=False,
                quant_name=f"{key}=int{bits}",
                bit_index=0,
                bit_total=bits,
                head_group=0,
                step=1,
                micro_step=0,
            )
            self.add_edge("embed_tick_00", anchor_key, kind="quant_route", magnitude=float(bits), color=color, src_stage="embed", dst_stage=stage)
            for bit in range(bits):
                y = float(self.layout["base_y"]) + 0.30 + bit * float(self.layout["bar_point_spacing"])
                tick_key = f"{key}_tick_{bit:02d}"
                rgb = hsl_to_rgb(
                    self.palette[palette_key][0],
                    self.palette[palette_key][1],
                    clamp(self.palette[palette_key][2] + 0.03 * (bit / max(1, bits - 1)) - 0.07, 0.0, 1.0),
                )
                self.add_point(
                    tick_key,
                    x=x,
                    y=y,
                    z=z,
                    color=rgb,
                    label=f"{key.upper()} int{bits} | bit {bit + 1}/{bits}",
                    stage=stage,
                    role=stage,
                    component=key,
                    layer_index=0,
                    bitwidth=bits,
                    q_heads=self.facts["query_heads"] if stage == "attn_q" else 0,
                    kv_heads=self.facts["kv_heads"] if stage == "attn_q" else 0,
                    gqa_ratio=self.facts["gqa_ratio"] if stage == "attn_q" else 0.0,
                    rope_dims=0,
                    bigram_dim=0,
                    xsa_active=False,
                    quant_name=f"{key}=int{bits}",
                    bit_index=bit + 1,
                    bit_total=bits,
                    head_group=0,
                    step=1,
                    micro_step=bit,
                )
                self.add_edge(anchor_key, tick_key, kind="quant_band", magnitude=1.0, color=rgb, src_stage=stage, dst_stage=stage)

    def build_layer_stack(self) -> None:
        layers = int(self.facts["layers"])
        q_heads = int(self.facts["query_heads"])
        kv_heads = int(self.facts["kv_heads"])
        gqa_ratio = float(self.facts["gqa_ratio"])
        for layer in range(1, layers + 1):
            x, y, z = self.layer_center(layer)
            layer_color = hsl_to_rgb(0.70 + 0.02 * layer, 0.62, 0.48 + 0.01 * layer)
            layer_key = f"layer_{layer:02d}"
            self.add_point(
                layer_key,
                x=x,
                y=y,
                z=z,
                color=layer_color,
                label=f"Layer {layer:02d} | XSA={'on' if layer >= 2 else 'off'}",
                stage="layer",
                role="layer",
                component="core",
                layer_index=layer,
                bitwidth=0,
                q_heads=q_heads,
                kv_heads=kv_heads,
                gqa_ratio=gqa_ratio,
                rope_dims=16,
                bigram_dim=0,
                xsa_active=layer >= 2,
                quant_name="layer-core",
                bit_index=0,
                bit_total=0,
                head_group=0,
                step=layer,
                micro_step=0,
            )
            if layer > 1:
                self.add_edge(f"layer_{layer - 1:02d}", layer_key, kind="layer_stack", magnitude=1.0, color=layer_color, src_stage="layer", dst_stage="layer")

            if layer >= 2:
                xsa_color = rgb_from_palette(self.palette, "xsa", lightness_boost=0.02 * ((layer - 2) % 3))
                xsa_key = f"xsa_{layer:02d}"
                self.add_point(
                    xsa_key,
                    x=x + 0.36 * math.cos(layer * 0.35),
                    y=y + 0.34,
                    z=z + 0.36 * math.sin(layer * 0.35),
                    color=xsa_color,
                    label=f"XSA active on layer {layer:02d}",
                    stage="xsa",
                    role="xsa",
                    component="xsa",
                    layer_index=layer,
                    bitwidth=0,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    gqa_ratio=gqa_ratio,
                    rope_dims=16,
                    bigram_dim=0,
                    xsa_active=True,
                    quant_name="xsa",
                    bit_index=0,
                    bit_total=0,
                    head_group=0,
                    step=layer,
                    micro_step=0,
                )
                self.add_edge(layer_key, xsa_key, kind="xsa_gate", magnitude=1.0, color=xsa_color, src_stage="layer", dst_stage="xsa")

            base_angle = 0.42 * (layer - 1)
            for head in range(q_heads):
                angle = base_angle + (2.0 * math.pi * head / q_heads)
                q_color = rgb_from_palette(self.palette, "attn_q", hue_shift=0.01 * ((layer + head) % 3), lightness_boost=0.02 * ((head % 2) - 0.5))
                q_key = f"layer_{layer:02d}_q_{head:02d}"
                q_group = head // 2
                self.add_point(
                    q_key,
                    x=x + float(self.layout["head_q_radius"]) * math.cos(angle),
                    y=y + 0.02 * head,
                    z=z + float(self.layout["head_q_radius"]) * math.sin(angle),
                    color=q_color,
                    label=f"L{layer:02d} Q{head:02d} | group={q_group} bits=5",
                    stage="attn_q",
                    role="attn_q",
                    component="attention_query",
                    layer_index=layer,
                    bitwidth=5,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    gqa_ratio=gqa_ratio,
                    rope_dims=16,
                    bigram_dim=0,
                    xsa_active=layer >= 2,
                    quant_name="attn=int5",
                    bit_index=head + 1,
                    bit_total=q_heads,
                    head_group=q_group,
                    head=head,
                    kv_head=q_group,
                    step=layer,
                    micro_step=head,
                )
                self.add_edge(layer_key, q_key, kind="attn_query", magnitude=5.0, color=q_color, src_stage="layer", dst_stage="attn_q", head=head, kv_head=q_group, step=layer)

            for kv in range(kv_heads):
                angle = base_angle + (math.pi / q_heads) + (2.0 * math.pi * kv / kv_heads)
                kv_color = rgb_from_palette(self.palette, "attn_kv", hue_shift=0.005 * ((layer + kv) % 2), lightness_boost=0.02 * kv)
                kv_key = f"layer_{layer:02d}_kv_{kv:02d}"
                self.add_point(
                    kv_key,
                    x=x + float(self.layout["head_kv_radius"]) * math.cos(angle),
                    y=y + 0.015 * kv,
                    z=z + float(self.layout["head_kv_radius"]) * math.sin(angle),
                    color=kv_color,
                    label=f"L{layer:02d} KV{kv:02d} | group={kv} bits=5",
                    stage="attn_kv",
                    role="attn_kv",
                    component="attention_kv",
                    layer_index=layer,
                    bitwidth=5,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    gqa_ratio=gqa_ratio,
                    rope_dims=16,
                    bigram_dim=0,
                    xsa_active=layer >= 2,
                    quant_name="attn=int5",
                    bit_index=kv + 1,
                    bit_total=kv_heads,
                    head_group=kv,
                    head=-1,
                    kv_head=kv,
                    step=layer,
                    micro_step=kv,
                )
                self.add_edge(layer_key, kv_key, kind="attn_key_value", magnitude=5.0, color=kv_color, src_stage="layer", dst_stage="attn_kv", kv_head=kv, step=layer)

            for head in range(q_heads):
                kv = head // 2
                q_key = f"layer_{layer:02d}_q_{head:02d}"
                kv_key = f"layer_{layer:02d}_kv_{kv:02d}"
                self.add_edge(q_key, kv_key, kind="gqa_share", magnitude=2.0, src_stage="attn_q", dst_stage="attn_kv", head=head, kv_head=kv, qk_align=0.75, transfer=0.65, step=layer)

        # Connect each layer to the model-wide quantization bars to show the bit-width budget.
        for layer in range(1, layers + 1):
            layer_key = f"layer_{layer:02d}"
            for module_key, bits in [("attn", 5), ("mlp", 6), ("aux", 6), ("other", 8)]:
                self.add_edge(layer_key, f"{module_key}_anchor", kind=f"quant_{module_key}", magnitude=float(bits), src_stage="layer", dst_stage=module_key, step=layer)
        self.add_edge(f"layer_{layers:02d}", "artifact", kind="artifact_pack", magnitude=1.0, src_stage="layer", dst_stage="artifact", step=layers)

    def build_artifact_and_metric(self) -> None:
        artifact_color = rgb_from_palette(self.palette, "artifact")
        self.add_point(
            "artifact",
            x=0.0,
            y=float(self.layout["artifact_y"]),
            z=0.0,
            color=artifact_color,
            label=f"Artifact 15.63MB | bytes_code={self.facts['bytes_code']} | bytes_total={self.facts['bytes_total']}",
            stage="compression",
            role="artifact",
            component="artifact",
            layer_index=12,
            bitwidth=0,
            q_heads=0,
            kv_heads=0,
            gqa_ratio=0.0,
            rope_dims=0,
            bigram_dim=0,
            xsa_active=False,
            quant_name="mixed+brotli",
            bit_index=0,
            bit_total=0,
            head_group=0,
            step=12,
            micro_step=0,
            bytes_code=self.facts["bytes_code"],
            bytes_total=self.facts["bytes_total"],
        )
        metric_color = rgb_from_palette(self.palette, "other", hue_shift=0.05)
        self.add_point(
            "metric",
            x=4.0,
            y=float(self.layout["artifact_y"]) - 0.25,
            z=1.8,
            color=metric_color,
            label="Sliding-window exact val_bpb=1.10597186",
            stage="metric",
            role="metric",
            component="score",
            layer_index=12,
            bitwidth=0,
            q_heads=0,
            kv_heads=0,
            gqa_ratio=0.0,
            rope_dims=0,
            bigram_dim=0,
            xsa_active=False,
            quant_name="score",
            bit_index=0,
            bit_total=0,
            head_group=0,
            step=12,
            micro_step=0,
            val_bpb=1.10597186,
        )
        self.add_edge("artifact", "metric", kind="score_line", magnitude=1.0, color=metric_color, src_stage="compression", dst_stage="metric")

    def build_points(self) -> None:
        self.build_bigram_and_embed()
        self.build_rope_ring()
        self.build_global_quant_bars()
        self.build_artifact_and_metric()
        self.build_layer_stack()

    def build_flow(self) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        node_index: dict[str, int] = {}
        flow_edges: list[dict[str, Any]] = []

        def add_node(key: str, **attrs: Any) -> int:
            if key in node_index:
                return node_index[key]
            idx = len(nodes)
            node_index[key] = idx
            nodes.append({"id": key, **attrs})
            return idx

        for key, idx in self.flow_nodes.items():
            p = self.points[idx]
            add_node(
                key,
                stage=p.get("stage", ""),
                role=p.get("role", p.get("stage", "")),
                component=p.get("component", ""),
                layer_index=p.get("layer_index", 0),
                bitwidth=p.get("bitwidth", 0),
                q_heads=p.get("q_heads", 0),
                kv_heads=p.get("kv_heads", 0),
                xsa_active=bool(p.get("xsa_active", False)),
                label=p.get("label", key),
            )

        for edge in self.edges:
            src = self.points[edge["i"]]
            dst = self.points[edge["j"]]
            flow_edges.append(
                {
                    "src": edge["src_key"],
                    "dst": edge["dst_key"],
                    "kind": edge["flow_kind"],
                    "count": 1,
                    "magnitude_sum": float(edge["magnitude"]),
                    "magnitude_avg": float(edge["magnitude"]),
                    "src_stage": src.get("stage", ""),
                    "dst_stage": dst.get("stage", ""),
                }
            )

        return {
            "meta": {
                "source": self.spec.get("model_name", "Midnight 12L"),
                "model_name": self.spec.get("model_name", "Midnight 12L"),
                "architecture": self.spec.get("architecture", ""),
                "summary": self.spec.get("summary", ""),
                "tags": self.spec.get("tags", []),
                "facts": self.facts,
                "nodes": len(nodes),
                "edges": len(flow_edges),
            },
            "nodes": nodes,
            "edges": sorted(flow_edges, key=lambda e: (-float(e["magnitude_sum"]), e["src"], e["dst"])),
        }

    def build_points_payload(self) -> dict[str, Any]:
        return {
            "meta": {
                "source": "midnight_12l_spec.json",
                "model_name": self.spec.get("model_name", "Midnight 12L"),
                "architecture": self.spec.get("architecture", ""),
                "architecture_summary": self.spec.get("summary", ""),
                "architecture_tags": self.spec.get("tags", []),
                "facts": self.facts,
                "visual_language": "glass tower / orbital heads / quant barcode",
                "layout": self.layout,
            },
            "points": self.points,
            "edges": self.edges,
        }


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())
    builder = Builder(spec)
    builder.build_points()

    points_payload = builder.build_points_payload()
    flow_payload = builder.build_flow()

    points_path = Path(args.output_points)
    flow_path = Path(args.output_flow)
    points_path.parent.mkdir(parents=True, exist_ok=True)
    flow_path.parent.mkdir(parents=True, exist_ok=True)
    points_path.write_text(json.dumps(points_payload, indent=2), encoding="utf-8")
    flow_path.write_text(json.dumps(flow_payload, indent=2), encoding="utf-8")

    print(f"wrote points to {points_path}")
    print(f"wrote flow to {flow_path}")
    print(f"points={len(points_payload['points'])} edges={len(points_payload['edges'])} flow_nodes={flow_payload['meta']['nodes']} flow_edges={flow_payload['meta']['edges']}")


if __name__ == "__main__":
    main()
