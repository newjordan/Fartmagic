#!/usr/bin/env python3
"""Quantization calibration sweep — load a float checkpoint, test many quant configs, report roundtrip BPB.

Usage:
    python quant_sweep.py [--checkpoint final_model.pt] [--script neural/experiments/Lucky_V/train_gpt.py]

Runs on a SINGLE GPU, no torchrun needed. Each config takes ~15-20s.
"""

import os, sys, io, time, math, argparse, importlib.util, types
import torch
import torch.nn.functional as F

def load_train_module(script_path):
    """Import train_gpt.py as a module without running main()."""
    spec = importlib.util.spec_from_file_location("train_gpt", script_path)
    mod = importlib.util.module_from_spec(spec)
    # Prevent distributed init issues
    os.environ.pop("RANK", None)
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("LOCAL_RANK", None)
    spec.loader.exec_module(mod)
    return mod


def quantize_sweep_config(sd_cpu, config, tgt):
    """Quantize a state dict with a specific config and return compressed blob + meta."""
    clip_range = config.get("clip_range", 31)
    int6_cats = config.get("int6_cats", {"mlp", "attn", "aux", "embed"})
    byte_shuffle = config.get("byte_shuffle", True)
    byte_shuffle_stride = config.get("byte_shuffle_stride", 2)
    extra_pcts = config.get("extra_pcts", None)

    result = {}
    meta = {}
    for name, tensor in sd_cpu.items():
        t = tensor.detach().cpu().contiguous()
        cat = tgt._classify_param(name)

        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue

        if any(p in name for p in tgt.CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue

        if cat in int6_cats and t.ndim == 2:
            q, s = quantize_int6_configurable(t, clip_range=clip_range, extra_pcts=extra_pcts)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        elif cat in int6_cats and t.ndim >= 1:
            t_2d = t.reshape(-1, t.shape[-1]) if t.ndim > 2 else t
            q, s = quantize_int6_configurable(t_2d, clip_range=clip_range, extra_pcts=extra_pcts)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            t_q = t.reshape(-1, t.shape[-1]) if t.ndim > 2 else t
            q, s = tgt.quantize_float_tensor(t_q)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}

    # Compress
    import brotli
    quant_buf = io.BytesIO()
    torch.save({"w": result, "m": meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    if byte_shuffle:
        quant_raw = tgt._byte_shuffle(quant_raw, byte_shuffle_stride)
    quant_blob = brotli.compress(quant_raw, quality=11)
    return quant_blob, meta


def quantize_int6_configurable(t, clip_range=31, extra_pcts=None):
    """Like quantize_int6_per_row but with configurable clip_range and percentiles."""
    t32 = t.float()
    pcts = [0.9990, 0.9995, 0.9999, 0.99999, 1.0]
    if extra_pcts:
        pcts = sorted(set(pcts + extra_pcts))

    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in pcts:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def decompress_and_dequantize(quant_blob, sd_cpu, tgt, byte_shuffle=True, byte_shuffle_stride=2):
    """Decompress and dequantize a blob back to a state dict."""
    import brotli
    raw = brotli.decompress(quant_blob)
    if byte_shuffle:
        raw = tgt._byte_unshuffle(raw)
    quant_state = torch.load(io.BytesIO(raw), map_location="cpu")
    return tgt.dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)


def eval_roundtrip(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, device, seq_len=2048):
    """Fast single-GPU eval — returns (val_loss, val_bpb)."""
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_size = 64  # sequences per batch
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, total_seqs, batch_size):
            batch_end = min(batch_start + batch_size, total_seqs)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            n_tok = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * n_tok
            val_token_count += n_tok

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    val_loss = (val_loss_sum / val_token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    return val_loss, bits_per_token * tokens_per_byte


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final_model.pt")
    parser.add_argument("--script", default="neural/experiments/Lucky_V/train_gpt.py")
    args = parser.parse_args()

    print(f"Loading module from {args.script}...")
    tgt = load_train_module(args.script)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    hp = tgt.Hyperparameters()

    # Load val data using the same loader as training
    print("Loading validation data...")
    val_tokens = tgt.load_validation_tokens(hp.val_files, hp.train_seq_len)
    print(f"Val tokens: {val_tokens.numel():,}")

    # Build BPB LUTs
    print("Building BPB lookup tables...")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tgt.build_sentencepiece_luts(sp, hp.vocab_size, device)

    # Load checkpoint
    print(f"Loading checkpoint {args.checkpoint}...")
    sd_cpu = torch.load(args.checkpoint, map_location="cpu")

    # Create model template
    print("Creating model...")
    model = tgt.GPT(
        vocab_size=hp.vocab_size, num_layers=hp.num_layers, model_dim=hp.model_dim,
        num_heads=hp.num_heads, num_kv_heads=hp.num_kv_heads, mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings, tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap, rope_base=hp.rope_base, qk_gain_init=hp.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=hp.bigram_vocab_size, bigram_dim=hp.bigram_dim,
        xsa_last_n=hp.xsa_last_n, rope_dims=hp.rope_dims, ln_scale=hp.ln_scale,
        dtg=hp.dtg_enabled, ve_enabled=hp.ve_enabled, ve_dim=hp.ve_dim, ve_layers=hp.ve_layers,
        gated_attention=hp.gated_attention, value_residual=hp.value_residual,
    ).to(device).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for m in model.modules():
        if isinstance(m, tgt.CastedLinear):
            m.float()
    tgt.restore_low_dim_params_to_fp32(model)

    # Float baseline: load original weights and eval
    print("\n=== FLOAT BASELINE ===")
    model.load_state_dict(sd_cpu, strict=True)
    t0 = time.perf_counter()
    float_loss, float_bpb = eval_roundtrip(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, device)
    print(f"float: val_loss={float_loss:.6f} val_bpb={float_bpb:.6f} time={time.perf_counter()-t0:.1f}s")

    # Define sweep configs
    configs = [
        {"name": "baseline_int6_cr31",
         "clip_range": 31, "int6_cats": {"mlp", "attn", "aux", "embed"},
         "byte_shuffle": True, "byte_shuffle_stride": 2},

        {"name": "embed_int8",
         "clip_range": 31, "int6_cats": {"mlp", "attn", "aux"},
         "byte_shuffle": True, "byte_shuffle_stride": 2},

        {"name": "attn_int8",
         "clip_range": 31, "int6_cats": {"mlp", "aux", "embed"},
         "byte_shuffle": True, "byte_shuffle_stride": 2},

        {"name": "mlp_int8",
         "clip_range": 31, "int6_cats": {"attn", "aux", "embed"},
         "byte_shuffle": True, "byte_shuffle_stride": 2},

        {"name": "int5_cr15",
         "clip_range": 15, "int6_cats": {"mlp", "attn", "aux", "embed"},
         "byte_shuffle": True, "byte_shuffle_stride": 2},

        {"name": "int7_cr63",
         "clip_range": 63, "int6_cats": {"mlp", "attn", "aux", "embed"},
         "byte_shuffle": True, "byte_shuffle_stride": 2},

        {"name": "cr25",
         "clip_range": 25, "int6_cats": {"mlp", "attn", "aux", "embed"},
         "byte_shuffle": True, "byte_shuffle_stride": 2},

        {"name": "cr40",
         "clip_range": 40, "int6_cats": {"mlp", "attn", "aux", "embed"},
         "byte_shuffle": True, "byte_shuffle_stride": 2},

        {"name": "shuffle_s1",
         "clip_range": 31, "int6_cats": {"mlp", "attn", "aux", "embed"},
         "byte_shuffle": True, "byte_shuffle_stride": 1},

        {"name": "shuffle_s4",
         "clip_range": 31, "int6_cats": {"mlp", "attn", "aux", "embed"},
         "byte_shuffle": True, "byte_shuffle_stride": 4},

        {"name": "no_shuffle",
         "clip_range": 31, "int6_cats": {"mlp", "attn", "aux", "embed"},
         "byte_shuffle": False, "byte_shuffle_stride": 2},

        {"name": "more_pcts",
         "clip_range": 31, "int6_cats": {"mlp", "attn", "aux", "embed"},
         "byte_shuffle": True, "byte_shuffle_stride": 2,
         "extra_pcts": [0.99, 0.995, 0.998, 0.9992, 0.9997, 0.99995]},

        {"name": "mlp_attn_int8_embed_int6",
         "clip_range": 31, "int6_cats": {"embed", "aux"},
         "byte_shuffle": True, "byte_shuffle_stride": 2},
    ]

    print(f"\n=== QUANTIZATION SWEEP: {len(configs)} configs ===")
    print(f"{'name':<30} {'val_bpb':>10} {'slide':>8} {'size_MB':>10} {'time':>6}")
    print("-" * 70)

    results = []
    for cfg in configs:
        name = cfg["name"]
        t0 = time.perf_counter()

        # Quantize + compress
        quant_blob, meta = quantize_sweep_config(sd_cpu, cfg, tgt)
        blob_size = len(quant_blob)

        # Decompress + dequantize
        deq = decompress_and_dequantize(
            quant_blob, sd_cpu, tgt,
            byte_shuffle=cfg.get("byte_shuffle", True),
            byte_shuffle_stride=cfg.get("byte_shuffle_stride", 2),
        )

        # Load and eval
        model.load_state_dict(deq, strict=True)
        val_loss, val_bpb = eval_roundtrip(
            model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, device,
        )
        elapsed = time.perf_counter() - t0
        slide = val_bpb - float_bpb
        total_size = blob_size + len(open(args.script, "rb").read())

        results.append({
            "name": name, "val_loss": val_loss, "val_bpb": val_bpb,
            "slide": slide, "blob_bytes": blob_size, "total_bytes": total_size,
            "time": elapsed,
        })
        print(f"{name:<30} {val_bpb:>10.6f} {slide:>+8.5f} {total_size/1e6:>10.3f} {elapsed:>5.1f}s")

    # Summary sorted by BPB
    print(f"\n=== RANKED BY ROUNDTRIP BPB ===")
    print(f"{'rank':<5} {'name':<30} {'val_bpb':>10} {'slide':>8} {'size_MB':>10}")
    print("-" * 70)
    for i, r in enumerate(sorted(results, key=lambda x: x["val_bpb"])):
        flag = " *" if r["total_bytes"] > 16_000_000 else ""
        print(f"{i+1:<5} {r['name']:<30} {r['val_bpb']:>10.6f} {r['slide']:>+8.5f} {r['total_bytes']/1e6:>10.3f}{flag}")

    print(f"\nFloat baseline: val_bpb={float_bpb:.6f}")
    print(f"* = over 16MB budget")


if __name__ == "__main__":
    main()
