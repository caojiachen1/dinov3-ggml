#!/usr/bin/env python3
"""
Convert PyTorch DINOv2/v3 ViT weights to raw binary format for ggml_vit.

Usage:
    python convert_weights.py --input model.pth --output weights.bin [--model vit_small_16]

The output binary format is a sequence of f32 arrays:
    patch_embed_weight, patch_embed_bias, cls_token, register_tokens,
    then for each layer: norm1_w, norm1_b, qkv_w, qkv_b, proj_w, proj_b,
                         norm2_w, norm2_b, fc1_w, fc1_b, fc2_w, fc2_b,
                         ls1_gamma, ls2_gamma
    then: norm_w, norm_b
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)


# Model configurations
CONFIGS = {
    "vit_small_16": {
        "hidden_size": 384,
        "num_layers": 12,
        "num_heads": 6,
        "intermediate_size": 1536,
        "patch_size": 16,
        "num_register_tokens": 4,
    },
    "vit_base_16": {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "patch_size": 16,
        "num_register_tokens": 4,
    },
    "vit_large_16": {
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "intermediate_size": 4096,
        "patch_size": 16,
        "num_register_tokens": 4,
    },
    "vit_huge_16": {
        "hidden_size": 1280,
        "num_layers": 32,
        "num_heads": 16,
        "intermediate_size": 5120,
        "patch_size": 16,
        "num_register_tokens": 4,
    },
}


def write_f32_array(f, arr: np.ndarray):
    """Write a numpy array as raw f32 bytes."""
    data = arr.astype(np.float32).tobytes()
    f.write(data)
    return len(data)


def convert_dinov2_weights(state_dict: dict, config: dict, output_path: str):
    """Convert DINOv2 state dict to raw binary format."""
    H = config["hidden_size"]
    n_layers = config["num_layers"]
    n_reg = config["num_register_tokens"]

    total_bytes = 0

    with open(output_path, "wb") as f:
        # Patch embedding
        pe_w = state_dict.get("patch_embed.proj.weight")
        if pe_w is None:
            # Try alternative key names
            pe_w = state_dict.get("module.patch_embed.proj.weight")
        pe_w = pe_w.detach().cpu().numpy()
        total_bytes += write_f32_array(f, pe_w)
        print(f"  patch_embed_weight: {pe_w.shape}")

        pe_b = state_dict.get("patch_embed.proj.bias")
        if pe_b is None:
            pe_b = state_dict.get("module.patch_embed.proj.bias")
        pe_b = pe_b.detach().cpu().numpy()
        total_bytes += write_f32_array(f, pe_b)
        print(f"  patch_embed_bias: {pe_b.shape}")

        # CLS token
        cls = state_dict.get("cls_token")
        if cls is None:
            cls = state_dict.get("module.cls_token")
        cls = cls.detach().cpu().numpy().flatten()
        total_bytes += write_f32_array(f, cls)
        print(f"  cls_token: {cls.shape}")

        # Register tokens
        reg = state_dict.get("reg_token")
        if reg is None:
            reg = state_dict.get("register_tokens")
        if reg is None:
            reg = state_dict.get("module.reg_token")
        if reg is None:
            reg = state_dict.get("module.register_tokens")
        if reg is not None:
            reg = reg.detach().cpu().numpy().flatten()
            total_bytes += write_f32_array(f, reg)
            print(f"  register_tokens: {reg.shape}")
        else:
            # Write zeros if no register tokens
            reg = np.zeros(n_reg * H, dtype=np.float32)
            total_bytes += write_f32_array(f, reg)
            print(f"  register_tokens: zeros ({reg.shape})")

        # Per-layer weights
        for l in range(n_layers):
            prefix = f"blocks.{l}."
            # Try alternative prefix
            if prefix + "norm1.weight" not in state_dict:
                prefix = f"module.blocks.{l}."

            # norm1
            w = state_dict[prefix + "norm1.weight"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, w)
            b = state_dict[prefix + "norm1.bias"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, b)

            # QKV (fused)
            qkv_w = state_dict[prefix + "attn.qkv.weight"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, qkv_w)
            qkv_b = state_dict[prefix + "attn.qkv.bias"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, qkv_b)

            # Projection
            proj_w = state_dict[prefix + "attn.proj.weight"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, proj_w)
            proj_b = state_dict[prefix + "attn.proj.bias"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, proj_b)

            # norm2
            w = state_dict[prefix + "norm2.weight"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, w)
            b = state_dict[prefix + "norm2.bias"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, b)

            # MLP fc1
            fc1_w = state_dict[prefix + "mlp.fc1.weight"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, fc1_w)
            fc1_b = state_dict[prefix + "mlp.fc1.bias"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, fc1_b)

            # MLP fc2
            fc2_w = state_dict[prefix + "mlp.fc2.weight"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, fc2_w)
            fc2_b = state_dict[prefix + "mlp.fc2.bias"].detach().cpu().numpy()
            total_bytes += write_f32_array(f, fc2_b)

            # LayerScale gammas
            ls1_key = prefix + "ls1.gamma"
            ls2_key = prefix + "ls2.gamma"
            if ls1_key in state_dict:
                ls1 = state_dict[ls1_key].detach().cpu().numpy()
                ls2 = state_dict[ls2_key].detach().cpu().numpy()
            else:
                # Default LayerScale: all ones
                ls1 = np.ones(H, dtype=np.float32)
                ls2 = np.ones(H, dtype=np.float32)
            total_bytes += write_f32_array(f, ls1)
            total_bytes += write_f32_array(f, ls2)

            if (l + 1) % 4 == 0 or l == n_layers - 1:
                print(f"  Layer {l+1}/{n_layers} done")

        # Final norm
        norm_w = state_dict.get("norm.weight")
        if norm_w is None:
            norm_w = state_dict.get("module.norm.weight")
        norm_w = norm_w.detach().cpu().numpy()
        total_bytes += write_f32_array(f, norm_w)

        norm_b = state_dict.get("norm.bias")
        if norm_b is None:
            norm_b = state_dict.get("module.norm.bias")
        norm_b = norm_b.detach().cpu().numpy()
        total_bytes += write_f32_array(f, norm_b)

    print(f"\nTotal: {total_bytes / 1024 / 1024:.1f} MB written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert DINOv2/v3 weights to GGML binary format")
    parser.add_argument("--input", "-i", required=True, help="Path to PyTorch model (.pth)")
    parser.add_argument("--output", "-o", required=True, help="Output binary file path")
    parser.add_argument("--model", "-m", default="vit_small_16", choices=CONFIGS.keys(),
                        help="Model architecture")
    args = parser.parse_args()

    config = CONFIGS[args.model]
    print(f"Model: {args.model}")
    print(f"Config: {config}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    # Load PyTorch model
    print("\nLoading PyTorch model...")
    state_dict = torch.load(args.input, map_location="cpu", weights_only=True)

    # Handle wrapped state dicts
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    print(f"State dict has {len(state_dict)} keys")
    print(f"Sample keys: {list(state_dict.keys())[:5]}")

    # Convert
    print("\nConverting weights...")
    convert_dinov2_weights(state_dict, config, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
