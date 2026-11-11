# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode a canonical IQ2R-v2 checkpoint back into a block-FP8 checkpoint.

Reference serving for quality checks before (or without) native kernels: each
routed expert is decoded with the reference decoder ``iq2r_v2_materialize``,
its interleaved gate/up rows are split, and every matrix is re-quantized to the
source's 128x128 block FP8 (E4M3 values, FP32 ``weight_scale_inv``). The output
has the source checkpoint's tensor names and quantization_config, so any FP8
runtime serves the 2-bit experts through its ordinary FP8 MoE. The
re-quantization error is reported per layer; it is small next to the 2-bit
error.

Input is a ``glm5next-canonical-v2`` checkpoint written by
``aiter.quantization.iq2r.glm5.compile``. Its other tensors are hard-linked
(or copied) into the output, so put both on one filesystem:

    # one process per GPU, each on its own MoE layers
    HIP_VISIBLE_DEVICES=$i python3 -m aiter.quantization.iq2r.glm5.decode_fp8 \\
        --iq2r-dir $IQ2R --output-dir $OUT --layers 3-8
    # then one run without --layers writes the index and config
    python3 -m aiter.quantization.iq2r.glm5.decode_fp8 --iq2r-dir $IQ2R \\
        --output-dir $OUT
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from aiter.ops.iq2r_v2 import IQ2R_V2_LEVELS_C4, iq2r_v2_materialize

from .compile import (
    _MODEL_FILES,
    _PROJECTIONS,
    CANONICAL_LAYOUT,
    FORMAT,
    _atomic_write_json,
    _layer_file,
    _parse_layers,
    _read_json,
    glm5_source_layout,
    iq2r_compiled_tensor_keys,
    iq2r_glm5_source_keys,
)

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_DEVICE = "cuda:0"


def quantize_block_fp8(weight: Tensor, block: int = 128) -> tuple[Tensor, Tensor]:
    """``weight [N, K]`` -> E4M3 values and FP32 ``scale_inv [N/block, K/block]``."""

    n, k = weight.shape
    if n % block or k % block:
        raise ValueError(f"{tuple(weight.shape)} is not a multiple of {block}")
    blocks = weight.float().reshape(n // block, block, k // block, block)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / _FP8_MAX
    values = (blocks / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return values.reshape(n, k), scale.reshape(n // block, k // block).contiguous()


def _dequantize(values: Tensor, scale_inv: Tensor, block: int = 128) -> Tensor:
    n, k = values.shape
    return (
        values.float().reshape(n // block, block, k // block, block)
        * scale_inv[:, None, :, None]
    ).reshape(n, k)


def _decode_layer(iq2r_dir: Path, layout, layer: int, device) -> tuple[dict, dict]:
    """FP8 tensors of one layer's routed experts and the re-quantization error."""

    shapes = {
        "gate_up": (2 * layout.intermediate_size, layout.hidden_size),
        "down": (layout.hidden_size, layout.intermediate_size),
    }
    tensors: dict[str, Tensor] = {}
    error = {name: [0.0, 0.0] for name in ("gate_proj", "up_proj", "down_proj")}
    for projection, (n, k) in shapes.items():
        keys = iq2r_compiled_tensor_keys(
            layer, projection, CANONICAL_LAYOUT, layout.prefix
        )
        with safe_open(iq2r_dir / _layer_file(layer, projection), "pt") as handle:
            data = handle.get_tensor(keys["data"]).to(device)
            auxiliary = handle.get_tensor(keys["auxiliary"]).to(device)
        for expert in range(layout.expert_count):
            decoded = iq2r_v2_materialize(
                data[expert], auxiliary[expert], n, k, IQ2R_V2_LEVELS_C4
            )
            names = iq2r_glm5_source_keys(layer, expert, layout.prefix)
            # compile interleaves gate/up rows as gate0, up0, gate1, ...
            parts = (
                {"gate_proj": decoded[0::2], "up_proj": decoded[1::2]}
                if projection == "gate_up"
                else {"down_proj": decoded}
            )
            for name, matrix in parts.items():
                values, scale_inv = quantize_block_fp8(
                    matrix.contiguous(), layout.block_n
                )
                residual = _dequantize(values, scale_inv, layout.block_n) - matrix
                error[name][0] += float(residual.square().sum())
                error[name][1] += float(matrix.square().sum())
                tensors[names[f"{name}_weight"]] = values.cpu()
                tensors[names[f"{name}_weight_scale_inv"]] = scale_inv.cpu()
    relative = {
        name: (num / max(den, 1e-30)) ** 0.5 for name, (num, den) in error.items()
    }
    return tensors, relative


def _fp8_layer_file(layer: int) -> str:
    return f"experts-fp8-layer-{layer:04d}.safetensors"


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def decode_iq2r_to_fp8(
    iq2r_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    layer_indices: list[int] | None = None,
) -> Path | None:
    """Write block-FP8 experts for the selected layers; a full run also writes
    the index, config and the remaining files and returns the config path."""

    iq2r_dir = Path(iq2r_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == iq2r_dir:
        raise ValueError("the output dir must differ from the IQ2R checkpoint")
    config = _read_json(iq2r_dir / "config.json")
    quantization = config.get("quantization_config") or {}
    if (
        quantization.get("quant_method") != "iq2r"
        or quantization.get("iq2r_layout") != CANONICAL_LAYOUT
        or quantization.get("iq2r_format") != FORMAT
    ):
        raise ValueError(
            f"{iq2r_dir} is not a {CANONICAL_LAYOUT} {FORMAT} IQ2R checkpoint"
        )
    source_config = {
        **config,
        "quantization_config": quantization["base_quantization_config"],
    }
    layout = glm5_source_layout(source_config)
    moe_layers = list(range(layout.first_moe_layer, layout.layer_count))
    selected = moe_layers if layer_indices is None else sorted(set(layer_indices))
    if not selected or any(layer not in moe_layers for layer in selected):
        raise ValueError(f"layers must be inside {moe_layers[0]}..{moe_layers[-1]}")

    device = torch.device(_DEVICE)
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer in selected:
        path = output_dir / _fp8_layer_file(layer)
        if path.exists():
            continue
        tensors, relative = _decode_layer(iq2r_dir, layout, layer, device)
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_file(tensors, temporary, metadata={"format": "pt"})
        os.replace(temporary, path)
        print(
            json.dumps(
                {"layer": layer, "fp8_requant_relative_rmse": relative},
                sort_keys=True,
            ),
            flush=True,
        )
    if selected != moe_layers:
        return None

    index = _read_json(iq2r_dir / "model.safetensors.index.json")
    iq2r_keys = {
        key
        for layer in moe_layers
        for projection in _PROJECTIONS
        for key in iq2r_compiled_tensor_keys(
            layer, projection, CANONICAL_LAYOUT, layout.prefix
        ).values()
    }
    weight_map = {
        name: shard
        for name, shard in index["weight_map"].items()
        if name not in iq2r_keys
    }
    for shard in set(weight_map.values()):
        _link_or_copy(iq2r_dir / shard, output_dir / shard)
    for layer in moe_layers:
        shard = _fp8_layer_file(layer)
        with safe_open(output_dir / shard, "pt") as handle:
            weight_map.update(dict.fromkeys(handle.keys(), shard))
    _atomic_write_json(
        output_dir / "model.safetensors.index.json",
        {"metadata": index.get("metadata", {}), "weight_map": weight_map},
    )
    config_path = output_dir / "config.json"
    _atomic_write_json(config_path, source_config)
    for name in _MODEL_FILES:
        if (iq2r_dir / name).is_file():
            shutil.copyfile(iq2r_dir / name, output_dir / name)
    return config_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--iq2r-dir",
        type=Path,
        required=True,
        help=f"{CANONICAL_LAYOUT} checkpoint written by compile",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--layers",
        type=_parse_layers,
        help="MoE layers to decode, e.g. 3-8; without it every layer is "
        "decoded (or kept) and the index and config are written",
    )
    args = parser.parse_args(argv)
    config_path = decode_iq2r_to_fp8(
        args.iq2r_dir, args.output_dir, layer_indices=args.layers
    )
    print(json.dumps({"config": config_path and str(config_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
