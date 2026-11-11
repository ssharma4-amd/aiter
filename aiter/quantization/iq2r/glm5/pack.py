# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pack a canonical GLM-5.3-Flash IQ2R-v2 checkpoint for the serving kernels.

``aiter.quantization.iq2r.glm5.compile`` writes GLM-5.3-Flash's routed experts
in the canonical IQ2R-v2 layout (``glm5next-canonical-v2``). This tool packs
each layer with ``aiter.iq2r_glm53.iq2r_glm53_pack_v2`` for the GLM5_NEXT
kernel shape and writes a ``glm5next-packed-v2`` checkpoint with the tensor
names of GLM-5.3's packed checkpoints. The other tensors are hard-linked (or
copied), so put both directories on one filesystem:

    python3 -m aiter.quantization.iq2r.glm5.pack --iq2r-dir $CANONICAL \\
        --output-dir $PACKED
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

from aiter.iq2r_glm53 import GLM5_NEXT, iq2r_glm53_pack_v2

from .compile import (
    _MODEL_FILES,
    _PROJECTIONS,
    CANONICAL_LAYOUT,
    FORMAT,
    _atomic_write_json,
    _layer_file,
    _read_json,
    glm5_source_layout,
    iq2r_compiled_tensor_keys,
)
from .decode_fp8 import _link_or_copy

PACKED_LAYOUT = "glm5next-packed-v2"
_DEVICE = "cuda:0"


def pack_glm5_next(
    iq2r_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]
) -> Path:
    """Write the packed checkpoint; returns its config path."""

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
    layout = glm5_source_layout(
        {**config, "quantization_config": quantization["base_quantization_config"]}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index = _read_json(iq2r_dir / "model.safetensors.index.json")
    weight_map = dict(index["weight_map"])
    device = torch.device(_DEVICE)
    for layer in range(layout.first_moe_layer, layout.layer_count):
        canonical = []
        metadata = {}
        for projection in _PROJECTIONS:
            keys = iq2r_compiled_tensor_keys(
                layer, projection, CANONICAL_LAYOUT, layout.prefix
            )
            with safe_open(iq2r_dir / _layer_file(layer, projection), "pt") as handle:
                canonical += [
                    handle.get_tensor(keys["data"]).to(device),
                    handle.get_tensor(keys["auxiliary"]).to(device),
                ]
                metadata[projection] = handle.metadata()
            for key in keys.values():
                del weight_map[key]
        packed = iq2r_glm53_pack_v2(
            *canonical, intermediate_size=layout.intermediate_size, shape=GLM5_NEXT
        )
        for projection, data, auxiliary in zip(_PROJECTIONS, packed[::2], packed[1::2]):
            keys = iq2r_compiled_tensor_keys(layer, projection, prefix=layout.prefix)
            path = output_dir / _layer_file(layer, projection)
            temporary = path.with_suffix(path.suffix + ".tmp")
            save_file(
                {
                    keys["data"]: data.cpu().contiguous(),
                    keys["auxiliary"]: auxiliary.cpu().contiguous(),
                    keys["tile_n"]: torch.tensor([128], dtype=torch.int32),
                },
                temporary,
                metadata={**metadata[projection], "iq2r_layout": PACKED_LAYOUT},
            )
            os.replace(temporary, path)
            weight_map.update(dict.fromkeys(keys.values(), path.name))
        print(json.dumps({"layer": layer, "packed": True}), flush=True)

    layer_files = {
        _layer_file(layer, projection)
        for layer in range(layout.first_moe_layer, layout.layer_count)
        for projection in _PROJECTIONS
    }
    for shard in set(weight_map.values()) - layer_files:
        _link_or_copy(iq2r_dir / shard, output_dir / shard)
    _atomic_write_json(
        output_dir / "model.safetensors.index.json",
        {"metadata": index.get("metadata", {}), "weight_map": weight_map},
    )
    config_path = output_dir / "config.json"
    _atomic_write_json(
        config_path,
        {
            **config,
            "quantization_config": {**quantization, "iq2r_layout": PACKED_LAYOUT},
        },
    )
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
    args = parser.parse_args(argv)
    print(json.dumps({"config": str(pack_glm5_next(args.iq2r_dir, args.output_dir))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
