# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build a GLM-5.3 IQ2R model directory around compiled expert shards.

``aiter.iq2r_glm5_compile`` writes only the routed experts (and the fused
shared expert) as ``iq2r-layer-NNNN-{gate-up,down}.safetensors``. This tool
symlinks those shards and the base FP8 shards into one directory, writes an
index in which the compiled tensors replace the FP8 expert tensors, and
sets ``quant_method: "iq2r"`` in config.json.
``aiter.iq2r_glm53_pack_checkpoint`` turns the result into the packed
checkpoint ATOM serves.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .iq2r_glm5_compile import (
    GLM5Layout,
    _atomic_write_json,
    _projection_metadata,
    _read_json,
    _require_safetensors,
    _validate_projection_shard,
    glm5_source_layout,
    iq2r_compiled_tensor_keys,
    iq2r_glm5_shared_source_keys,
    iq2r_glm5_source_keys,
)
from .ops.iq2r_format import (
    IQ2R_ACTIVATION_BASIS,
    IQ2R_FORMAT_NAME,
)

_SHARD_SUFFIXES = {"gate_up": "gate-up", "down": "down"}


def _replace_with_symlink(source: Path, destination: Path, *, force: bool) -> None:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        if not force:
            raise FileExistsError(f"refusing to replace {destination}; pass --force")
        destination.unlink()
    destination.symlink_to(source.resolve())


def _link_base_model_files(
    model_dir: Path,
    output_dir: Path,
    source_weight_map: dict[str, str],
    *,
    force: bool,
) -> list[str]:
    source_shards = sorted(set(source_weight_map.values()))
    for shard in source_shards:
        source = model_dir / shard
        if not source.is_file():
            raise FileNotFoundError(f"source checkpoint shard is missing: {source}")
        _replace_with_symlink(source, output_dir / Path(shard).name, force=force)
    for source in model_dir.iterdir():
        if (
            not source.is_file()
            or source.name == "config.json"
            or source.name == "model.safetensors.index.json"
            or source.suffix == ".safetensors"
        ):
            continue
        _replace_with_symlink(source, output_dir / source.name, force=force)
    return source_shards


def _compiled_contract(compiled_dir: Path, layout: GLM5Layout) -> tuple[bool, str]:
    """Check the compiled config and return (fused_shared_expert, quality)."""

    config = _read_json(compiled_dir / "config.json")
    iq2r = config.get("iq2r")
    if not isinstance(iq2r, dict):
        raise TypeError("compiled config does not contain an IQ2R format declaration")
    identity = (iq2r.get("format"), iq2r.get("activation_basis"))
    expected = (IQ2R_FORMAT_NAME, IQ2R_ACTIVATION_BASIS)
    if identity != expected:
        raise ValueError(
            f"unsupported IQ2R checkpoint identity {identity!r}; expected {expected!r}"
        )
    if (
        config.get("compiled_tensor_parallel_size"),
        config.get("compiled_expert_parallel_size"),
    ) != (1, 1):
        raise ValueError("compiled IQ2R checkpoint must be TP1/EP1")
    if iq2r.get("model_family") != layout.model_family:
        raise ValueError(
            f"compiled IQ2R model family {iq2r.get('model_family')!r} does not "
            f"match the source {layout.model_family!r}"
        )
    moe_layers = list(range(layout.first_moe_layer, layout.layer_count))
    if iq2r.get("compiled_layers") != moe_layers:
        raise ValueError(
            "compiled IQ2R config does not cover MoE layers "
            f"{moe_layers[0]}-{moe_layers[-1]}; rerun aiter.iq2r_glm5_compile "
            "over all of them with --resume"
        )
    fused = iq2r.get("fused_shared_expert")
    if not isinstance(fused, bool):
        raise TypeError("compiled IQ2R config has no boolean fused_shared_expert")
    if fused and layout.shared_expert_count != 1:
        raise ValueError(
            "compiled IQ2R checkpoint fuses one shared expert, but the source "
            f"declares {layout.shared_expert_count}"
        )
    if iq2r.get("compiled_expert_count") != layout.expert_count + int(fused):
        raise ValueError(
            "compiled IQ2R expert count "
            f"{iq2r.get('compiled_expert_count')!r} does not match the source"
        )
    quality = iq2r.get("quality")
    if not isinstance(quality, str):
        raise TypeError("compiled IQ2R config has no quality string")
    return fused, quality


def create_glm5_iq2r_overlay(
    model_dir: str | os.PathLike[str],
    compiled_iq2r_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    force: bool = False,
) -> Path:
    """Create a GLM-5.3 IQ2R overlay and return its manifest path."""

    safe_open, _ = _require_safetensors()
    model_dir = Path(model_dir).resolve()
    compiled_iq2r_dir = Path(compiled_iq2r_dir).resolve()
    output_dir = Path(output_dir).resolve()

    index_path = model_dir / "model.safetensors.index.json"
    config = _read_json(model_dir / "config.json")
    layout = glm5_source_layout(config)
    base_quantization_config = config.get("quantization_config")
    if not isinstance(base_quantization_config, dict):
        raise TypeError("GLM-5 source config has no base quantization_config")
    source_index = _read_json(index_path)
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in source_weight_map.items()
    ):
        raise ValueError(f"{index_path} does not contain a string weight_map object")

    # Validate every compiled shard before creating any output.
    fused, quality = _compiled_contract(compiled_iq2r_dir, layout)
    moe_layers = range(layout.first_moe_layer, layout.layer_count)
    for layer in moe_layers:
        for projection, suffix in _SHARD_SUFFIXES.items():
            _validate_projection_shard(
                compiled_iq2r_dir / f"iq2r-layer-{layer:04d}-{suffix}.safetensors",
                layout,
                layer,
                projection,
                quality,
                safe_open,
                fuse_shared_expert=fused,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_shards = _link_base_model_files(
        model_dir, output_dir, source_weight_map, force=force
    )
    weight_map = dict(source_weight_map)
    removed_source_tensor_count = 0
    layer_records: list[dict[str, Any]] = []
    total_iq2r_bytes = 0
    for layer in moe_layers:
        source_keys = [
            iq2r_glm5_source_keys(layer, expert, root=layout.source_root)
            for expert in range(layout.expert_count)
        ]
        if fused:
            source_keys.append(
                iq2r_glm5_shared_source_keys(layer, root=layout.source_root)
            )
        for keys in source_keys:
            for name in keys.values():
                if name not in weight_map:
                    raise KeyError(f"source index is missing {name!r}")
                del weight_map[name]
                removed_source_tensor_count += 1

        shard_names = []
        for projection, suffix in _SHARD_SUFFIXES.items():
            shard = f"iq2r-layer-{layer:04d}-{suffix}.safetensors"
            _replace_with_symlink(
                compiled_iq2r_dir / shard, output_dir / shard, force=force
            )
            keys = iq2r_compiled_tensor_keys(layer, projection)
            weight_map[keys["data"]] = shard
            weight_map[keys["auxiliary"]] = shard
            shard_names.append(shard)
        shard_bytes = sum((output_dir / name).stat().st_size for name in shard_names)
        total_iq2r_bytes += shard_bytes
        layer_records.append(
            {
                "layer_index": layer,
                "overlay_shards": shard_names,
                "overlay_shard_bytes": shard_bytes,
            }
        )

    overlay_config = dict(config)
    overlay_config["quantization_config"] = {
        "quant_method": "iq2r",
        "base_quantization_config": base_quantization_config,
        "iq2r_modules": [
            "model.layers.*.mlp.experts",
            *(["model.layers.*.mlp.shared_experts"] if fused else []),
        ],
    }
    _atomic_write_json(output_dir / "config.json", overlay_config)

    overlay_index = {
        "metadata": {
            **(source_index.get("metadata") or {}),
            "iq2r_layer_count": len(moe_layers),
            "iq2r_first_layer": layout.first_moe_layer,
            "iq2r_bytes": total_iq2r_bytes,
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    _atomic_write_json(output_dir / "model.safetensors.index.json", overlay_index)

    compiled_experts = layout.expert_count + int(fused)
    payload_bytes_per_layer = compiled_experts * sum(
        metadata.data_bytes + metadata.auxiliary_bytes
        for metadata in (
            _projection_metadata(layout, projection) for projection in _SHARD_SUFFIXES
        )
    )
    manifest = {
        "model_family": layout.model_family,
        "source_root": layout.source_root,
        "base_model": str(model_dir),
        "compiled_iq2r_checkpoint": str(compiled_iq2r_dir),
        "quality": quality,
        "first_moe_layer": layout.first_moe_layer,
        "layer_count": len(moe_layers),
        "expert_count": layout.expert_count,
        "compiled_expert_count": compiled_experts,
        "fused_shared_expert": fused,
        "hidden_size": layout.hidden_size,
        "intermediate_size": layout.intermediate_size,
        "iq2r_bytes": total_iq2r_bytes,
        "iq2r_payload_bytes": payload_bytes_per_layer * len(moe_layers),
        "removed_source_tensor_count": removed_source_tensor_count,
        "source_shards": source_shards,
        "layers": layer_records,
    }
    manifest_path = output_dir / "iq2r-overlay-manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--compiled-iq2r-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest = create_glm5_iq2r_overlay(
        args.model_dir, args.compiled_iq2r_dir, args.output_dir, force=args.force
    )
    print(json.dumps({"manifest": str(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["create_glm5_iq2r_overlay"]
