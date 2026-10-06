# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 IQ2R overlay on a tiny synthetic FP8 model and compiled checkpoint."""

import json

import pytest
import torch
from safetensors.torch import save_file

import aiter.iq2r_glm53_pack_checkpoint as pack_checkpoint
from aiter.iq2r_glm5_compile import (
    iq2r_compiled_tensor_keys,
    iq2r_glm5_shared_source_keys,
    iq2r_glm5_source_keys,
)
from aiter.iq2r_overlay import create_glm5_iq2r_overlay, main
from aiter.ops.iq2r_format import (
    IQ2R_ACTIVATION_BASIS,
    IQ2R_FORMAT_NAME,
    IQ2RMetadata,
)

_LAYERS = 5
_FIRST_MOE = 3
_EXPERTS = 2
_HIDDEN = 32
_INTERMEDIATE = 32
_QUALITY = "calibrated-o0"


def _write_source_model(path) -> None:
    path.mkdir()
    tensors = {
        "model.embed_tokens.weight": torch.arange(4),
        "model.layers.0.mlp.gate_proj.weight": torch.tensor([10]),
        "model.layers.3.self_attn.q_proj.weight": torch.tensor([12]),
    }
    # Layer 5 is the MTP layer after num_hidden_layers. It must stay FP8.
    for layer in range(_FIRST_MOE, _LAYERS + 1):
        for index, name in enumerate(iq2r_glm5_shared_source_keys(layer).values()):
            tensors[name] = torch.tensor([layer, -1, index])
        for expert in range(_EXPERTS):
            for index, name in enumerate(iq2r_glm5_source_keys(layer, expert).values()):
                tensors[name] = torch.tensor([layer, expert, index])
    save_file(tensors, path / "model-00001-of-00001.safetensors")
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": _LAYERS,
        "first_k_dense_replace": _FIRST_MOE,
        "n_routed_experts": _EXPERTS,
        "n_shared_experts": 1,
        "hidden_size": _HIDDEN,
        "moe_intermediate_size": _INTERMEDIATE,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 789}, "weight_map": weight_map})
    )
    (path / "tokenizer.json").write_text("{}")


def _write_compiled(path, *, fused: bool, layers=None, experts=None) -> None:
    """Write shards and a config in the form ``aiter.iq2r_glm5_compile`` emits."""
    path.mkdir()
    layers = list(range(_FIRST_MOE, _LAYERS)) if layers is None else layers
    compiled_experts = _EXPERTS + int(fused)
    shard_experts = compiled_experts if experts is None else experts
    for layer in layers:
        for projection, suffix, metadata in (
            ("gate_up", "gate-up", IQ2RMetadata(2 * _INTERMEDIATE, _HIDDEN)),
            ("down", "down", IQ2RMetadata(_HIDDEN, _INTERMEDIATE)),
        ):
            keys = iq2r_compiled_tensor_keys(layer, projection)
            save_file(
                {
                    keys["data"]: torch.full(
                        (shard_experts, metadata.data_bytes), layer, dtype=torch.uint8
                    ),
                    keys["auxiliary"]: torch.zeros(
                        (shard_experts, metadata.auxiliary_bytes), dtype=torch.uint8
                    ),
                    keys["tile_n"]: torch.tensor([128], dtype=torch.int32),
                },
                path / f"iq2r-layer-{layer:04d}-{suffix}.safetensors",
                metadata={
                    "format": "pt",
                    "iq2r_format": IQ2R_FORMAT_NAME,
                    "iq2r_activation_basis": IQ2R_ACTIVATION_BASIS,
                    "iq2r_layer": str(layer),
                    "iq2r_projection": projection,
                    "iq2r_quality": _QUALITY,
                },
            )
    config = {
        "model_type": "glm_moe_dsa",
        "compiled_tensor_parallel_size": 1,
        "compiled_expert_parallel_size": 1,
        "iq2r": {
            "format": IQ2R_FORMAT_NAME,
            "activation_basis": IQ2R_ACTIVATION_BASIS,
            "model_family": "glm_moe_dsa",
            "quality": _QUALITY,
            "compiled_expert_count": compiled_experts,
            "fused_shared_expert": fused,
            "compiled_layers": layers,
        },
    }
    (path / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("fused", [True, False])
def test_overlay_replaces_routed_experts_with_compiled_shards(tmp_path, fused):
    source, compiled, output = (tmp_path / n for n in ("source", "compiled", "out"))
    _write_source_model(source)
    _write_compiled(compiled, fused=fused)
    manifest_path = create_glm5_iq2r_overlay(source, compiled, output)

    config = json.loads((output / "config.json").read_text())
    quantization = config["quantization_config"]
    assert quantization["quant_method"] == "iq2r"
    assert quantization["base_quantization_config"]["quant_method"] == "fp8"
    assert quantization["iq2r_modules"] == ["model.layers.*.mlp.experts"] + (
        ["model.layers.*.mlp.shared_experts"] if fused else []
    )

    weight_map = json.loads((output / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    for layer in range(_FIRST_MOE, _LAYERS):
        for expert in range(_EXPERTS):
            assert not set(iq2r_glm5_source_keys(layer, expert).values()) & set(
                weight_map
            )
        shared = set(iq2r_glm5_shared_source_keys(layer).values())
        assert (shared & set(weight_map)) == (set() if fused else shared)
        for projection, suffix in (("gate_up", "gate-up"), ("down", "down")):
            shard = f"iq2r-layer-{layer:04d}-{suffix}.safetensors"
            keys = iq2r_compiled_tensor_keys(layer, projection)
            assert weight_map[keys["data"]] == shard
            assert weight_map[keys["auxiliary"]] == shard
            assert (output / shard).resolve() == (compiled / shard).resolve()
    # Dense layers, attention and the MTP layer keep their FP8 tensors.
    assert "model.layers.0.mlp.gate_proj.weight" in weight_map
    assert "model.layers.3.self_attn.q_proj.weight" in weight_map
    assert set(iq2r_glm5_source_keys(_LAYERS, 0).values()) <= set(weight_map)
    assert (output / "model-00001-of-00001.safetensors").is_symlink()
    assert (output / "tokenizer.json").is_symlink()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["compiled_expert_count"] == _EXPERTS + int(fused)
    assert manifest["fused_shared_expert"] is fused
    assert manifest["removed_source_tensor_count"] == 2 * (_EXPERTS + fused) * 6


def test_overlay_feeds_the_pack_tool(tmp_path):
    source, compiled, overlay, packed = (
        tmp_path / n for n in ("source", "compiled", "overlay", "packed")
    )
    _write_source_model(source)
    _write_compiled(compiled, fused=True)
    assert (
        main(
            [
                "--model-dir",
                str(source),
                "--compiled-iq2r-dir",
                str(compiled),
                "--output-dir",
                str(overlay),
            ]
        )
        == 0
    )
    pack_checkpoint.main([str(overlay), str(packed), "base"])
    # The iq2r stage needs gfx950; its file names are what meta indexes.
    for path in overlay.glob("iq2r-layer-*.safetensors"):
        (packed / path.name).write_bytes(path.read_bytes())
    pack_checkpoint.main([str(overlay), str(packed), "meta"])

    config = json.loads((packed / "config.json").read_text())
    assert config["quantization_config"]["iq2r_layout"] == pack_checkpoint.LAYOUT
    weight_map = json.loads((packed / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert all((packed / name).is_file() for name in set(weight_map.values()))
    keys = iq2r_compiled_tensor_keys(_FIRST_MOE, "gate_up")
    assert weight_map[keys["data"]] == "iq2r-layer-0003-gate-up.safetensors"


@pytest.mark.parametrize(
    "compiled_kwargs, match",
    [
        ({"layers": [3]}, "does not cover MoE layers 3-4"),
        ({"experts": _EXPERTS}, "invalid compiled shard.*expected U8"),
    ],
)
def test_overlay_validates_compiled_checkpoint_before_writing(
    tmp_path, compiled_kwargs, match
):
    source, compiled, output = (tmp_path / n for n in ("source", "compiled", "out"))
    _write_source_model(source)
    _write_compiled(compiled, fused=True, **compiled_kwargs)
    with pytest.raises(ValueError, match=match):
        create_glm5_iq2r_overlay(source, compiled, output)
    assert not output.exists()


def test_overlay_rejects_non_glm_source(tmp_path):
    source, compiled = tmp_path / "source", tmp_path / "compiled"
    _write_source_model(source)
    config = json.loads((source / "config.json").read_text())
    config.update(architectures=["LlamaForCausalLM"], model_type="llama")
    (source / "config.json").write_text(json.dumps(config))
    _write_compiled(compiled, fused=True)
    with pytest.raises(ValueError, match="GLM MoE DSA"):
        create_glm5_iq2r_overlay(source, compiled, tmp_path / "out")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
