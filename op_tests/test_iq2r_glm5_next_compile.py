# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3-Flash (glm5_next) IQ2R-v2 compilation: canonical layer files."""

import functools
import json
import re
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import aiter.quantization.iq2r.glm5.compile as compile_module
from aiter.iq2r_glm53 import GLM5_NEXT, iq2r_glm53_pack_v2
from aiter.ops.iq2r_v2 import IQ2R_V2_LEVELS_C4, iq2r_v2_materialize
from aiter.quantization.iq2r.glm5.compile import (
    CANONICAL_LAYOUT,
    GLM5Layout,
    compile_glm5_iq2r,
    glm5_source_layout,
    iq2r_compiled_tensor_keys,
    iq2r_glm5_source_keys,
    iq2r_v2_canonical_aux_bytes,
)
from aiter.quantization.iq2r.glm5.decode_fp8 import decode_iq2r_to_fp8
from aiter.quantization.iq2r.glm5.pack import PACKED_LAYOUT, pack_glm5_next

_PREFIX = "model.language_model."
_HIDDEN = 4096
_INTERMEDIATE = 2048
_EXPERTS = 2
_LAYERS = 5


def _flash_config(**text) -> dict:
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "first_k_dense_replace": 3,
            "n_routed_experts": 288,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "hidden_size": _HIDDEN,
            "moe_intermediate_size": _INTERMEDIATE,
            **text,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }


def test_source_layout_reads_glm5_next_config():
    layout = glm5_source_layout(_flash_config())
    assert layout == GLM5Layout(
        45, 3, 288, 4096, 2048, 128, 128, _PREFIX, CANONICAL_LAYOUT
    )
    assert layout.moe_layers == 42
    assert not layout.packed
    with pytest.raises(ValueError, match="one shared expert"):
        glm5_source_layout(_flash_config(n_shared_experts=2))
    for name, value in (("hidden_size", 6144), ("n_routed_experts", 256)):
        with pytest.raises(ValueError, match=re.escape(f"has {{{name!r}: {value}}}")):
            glm5_source_layout(_flash_config(**{name: value}))


def test_source_and_canonical_keys_use_language_model_prefix():
    names = iq2r_glm5_source_keys(3, 287, _PREFIX)
    assert names["up_proj_weight"] == (
        "model.language_model.layers.3.mlp.experts.287.up_proj.weight"
    )
    assert iq2r_compiled_tensor_keys(7, "gate_up", CANONICAL_LAYOUT, _PREFIX) == {
        "data": "model.language_model.layers.7.mlp.experts.iq2r_gate_up_data",
        "auxiliary": "model.language_model.layers.7.mlp.experts.iq2r_gate_up_auxiliary",
    }
    # One E4M3 codebook, one E8M0 d per 256 inputs per row, two FP32 gains.
    assert iq2r_v2_canonical_aux_bytes(4096, 2048) == 4096 + 4096 * 8 + 8


def _has_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in (
        torch.cuda.get_device_properties(0).gcnArchName
    )


@pytest.fixture
def two_experts(monkeypatch):
    # Two routed experts instead of 288 keep the checkpoint test small.
    monkeypatch.setitem(compile_module._GLM5_NEXT_SHAPE, "n_routed_experts", _EXPERTS)


def _write_source_model(path: Path) -> None:
    """Write a GLM-5.3-Flash-shaped FP8 model with placeholder tensors."""
    path.mkdir()
    tensors = {
        "model.language_model.embed_tokens.weight": torch.arange(4.0),
        "model.visual.blocks.0.attn.qkv.weight": torch.tensor([7]),
    }
    # Layer 5 is the MTP layer after num_hidden_layers. It must stay FP8.
    for layer in range(3, _LAYERS + 1):
        for kind in ("weight", "weight_scale_inv"):
            tensors[f"{_PREFIX}layers.{layer}.mlp.shared_experts.up_proj.{kind}"] = (
                torch.tensor([layer, -1])
            )
        for expert in range(_EXPERTS):
            for index, name in enumerate(
                iq2r_glm5_source_keys(layer, expert, _PREFIX).values()
            ):
                tensors[name] = torch.tensor([layer, expert, index])
    save_file(tensors, path / "model-00001-of-00001.safetensors")
    config = _flash_config(num_hidden_layers=_LAYERS, n_routed_experts=_EXPERTS)
    (path / "config.json").write_text(json.dumps(config))
    weight_map = dict.fromkeys(tensors, "model-00001-of-00001.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    (path / "chat_template.jinja").write_text("{{ messages }}")


@pytest.mark.skipif(not _has_gfx950(), reason="IQ2R encoding requires gfx950")
def test_compile_writes_canonical_layers_that_decode(
    tmp_path, monkeypatch, two_experts
):
    source, output = tmp_path / "source", tmp_path / "out"
    _write_source_model(source)
    generator = torch.Generator(device="cuda").manual_seed(0x4096)
    weights = {
        (layer, expert, projection): torch.randn(
            shape, generator=generator, device="cuda"
        )
        * 0.02
        for layer in (3, 4)
        for expert in range(_EXPERTS)
        for projection, shape in (
            ("gate_up", (2 * _INTERMEDIATE, _HIDDEN)),
            ("down", (_HIDDEN, _INTERMEDIATE)),
        )
    }
    monkeypatch.setattr(
        compile_module,
        "_source_projection",
        lambda reader, layout, layer, expert, projection, device: weights[
            layer, expert, projection
        ],
    )
    run = functools.partial(
        compile_glm5_iq2r,
        source,
        output,
        diagnostic_uniform_importance=True,
        iterations=1,
        sample_vectors=4096,
    )
    config_path = run()
    config = json.loads(config_path.read_text())
    assert config["quantization_config"]["iq2r_layout"] == CANONICAL_LAYOUT
    assert config["text_config"]["hidden_size"] == _HIDDEN

    shapes = {"gate_up": (2 * _INTERMEDIATE, _HIDDEN), "down": (_HIDDEN, _INTERMEDIATE)}
    reference = {}
    for layer in (3, 4):
        for projection, suffix in (("gate_up", "gate-up"), ("down", "down")):
            keys = iq2r_compiled_tensor_keys(
                layer, projection, CANONICAL_LAYOUT, _PREFIX
            )
            path = output / f"iq2r-layer-{layer:04d}-{suffix}.safetensors"
            with safe_open(path, "pt") as handle:
                assert set(handle.keys()) == set(keys.values())
                assert handle.metadata()["iq2r_layout"] == CANONICAL_LAYOUT
                data = handle.get_tensor(keys["data"]).cuda()
                auxiliary = handle.get_tensor(keys["auxiliary"]).cuda()
            n, k = shapes[projection]
            assert tuple(auxiliary.shape) == (
                _EXPERTS,
                iq2r_v2_canonical_aux_bytes(n, k),
            )
            for expert in range(_EXPERTS):
                weight = weights[layer, expert, projection]
                decoded = iq2r_v2_materialize(
                    data[expert], auxiliary[expert], n, k, IQ2R_V2_LEVELS_C4
                )
                reference[layer, expert, projection] = decoded
                # A 2-bit reconstruction: correlated with the source, scale kept.
                cosine = torch.nn.functional.cosine_similarity(
                    decoded.flatten(), weight.flatten(), dim=0
                )
                assert cosine > 0.9
                gain = (decoded * weight).sum() / (weight * weight).sum()
                assert 0.9 < gain < 1.05

    index = json.loads((output / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"])
    # Routed experts of the compiled layers are replaced; the MTP layer's and
    # the shared experts stay, as do the vision tower and the rest.
    assert f"{_PREFIX}layers.3.mlp.experts.0.up_proj.weight" not in names
    assert f"{_PREFIX}layers.5.mlp.experts.0.up_proj.weight" in names
    assert f"{_PREFIX}layers.4.mlp.shared_experts.up_proj.weight" in names
    assert "model.visual.blocks.0.attn.qkv.weight" in names
    assert (output / "chat_template.jinja").is_file()

    # The block-FP8 decode serves the same 2-bit experts under the source names.
    decoded_dir = tmp_path / "decoded"
    assert decode_iq2r_to_fp8(output, decoded_dir, layer_indices=[3]) is None
    config_path = decode_iq2r_to_fp8(output, decoded_dir)
    config = json.loads(config_path.read_text())
    assert config["quantization_config"]["quant_method"] == "fp8"
    index = json.loads((decoded_dir / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"])
    assert not any("iq2r" in name for name in names)
    for layer in (3, 4):
        for expert in range(_EXPERTS):
            keys = iq2r_glm5_source_keys(layer, expert, _PREFIX)
            matrices = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                shard = index["weight_map"][keys[f"{projection}_weight"]]
                with safe_open(decoded_dir / shard, "pt") as handle:
                    values = handle.get_tensor(keys[f"{projection}_weight"])
                    scale = handle.get_tensor(keys[f"{projection}_weight_scale_inv"])
                assert values.dtype == torch.float8_e4m3fn
                assert scale.dtype == torch.float32
                n, k = values.shape
                matrices[projection] = (
                    values.float().reshape(n // 128, 128, k // 128, 128)
                    * scale[:, None, :, None]
                ).reshape(n, k)
            for projection, expected in (
                ("gate_proj", reference[layer, expert, "gate_up"][0::2]),
                ("up_proj", reference[layer, expert, "gate_up"][1::2]),
                ("down_proj", reference[layer, expert, "down"]),
            ):
                expected = expected.cpu()
                error = (matrices[projection] - expected).norm() / expected.norm()
                assert error < 0.05, (layer, expert, projection, float(error))

    # The packed checkpoint holds the kernels' layout of the same layers under
    # GLM-5.3's packed tensor names.
    packed_dir = tmp_path / "packed"
    config = json.loads(pack_glm5_next(output, packed_dir).read_text())
    assert config["quantization_config"]["iq2r_layout"] == PACKED_LAYOUT
    index = json.loads((packed_dir / "model.safetensors.index.json").read_text())
    for layer in (3, 4):
        canonical = []
        for projection, suffix in (("gate_up", "gate-up"), ("down", "down")):
            keys = iq2r_compiled_tensor_keys(
                layer, projection, CANONICAL_LAYOUT, _PREFIX
            )
            path = output / f"iq2r-layer-{layer:04d}-{suffix}.safetensors"
            with safe_open(path, "pt") as handle:
                canonical += [handle.get_tensor(keys[k]).cuda() for k in keys]
        expected = iq2r_glm53_pack_v2(
            *canonical, intermediate_size=_INTERMEDIATE, shape=GLM5_NEXT
        )
        for projection, values in (("gate_up", expected[:2]), ("down", expected[2:])):
            keys = iq2r_compiled_tensor_keys(layer, projection, prefix=_PREFIX)
            assert keys["data"] not in names and keys["data"] in index["weight_map"]
            with safe_open(packed_dir / index["weight_map"][keys["data"]], "pt") as f:
                assert f.metadata()["iq2r_layout"] == PACKED_LAYOUT
                assert torch.equal(f.get_tensor(keys["data"]), values[0].cpu())
                assert torch.equal(f.get_tensor(keys["auxiliary"]), values[1].cpu())
                assert f.get_tensor(keys["tile_n"]).tolist() == [128]
    assert not any("iq2r_gate_up_data" in name for name in index["weight_map"])
    assert f"{_PREFIX}layers.4.mlp.shared_experts.up_proj.weight" in index["weight_map"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
