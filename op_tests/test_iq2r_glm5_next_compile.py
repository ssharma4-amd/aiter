# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3-Flash (glm5_next) IQ2R compilation: packed layer files for the
GLM5_NEXT kernel shape (down rows padded to 4,608), LDLQ on 4,096-wide
Hessians, and the checkpoint around them."""

import functools
import json
import re
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import aiter.quantization.iq2r.glm5.calibrate as calibrate_module
import aiter.quantization.iq2r.glm5.compile as compile_module
from aiter.iq2r_glm53 import (
    GLM5_NEXT,
    iq2r_glm5_down_bytes,
    iq2r_glm53_gate_bytes,
    iq2r_glm53_unpack_v2,
)
from aiter.ops.iq2r_v2 import IQ2R_V2_LEVELS_C4, iq2r_v2_materialize
from aiter.quantization.iq2r.glm5.compile import (
    GLM5_NEXT_LAYOUTS,
    compile_glm5_iq2r,
    glm5_source_layout,
    iq2r_compiled_tensor_keys,
    iq2r_glm5_source_keys,
)

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
            "swiglu_limit": 10.0,
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
    assert (layout.layer_count, layout.first_moe_layer, layout.expert_count) == (
        45,
        3,
        288,
    )
    assert (layout.hidden_size, layout.intermediate_size) == (4096, 2048)
    assert layout.shape == GLM5_NEXT and layout.prefix == _PREFIX
    assert layout.moe_layers == 42
    assert layout.layout_name(512) == "glm5next-packed-v2"
    assert layout.layout_name(1024) == "glm5next-packed-v3"
    with pytest.raises(ValueError, match="one shared expert"):
        glm5_source_layout(_flash_config(n_shared_experts=2))
    for name, value in (
        ("hidden_size", 6144),
        ("n_routed_experts", 256),
        ("swiglu_limit", 7.0),
    ):
        with pytest.raises(ValueError, match=re.escape(f"has {{{name!r}: {value}}}")):
            glm5_source_layout(_flash_config(**{name: value}))
    with pytest.raises(ValueError, match="text_config"):
        glm5_source_layout({**_flash_config(), "text_config": None})


def test_source_and_compiled_keys_use_language_model_prefix():
    names = iq2r_glm5_source_keys(3, 287, _PREFIX)
    assert names["up_proj_weight"] == (
        "model.language_model.layers.3.mlp.experts.287.up_proj.weight"
    )
    assert iq2r_compiled_tensor_keys(7, "gate_up", _PREFIX) == {
        "data": "model.language_model.layers.7.mlp.up_gate_proj.0.iq2r_data",
        "auxiliary": "model.language_model.layers.7.mlp.up_gate_proj.0.iq2r_auxiliary",
        "tile_n": "model.language_model.layers.7.mlp.up_gate_proj.0.iq2r_tN",
    }


def _has_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in (
        torch.cuda.get_device_properties(0).gcnArchName
    )


@pytest.fixture
def two_experts(monkeypatch):
    # Two routed experts instead of 288 keep the checkpoint tests small.
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
    (path / "processor_config.json").write_text("{}")


def _random_weights(seed: int) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return {
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


def _use_weights(monkeypatch, weights) -> None:
    monkeypatch.setattr(
        compile_module,
        "_source_projection",
        lambda reader, layout, layer, expert, projection, device: weights[
            layer, expert, projection
        ],
    )


def _read_layer(output: Path, layer: int) -> list[torch.Tensor]:
    """(gate_quad, gate_auxiliary, down_data, down_auxiliary) of a layer file pair."""
    tensors = []
    for projection, suffix in (("gate_up", "gate-up"), ("down", "down")):
        keys = iq2r_compiled_tensor_keys(layer, projection, _PREFIX)
        path = output / f"iq2r-layer-{layer:04d}-{suffix}.safetensors"
        with safe_open(path, "pt") as handle:
            assert set(handle.keys()) == set(keys.values())
            assert handle.get_tensor(keys["tile_n"]).tolist() == [128]
            tensors += [handle.get_tensor(keys[k]).cuda() for k in ("data", "auxiliary")]
    return tensors


@pytest.mark.skipif(not _has_gfx950(), reason="IQ2R encoding requires gfx950")
@pytest.mark.parametrize("entries", [512, 1024])
def test_compile_packs_glm5_next_layers_that_decode(
    tmp_path, monkeypatch, two_experts, entries
):
    source, output = tmp_path / "source", tmp_path / "out"
    _write_source_model(source)
    weights = _random_weights(0x4096 + entries)
    _use_weights(monkeypatch, weights)
    run = functools.partial(
        compile_glm5_iq2r,
        source,
        output,
        diagnostic_uniform_importance=True,
        iterations=1,
        sample_vectors=4096,
        codebook_entries=entries,
    )
    assert run(layer_indices=[3]) is None
    config_path = run()
    config = json.loads(config_path.read_text())
    layout_name = GLM5_NEXT_LAYOUTS[entries]
    assert config["quantization_config"]["iq2r_layout"] == layout_name
    assert config["quantization_config"]["iq2r_modules"] == [
        "model.layers.3.mlp.experts*",
        "model.layers.4.mlp.experts*",
    ]
    assert config["text_config"]["hidden_size"] == _HIDDEN

    for layer in (3, 4):
        for suffix in ("gate-up", "down"):
            path = output / f"iq2r-layer-{layer:04d}-{suffix}.safetensors"
            with safe_open(path, "pt") as handle:
                assert handle.metadata()["iq2r_layout"] == layout_name
        packed = _read_layer(output, layer)
        assert tuple(packed[0].shape) == (
            _EXPERTS,
            iq2r_glm53_gate_bytes(2 * _INTERMEDIATE, entries, GLM5_NEXT),
        )
        assert tuple(packed[2].shape) == (
            _EXPERTS,
            iq2r_glm5_down_bytes(_INTERMEDIATE, entries, GLM5_NEXT),
        )
        canonical = iq2r_glm53_unpack_v2(
            *packed,
            intermediate_size=_INTERMEDIATE,
            levels=IQ2R_V2_LEVELS_C4,
            shape=GLM5_NEXT,
        )
        for expert in range(_EXPERTS):
            for projection, data, aux, n, k in (
                ("gate_up", *canonical[:2], 2 * _INTERMEDIATE, _HIDDEN),
                ("down", *canonical[2:], GLM5_NEXT.down_rows, _INTERMEDIATE),
            ):
                decoded = iq2r_v2_materialize(
                    data[expert],
                    aux[expert],
                    n,
                    k,
                    IQ2R_V2_LEVELS_C4,
                    codebook_entries=entries,
                )
                weight = weights[layer, expert, projection]
                if projection == "down":
                    # The kernels' padding rows hold zero weights.
                    assert decoded[_HIDDEN:].abs().max() < 1e-30
                    decoded = decoded[:_HIDDEN]
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
    assert iq2r_compiled_tensor_keys(3, "down", _PREFIX)["data"] in names
    assert f"{_PREFIX}layers.5.mlp.experts.0.up_proj.weight" in names
    assert f"{_PREFIX}layers.4.mlp.shared_experts.up_proj.weight" in names
    assert "model.visual.blocks.0.attn.qkv.weight" in names
    for name in ("chat_template.jinja", "processor_config.json"):
        assert (output / name).is_file()


def _write_hessians(path: Path, layers, seed: int) -> None:
    """Random correlated 4,096-wide layer Hessians (calibrate --hessian-dir)."""
    generator = torch.Generator().manual_seed(seed)
    buckets = {}
    for name in ("all", "decision"):
        bucket = calibrate_module.LayerHessianAccumulator(layers)
        for layer in layers:
            basis = torch.randn((_HIDDEN, 32), generator=generator)
            x = torch.randn((256, 32), generator=generator) @ basis.T
            bucket.observe(layer, x + 0.3 * torch.randn(x.shape, generator=generator))
        buckets[name] = bucket
    calibrate_module.save_hessians(path, buckets, {})


@pytest.mark.skipif(not _has_gfx950(), reason="IQ2R encoding requires gfx950")
def test_compile_ldlq_reencodes_glm5_next_gate_up_only(
    tmp_path, monkeypatch, two_experts
):
    source, plain, ldlq = tmp_path / "source", tmp_path / "plain", tmp_path / "ldlq"
    _write_source_model(source)
    hessians = tmp_path / "hessians"
    _write_hessians(hessians, range(3, _LAYERS), 0x4D0)
    _use_weights(monkeypatch, _random_weights(0x4D1))
    run = functools.partial(
        compile_glm5_iq2r,
        source,
        diagnostic_uniform_importance=True,
        iterations=1,
        sample_vectors=4096,
        codebook_entries=1024,
        layer_indices=[3],
    )
    run(plain)
    run(ldlq, ldlq_hessian_dir=hessians)
    layout = glm5_source_layout(json.loads((source / "config.json").read_text()))
    factor = compile_module.load_glm5_hessians(hessians, layout).factor(3, "cuda")
    assert factor.u.shape == (_HIDDEN, _HIDDEN)
    for suffix, projection in (("gate-up", "gate_up"), ("down", "down")):
        keys = iq2r_compiled_tensor_keys(3, projection, _PREFIX)
        name = f"iq2r-layer-0003-{suffix}.safetensors"
        with safe_open(plain / name, "pt") as a, safe_open(ldlq / name, "pt") as b:
            assert json.loads(b.metadata()["iq2r_ldlq"])["projections"] == ["gate_up"]
            assert "iq2r_ldlq" not in a.metadata()
            same = torch.equal(a.get_tensor(keys["data"]), b.get_tensor(keys["data"]))
            # Down is not re-encoded; gate/up records change.
            assert same == (projection == "down")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
