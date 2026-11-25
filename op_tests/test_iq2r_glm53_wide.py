# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3-Flash IQ2R MoE on ranks wider than the kernels (TP1: intermediate
2048, TP2: 1024), run as 512-wide TP4 shards, against the dense Torch
reference and against the TP4 ranks they are made of (gfx950).
"""

import functools

import pytest
import torch

from aiter.iq2r_glm53 import (
    GLM5_NEXT,
    IQ2RGlm53Workspace,
    iq2r_glm53_moe_out,
    iq2r_glm53_shards,
    iq2r_glm53_slice_layer,
)
from test_iq2r_glm53 import (
    _check_moe,
    _glm53_layer,
    _has_gfx950,
    _reference_moe,
    _reference_shared,
    _routing,
    _shared_expert,
)

pytestmark = pytest.mark.skipif(not _has_gfx950(), reason="requires gfx950")

_FULL = 2048  # GLM-5.3-Flash moe_intermediate_size
_ENTRIES = 1024  # the v3 checkpoints' codebook
_STACKS = ("gate", "gate_aux", "down", "down_aux")


@pytest.fixture(scope="module")
def flash_layer():
    return _glm53_layer(_FULL, _ENTRIES, GLM5_NEXT.name)


@functools.lru_cache(maxsize=8)
def _rank(tp: int, rank: int):
    """Packed stacks of one rank of a TP-``tp`` deployment."""
    layer = _glm53_layer(_FULL, _ENTRIES, GLM5_NEXT.name)
    width = _FULL // tp
    return tuple(
        iq2r_glm53_slice_layer(
            name, weight, rank * width, width, _FULL, _ENTRIES, GLM5_NEXT
        )
        for name, weight in zip(_STACKS, layer.packed)
    )


def _run(tp, hidden, topk_weights, topk_ids, shared=None):
    """Every rank's output of a TP-``tp`` deployment, summed in FP32 as the
    all-reduce does (one rank at TP1)."""
    width = _FULL // tp
    total = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    for rank in range(tp):
        workspace = IQ2RGlm53Workspace.allocate(
            hidden.shape[0], width, device="cuda", shape=GLM5_NEXT
        )
        output = torch.empty_like(hidden)
        part = None
        if shared is not None:
            part = shared[rank]
        iq2r_glm53_moe_out(
            hidden, *_rank(tp, rank), topk_weights, topk_ids, output, workspace,
            shared=part,
        )  # fmt: skip
        total += output.float()
    return total


def test_wide_shards():
    assert [iq2r_glm53_shards(i) for i in (256, 512, 1024, 2048)] == [1, 1, 2, 4]
    with pytest.raises(ValueError, match="rank intermediate"):
        iq2r_glm53_shards(768)
    stacks = _rank(1, 0)
    assert all(t.dim() == 3 and t.shape[:2] == (4, GLM5_NEXT.experts) for t in stacks)


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize(
    "tokens", [1, 2, 3, 4, 8, 16, 33, 64, 256, 300, 1024, 4096, 5000]
)
def test_wide_moe_matches_reference(flash_layer, tp, tokens):
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5A00 + tokens, GLM5_NEXT)
    expected = _reference_moe(flash_layer, hidden, topk_weights, topk_ids)
    actual = _run(tp, hidden, topk_weights, topk_ids).to(torch.bfloat16)
    _check_moe(actual, expected)


@pytest.mark.parametrize("tokens", [1, 4, 16, 256, 1024])
def test_tp1_equals_tp4_ranks(flash_layer, tokens):
    """A TP1 rank is its four TP4 shards: the same kernels on the same slices,
    summed in FP32 like four TP4 ranks' outputs."""
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5A40 + tokens, GLM5_NEXT)
    workspace = IQ2RGlm53Workspace.allocate(
        tokens, _FULL, device="cuda", shape=GLM5_NEXT
    )
    tp1 = torch.empty_like(hidden)
    iq2r_glm53_moe_out(
        hidden, *_rank(1, 0), topk_weights, topk_ids, tp1, workspace
    )  # fmt: skip
    tp4 = _run(4, hidden, topk_weights, topk_ids).to(torch.bfloat16)
    assert torch.equal(tp1, tp4)


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 16, 64, 256, 1024])
def test_wide_moe_fused_shared_expert(flash_layer, tp, tokens):
    """The shared expert packed per rank (stacked 512-wide shards) as route
    slot 8, deterministic."""
    shared, exact = _shared_expert(_FULL, GLM5_NEXT.name)
    width = _FULL // tp
    ranks = (
        [shared]
        if tp == 1
        else [_pack_rank_shared(exact, rank * width, width) for rank in range(tp)]
    )
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5A80 + tokens, GLM5_NEXT)
    expected = (
        _reference_moe(flash_layer, hidden, topk_weights, topk_ids).float()
        + _reference_shared(exact, hidden, GLM5_NEXT.swiglu_limit).float()
    ).to(torch.bfloat16)
    actual = _run(tp, hidden, topk_weights, topk_ids, ranks)
    again = _run(tp, hidden, topk_weights, topk_ids, ranks)
    _check_moe(actual.to(torch.bfloat16), expected)
    assert torch.equal(actual, again)


def _pack_rank_shared(exact, start, width):
    from aiter.iq2r_glm53 import iq2r_glm53_pack_shared

    gate, up, down = exact
    rows = slice(start, start + width)
    return iq2r_glm53_pack_shared(gate[rows], up[rows], down[:, rows], GLM5_NEXT)


def test_wide_rejects_unstacked_weights(flash_layer):
    hidden, topk_weights, topk_ids = _routing(4, 0x5AF0, GLM5_NEXT)
    workspace = IQ2RGlm53Workspace.allocate(4, _FULL, device="cuda", shape=GLM5_NEXT)
    with pytest.raises(ValueError, match="stacked shards"):
        iq2r_glm53_moe_out(
            hidden, *(t[0] for t in _rank(1, 0)), topk_weights, topk_ids,
            torch.empty_like(hidden), workspace,
        )  # fmt: skip
