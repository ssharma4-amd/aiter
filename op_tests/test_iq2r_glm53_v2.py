# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 IQ2R-v2 packed MoE (TP4/TP8, gfx950) against a dense Torch reference.

The same kernels as IQ2R run IQ2R-v2 weights (aiter/ops/iq2r_v2.py); the
format is recognized from the packed auxiliary size. ``python3
op_tests/test_iq2r_glm53_v2.py`` times IQ2R and IQ2R-v2 side by side over a
token sweep, then runs the tests.
"""

import argparse
import functools
import itertools
from dataclasses import astuple
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import aiter
from aiter.iq2r_glm53 import (
    IQ2RGlm53Config,
    IQ2RGlm53Workspace,
    _config_problem,
    iq2r_glm53_moe_out,
    iq2r_glm53_pack_v2,
    iq2r_glm53_slice_gate,
    iq2r_glm53_v2_slice_down_aux,
    iq2r_glm53_v2_slice_gate_aux,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.iq2r_format import IQ2RMetadata, iq2r_slice_input_data
from aiter.ops.iq2r_v2 import (
    IQ2R_V2_LEVELS_C4,
    iq2r_v2_encode,
    iq2r_v2_gains,
    iq2r_v2_materialize,
    iq2r_v2_seed_codebook,
    iq2r_v2_write_auxiliary,
    iq2r_v2_write_data,
)
from aiter.test_common import run_perftest

SUPPORTED_GFX = ["gfx950"]
LEVELS = IQ2R_V2_LEVELS_C4

_EXPERTS = 257
_TOPK = 9
_HIDDEN = 6144


def _has_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in (
        torch.cuda.get_device_properties(0).gcnArchName
    )


pytestmark = pytest.mark.skipif(not _has_gfx950(), reason="requires gfx950")

# The reference shares the IQ2R test's activation model.
from test_iq2r_glm53 import (
    _check_moe,
    _glm53_layer,
    _reference_moe,
    _routing,
)


def _interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack((a, b), dim=1).reshape(-1, *a.shape[1:])


# A gain far from 1 makes a dropped or misapplied output gain fail the tests.
_TEST_GAIN = 0.8


def _encode_gate_up(gate, up, book, importance):
    """Canonical IQ2R-v2 gate/up: rows interleaved, one shared codebook."""
    weight = _interleave(gate, up)
    enc = iq2r_v2_encode(weight, importance, book, LEVELS)
    gains = iq2r_v2_gains(weight, enc.reconstruction, _TEST_GAIN, interleaved=True)
    data = iq2r_v2_write_data(enc.indices, enc.negative, enc.levels, LEVELS)
    return data, iq2r_v2_write_auxiliary(book[None], enc.d, gains)


def _encode_down(weight, book, importance):
    enc = iq2r_v2_encode(weight, importance, book, LEVELS)
    gains = iq2r_v2_gains(weight, enc.reconstruction, _TEST_GAIN)
    data = iq2r_v2_write_data(enc.indices, enc.negative, enc.levels, LEVELS)
    return data, iq2r_v2_write_auxiliary(book[None], enc.d, gains)


@functools.lru_cache(maxsize=2)
def _glm53_v2_layer(intermediate: int):
    seed = iq2r_v2_seed_codebook("cuda")
    books = (seed, (seed * 1.0625).to(torch.float8_e4m3fn).float())

    def weight(n: int, k: int, s: int):
        generator = torch.Generator(device="cuda").manual_seed(s)
        return torch.randn((n, k), generator=generator, device="cuda") * 0.02

    ones_h = torch.ones(_HIDDEN, device="cuda")
    ones_i = torch.ones(intermediate, device="cuda")
    gate = [
        _encode_gate_up(
            weight(intermediate, _HIDDEN, 0x5330 + i),
            weight(intermediate, _HIDDEN, 0x5350 + i),
            books[1],
            ones_h,
        )
        for i in range(3)
    ]
    down = [
        _encode_down(weight(_HIDDEN, intermediate, 0x5340 + i), books[i % 2], ones_i)
        for i in range(3)
    ]
    pick = torch.arange(_EXPERTS, device="cuda") % 3

    def stack(parts, index):
        return torch.stack([p[index].reshape(-1) for p in parts])[pick].contiguous()

    layer = SimpleNamespace(
        intermediate=intermediate,
        pick=pick,
        gate_weights=[
            iq2r_v2_materialize(d, a, 2 * intermediate, _HIDDEN, LEVELS)
            for d, a in gate
        ],
        down_weights=[
            iq2r_v2_materialize(d, a, _HIDDEN, intermediate, LEVELS) for d, a in down
        ],
        gate_data=stack(gate, 0),
        gate_auxiliary=stack(gate, 1),
        down_data=stack(down, 0),
        down_auxiliary=stack(down, 1),
    )
    layer.packed = iq2r_glm53_pack_v2(
        layer.gate_data,
        layer.gate_auxiliary,
        layer.down_data,
        layer.down_auxiliary,
        intermediate_size=intermediate,
        levels=LEVELS,
    )
    return layer


@pytest.fixture(scope="module", params=[256, 512], ids=["tp8", "tp4"])
def glm53_v2_layer(request):
    return _glm53_v2_layer(request.param)


@pytest.mark.parametrize(
    "tokens", [1, 2, 4, 8, 16, 24, 32, 64, 100, 128, 256, 512, 1024, 2048, 4096]
)
def test_glm53_v2_moe_matches_reference(glm53_v2_layer, tokens):
    layer = glm53_v2_layer
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5300 + tokens)
    expected = _reference_moe(layer, hidden, topk_weights, topk_ids)
    actual = torch.empty_like(hidden)
    workspace = IQ2RGlm53Workspace.allocate(tokens, layer.intermediate, device="cuda")
    iq2r_glm53_moe_out(hidden, *layer.packed, topk_weights, topk_ids, actual, workspace)
    _check_moe(actual, expected)


_EXPLICIT_CONFIGS = [
    IQ2RGlm53Config("nobarrier", 4, "route9", 4, 1),
    IQ2RGlm53Config("decode", 1, "single", 2, 1),
    IQ2RGlm53Config("nobarrier", 2, "single", 16, 1),
    IQ2RGlm53Config("decode", 4, "ordered", 8, 1),
    IQ2RGlm53Config("nobarrier", 1, "packed", 8, 1),
    IQ2RGlm53Config("prefill", 1, "prefill", 0, 2),
    IQ2RGlm53Config("prefill", 8, "prefill", 0, 1),
]


@pytest.mark.parametrize(
    "config", _EXPLICIT_CONFIGS, ids=lambda c: "-".join(map(str, astuple(c)))
)
@pytest.mark.parametrize("tokens", [4, 48, 1024])
def test_glm53_v2_moe_explicit_configs(glm53_v2_layer, tokens, config):
    layer = glm53_v2_layer
    problem = _config_problem(config, tokens, layer.intermediate)
    if problem:
        pytest.skip(problem)
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5390 + tokens)
    expected = _reference_moe(layer, hidden, topk_weights, topk_ids)
    actual = torch.empty_like(hidden)
    workspace = IQ2RGlm53Workspace.allocate(tokens, layer.intermediate, device="cuda")
    iq2r_glm53_moe_out(
        hidden, *layer.packed, topk_weights, topk_ids, actual, workspace, config
    )
    _check_moe(actual, expected)


@pytest.mark.parametrize("tp", [4, 8])
def test_glm53_v2_tp_slices_match_shard_pack(tp):
    """Slicing a full packed layer equals packing the TP shard's canonical buffers."""
    full_i = 2048
    shard = full_i // tp
    seed = iq2r_v2_seed_codebook("cuda")
    books = (seed, (seed * 1.0625).to(torch.float8_e4m3fn).float())
    generator = torch.Generator(device="cuda").manual_seed(tp)
    gate = torch.randn((full_i, _HIDDEN), generator=generator, device="cuda") * 0.02
    up = torch.randn((full_i, _HIDDEN), generator=generator, device="cuda") * 0.02
    down = torch.randn((_HIDDEN, full_i), generator=generator, device="cuda") * 0.02
    imp_h = torch.rand(_HIDDEN, generator=generator, device="cuda") + 0.5
    imp_i = torch.rand(full_i, generator=generator, device="cuda") + 0.5
    gd, ga = _encode_gate_up(gate, up, books[1], imp_h)
    dd, da = _encode_down(down, books[0], imp_i)
    full = iq2r_glm53_pack_v2(
        gd[None], ga[None], dd[None], da[None], intermediate_size=full_i, levels=LEVELS
    )
    rank = 1
    rows = slice(2 * shard * rank, 2 * shard * (rank + 1))
    cols = slice(shard * rank, shard * (rank + 1))
    # Canonical shard: re-encode-free slice of the canonical buffers is not
    # byte-addressable, so encode the shard from the same weights instead.
    sgd, sga = _encode_gate_up(gate[cols], up[cols], books[1], imp_h)
    sdd, sda = _encode_down(down[:, cols], books[0], imp_i[cols])
    want = iq2r_glm53_pack_v2(
        sgd[None],
        sga[None],
        sdd[None],
        sda[None],
        intermediate_size=shard,
        levels=LEVELS,
    )
    got_gate = iq2r_glm53_slice_gate(full[0], rows.start, rows.stop - rows.start)
    got_gate_aux = iq2r_glm53_v2_slice_gate_aux(
        full[1], rows.start, rows.stop - rows.start, 2 * full_i, LEVELS.classes
    )
    got_down = iq2r_slice_input_data(
        full[2], IQ2RMetadata(_HIDDEN, full_i), cols.start, shard
    )
    got_down_aux = iq2r_glm53_v2_slice_down_aux(
        full[3], cols.start, shard, full_i, LEVELS.classes
    )
    assert torch.equal(got_gate, want[0])
    # Gains are per matrix, so they differ between the full matrix and a shard.
    assert torch.equal(got_gate_aux[:, :-8], want[1][:, :-8])
    assert torch.equal(got_down, want[2])
    # Down codebook and d table: encoding is per 256-block, so the shard's d matches.
    assert torch.equal(got_down_aux[:, :-8], want[3][:, :-8])


def _time(layer, tokens: int) -> float:
    hidden, topk_weights, topk_ids = _routing(tokens, 0x5300 + tokens)
    workspace = IQ2RGlm53Workspace.allocate(tokens, layer.intermediate, device="cuda")
    output = torch.empty_like(hidden)
    moe = functools.partial(iq2r_glm53_moe_out, workspace=workspace)
    _, us = run_perftest(moe, hidden, *layer.packed, topk_weights, topk_ids, output)
    return us


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("GLM-5.3 IQ2R MoE unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(description="IQ2R vs IQ2R-v2 GLM-5.3 MoE timing")
    parser.add_argument(
        "-t", "--tokenNum", type=int, nargs="*", default=[2**i for i in range(13)]
    )
    parser.add_argument("-i", "--inter_dim", type=int, nargs="*", default=[256, 512])
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args()
    rows = []
    for inter, tokens in itertools.product(args.inter_dim, args.tokenNum):
        v1 = _time(_glm53_layer(inter), tokens)
        v2 = _time(_glm53_v2_layer(inter), tokens)
        rows.append(
            {
                "inter_dim": inter,
                "tokens": tokens,
                "iq2r us": v1,
                "iq2r-v2 us": v2,
                "v2/v1": v2 / v1,
            }
        )
    aiter.logger.info(
        "GLM-5.3 IQ2R vs IQ2R-v2 (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    if not args.no_tests:
        status = pytest.main([__file__, "-v"])
        if status != pytest.ExitCode.OK:
            raise SystemExit(status)


if __name__ == "__main__":
    main()
