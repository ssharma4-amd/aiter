# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""iq2r_glm53_unpack_v2 is the exact inverse of iq2r_glm53_pack_v2 (CPU only).

Canonical IQ2R-v2 buffers come from the format's own writers with random
indices, signs, levels, d and gains, so every bit of the records is exercised.
"""

import pytest
import torch

from aiter.iq2r_glm53 import (
    HIDDEN,
    iq2r_glm53_pack_v2,
    iq2r_glm53_unpack_v2,
    iq2r_glm53_v2_slice_down_aux,
    iq2r_glm53_v2_slice_gate_aux,
)
from aiter.ops.iq2r_format import IQ2RMetadata, iq2r_slice_input_data
from aiter.ops.iq2r_v2 import (
    IQ2R_V2_CODEBOOK_MAX_BYTE,
    IQ2R_V2_CODEBOOK_MIN_BYTE,
    IQ2R_V2_LEVELS_C4,
    iq2r_v2_materialize,
    iq2r_v2_write_auxiliary,
    iq2r_v2_write_data,
)

LEVELS = IQ2R_V2_LEVELS_C4


def _canonical(n: int, k: int, generator: torch.Generator):
    indices = torch.randint(0, 512, (n, k // 8), generator=generator)
    negative = torch.rand((n, k), generator=generator) < 0.5
    levels = torch.randint(0, 16, (n, k // 32), generator=generator)
    data = iq2r_v2_write_data(indices, negative, levels, LEVELS)
    book = torch.randint(
        IQ2R_V2_CODEBOOK_MIN_BYTE,
        IQ2R_V2_CODEBOOK_MAX_BYTE + 1,
        (1, 512, 8),
        generator=generator,
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    d = torch.randint(100, 140, (n, k // 256), generator=generator, dtype=torch.uint8)
    gains = tuple(float(g) for g in 0.5 + torch.rand(2, generator=generator))
    return data, iq2r_v2_write_auxiliary(book.float(), d, gains)


def _layer(intermediate: int, experts: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    gate = [_canonical(2 * intermediate, HIDDEN, generator) for _ in range(experts)]
    down = [_canonical(HIDDEN, intermediate, generator) for _ in range(experts)]

    def stack(parts, i):
        return torch.stack([p[i] for p in parts])

    return stack(gate, 0), stack(gate, 1), stack(down, 0), stack(down, 1)


@pytest.mark.parametrize("intermediate", [256, 512, 2048])
def test_unpack_inverts_pack(intermediate):
    canonical = _layer(intermediate, 2, intermediate)
    packed = iq2r_glm53_pack_v2(
        *canonical, intermediate_size=intermediate, levels=LEVELS
    )
    unpacked = iq2r_glm53_unpack_v2(
        *packed, intermediate_size=intermediate, levels=LEVELS
    )
    # Canonical bytes, including the zero padding blocks of the gate/up records.
    for got, want in zip(unpacked, canonical):
        assert torch.equal(got, want)
    repacked = iq2r_glm53_pack_v2(
        *unpacked, intermediate_size=intermediate, levels=LEVELS
    )
    for got, want in zip(repacked, packed):
        assert torch.equal(got, want)


def test_unpack_of_tp_slices():
    """Unpacking a TP shard sliced from a packed full layer equals slicing the canonical weights."""
    full, shard, rank = 1024, 256, 2
    canonical = _layer(full, 1, 7)
    packed = iq2r_glm53_pack_v2(*canonical, intermediate_size=full, levels=LEVELS)
    from aiter.iq2r_glm53 import iq2r_glm53_slice_gate

    rows = (2 * shard * rank, 2 * shard)
    sliced = (
        iq2r_glm53_slice_gate(packed[0], *rows),
        iq2r_glm53_v2_slice_gate_aux(packed[1], *rows, 2 * full, LEVELS.classes),
        iq2r_slice_input_data(
            packed[2], IQ2RMetadata(HIDDEN, full), shard * rank, shard
        ),
        iq2r_glm53_v2_slice_down_aux(
            packed[3], shard * rank, shard, full, LEVELS.classes
        ),
    )
    gd, ga, dd, da = iq2r_glm53_unpack_v2(
        *sliced, intermediate_size=shard, levels=LEVELS
    )
    dense_gate = iq2r_v2_materialize(
        canonical[0][0], canonical[1][0], 2 * full, HIDDEN, LEVELS, apply_gain=False
    )
    dense_down = iq2r_v2_materialize(
        canonical[2][0], canonical[3][0], HIDDEN, full, LEVELS, apply_gain=False
    )
    got_gate = iq2r_v2_materialize(
        gd[0], ga[0], 2 * shard, HIDDEN, LEVELS, apply_gain=False
    )
    got_down = iq2r_v2_materialize(
        dd[0], da[0], HIDDEN, shard, LEVELS, apply_gain=False
    )
    assert torch.equal(got_gate, dense_gate[rows[0] : rows[0] + rows[1]])
    assert torch.equal(got_down, dense_down[:, shard * rank : shard * (rank + 1)])
