# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""IQ2R-v2: 2-bit weights as 8-weight codewords with two-level, MFMA-native scales.

Each group of 8 weights is a 9-bit index into a learned 512-entry E4M3
codebook plus 8 explicit sign bits, in the record layout the gfx950 kernels
read. The scales are one E8M0 exponent ``d`` per row and 256 input columns,
plus a 4-bit *level* per 32 weights. Level ``l`` means
``m[l % 4] * 2**(l // 4)``, ``m[c] = 2**(c / 4)``: a power of two that joins
``d`` in the scaled-MFMA E8M0 operand, and a multiplier that selects one of 4
pre-scaled copies of the codebook, ``E4M3(codebook * m[c])``.

A weight therefore decodes as::

    w = 2**(d - 127) * 2**(l // 4) * E4M3(codebook[index] * m[l % 4]) * sign

and a per-matrix FP32 gain (two for the interleaved gate/up matrix) multiplies
the projection output. Every term is either part of the MFMA operands (E8M0
scales, FP8 codewords) or applied once per output: no extra MFMA and no
per-weight arithmetic. The codebook copies are part of the format, so the
encoder, the reference decoder and the kernels agree bit for bit.

The encoder follows the IQ2_XXS reference quantizer of ggml/llama.cpp (MIT,
Copyright (c) The ggml authors) as adapted for GLM-5 by antirez
(https://github.com/antirez/llama.cpp-deepseek-v4-flash, MIT):

1. Cost: each weight's squared error is weighted by
   ``importance[k] * sqrt(sigma^2 + w^2)``, sigma^2 being the mean square of
   its 256-weight block, so large weights in quiet channels still count.
2. Group scale: per 32 weights, an initial scale from ``make_qp_quants``, then
   13 candidate scales around it; each assigns every 8-vector its cheapest
   codeword and the candidate with the best least-squares fit wins, refit by
   weighted least squares.
3. Levels: per 256 weights, ``d`` and every group's level are searched
   exhaustively (three exponents for ``d``, all 16 levels per group, with full
   codeword re-assignment under each), minimizing the same weighted cost.
4. Gain: the projection output is scaled so the reconstruction's gain
   ``<W, Wq> / <W, W>`` is ``target_gain`` (0.98 by default).

``iq2r_v2_learn_codebook`` learns the per-matrix codebook with
importance-weighted k-means alternating with this encoder, seeded from the
IQ2_XXS grid (one phase at 1x, one at sqrt(2)), and returns it with the
matrix's encoding.

Codebook entries are normal E4M3 values from 2**-6 to the seed's maximum
(bytes 0x08..0x78 at most; ``IQ2R_V2_CODEBOOK_MIN_BYTE``,
``IQ2R_V2_CODEBOOK_MAX_BYTE``). The gfx950 decode kernels load only copy 0
and form copies 1..C-1 with integer byte arithmetic, which equals
``E4M3(codebook * m[c])`` only in that range. The learner keeps its codebooks
there, and :func:`iq2r_v2_check_codebook` rejects others.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .iq2r_format import (
    IQ2R_ATOM_TWO_METADATA_OFFSET,
    IQ2R_ATOM_TWO_RECORD_BYTES,
    IQ2R_ATOM_TWO_RECORDS_OFFSET,
    IQ2R_ATOMS_PER_TRIPLET,
    IQ2R_CODEBOOK_BYTES,
    IQ2R_CODEBOOK_ENTRIES,
    IQ2R_GROUP_BYTES,
    IQ2R_LANE_RECORD_BYTES,
    IQ2R_N_BLOCKS_PER_GROUP,
    IQ2R_SCALE_BLOCK,
    IQ2R_TILE_N,
    IQ2R_TRIPLET_BYTES,
    IQ2R_VECTOR_SIZE,
    iq2r_data_bytes,
    iq2r_k_tiles,
)

# The IQ2_XXS positive-magnitude grid (``iq2xxs_grid`` in ggml-common.h of
# ggml/llama.cpp, https://github.com/ggml-org/llama.cpp, MIT License,
# Copyright (c) The ggml authors) serialized as 256 eight-byte vectors.  Two
# E4M3 phases seed the 512-entry learned IQ2R codebook.
_IQ2_XXS_GRID_BYTES = base64.b64decode(
    "CAgICAgICAgrCAgICAgICBkZCAgICAgICCsICAgICAgrKwgICAgICBkIGQgICAgICBkZCAgICAgICCsICAgICCsIKwgICAgICCsrCAgICAgrKysICAgICBkICBkICAgICBkIGQgICAgICBkZCAgICAgrGRkICAgIGQgrGQgICAgIGSsZCAgICAgICCsICAgIKwgIKwgICAgrKwgrCAgICCsIKysICAgIGQgICBkICAgIGQgIGQgICAgIGQgZCAgIGRkZCBkICAgICAgZGQgICAgZCCsZCAgICCsZKxkICAgICAgIKwgICCsICAgrCAgIKwgrCCsICAgrCAgrKwgICBkICAgIGQgICBkICAgZCAgICBkICBkICBkIKwgIGQgICBkrCAgZCAgICAgZCBkICCsICBkIGQgICCsIGQgZCAgICCsZCBkICBkICCsIGQgICBkIKwgZCAgICBkrCBkICAgZKysIGQgICAgICBkZCAgrCAgIGRkICAgrCAgZGQgICAgrCBkZCAgrGQgZGRkICBkrKxkZGQgICAgIKxkZCAgZCBkrGRkICBkrCAgrGQgICAgZCCsZCAgICAgZKxkICAgZCCsrGQgICBkrKysZCAgICAgICCsICBkZCAgIKwgICCsICAgrCAgIGRkICCsICAgrKwgIKwgIGQgIGQgrCAgIGQgZCCsICAgIGRkIKwgIKwgZGQgrCAgIKwgrCCsICAgZCAgZKwgICAgIGRkrCAgrCAgIKysICAgZGQgrKwgIGQgICAgIGQgIGQgICAgZCAgIGQgICBkIGQgrCAgIGQgICAgZCAgZCAgIKxkICBkICBkIKwgIGQgICBkrCAgZCBkZGSsICBkICAgICBkIGQgIKwgIGQgZCAgIKwgZCBkICAgZGRkIGQgrKxkZGQgZCAgICCsZCBkICBkrCCsIGQgZGQgZKwgZCAgICAgIGRkICCsICAgZGQgICCsICBkZCBkZKwgIGRkIGSsIGQgZGQgICAgrCBkZCAgrGQgZGRkIKwgrGRkZGQgICAgIKxkZCCsZGQgrGRkIGQgICAgrGQgIGQgICCsZCAgIGQgIKxkICAgIGQgrGQgZCAgrCCsZCAgICAgZKxkIGRkICBkrGQgICCsrGSsZCBkIGRkrKxkICAgICAgIKwgrCAgICAgrCCsrCAgICCsICBkIGQgIKwgZCCsZCAgrCAgICCsICCsIKwgIKwgIKwgZKysIGQgrCAgrCBkZCCsICAgICCsIKwgrCAgIKwgrCBkICAgIGSsICBkICAgZKwgICBkICBkrCAgICBkIGSsIKxkZGQgZKwgICAgIGRkrCBkICBkZGSsICBkrGRkZKwgICBkrKxkrCAgrCAgIKysICAgrCAgrKwgIGRkrCCsrCAgZCBkrKysIGQgICAgICBkIGQgICAgIGQgIGQgICAgZCCsZCAgICBkZCCsICAgIGQgZKwgICAgZCAgIGQgICBkIKwgZCAgIGSsZGRkICAgZCAgrGQgICBkZCAgrCAgIGQgZCCsICAgZCAgZKwgICBkICAgIGQgIGQgIKwgZCAgZGQgrGRkICBkICAgrGQgIGRkZCCsZCAgZGQgICCsICBkICBkIKwgIGQgrCBkrCAgZKxkZGSsICBkIKysZKwgIGQgICAgIGQgZCCsICAgZCBkICCsICBkIGQgICCsIGQgZGSsZKwgZCBkrCBkIGRkIGQgZKwgZGQgZCAgICCsZCBkZCAgICCsIGQgZCAgIKwgZCAgZCAgrCBkICAgZCCsIGRkZCBkIKwgZCAgICBkrCBkIKxkZGSsIGRkIKxkZKwgZKwgIKxkrCBkZGQgZKysIGQgIGSsrKwgZCAgICAgIGRkIKwgICAgZGRkIGQgICBkZGSsZCAgIGRkICCsICAgZGQgICCsICBkZCCsIKwgIGRkIGQgIGQgZGSsICBkZCBkZCBkrKxkIGRkZCBkrKwgZGQgIGSsIGRkZKwgZKwgZGRkrKwgIGRkZGRkICAgrGRkZCBkZGSsZGRkICAgICCsZGRkIGQgIKxkZGSsZCAgrGRkIGSsZCCsZGQgICBkZKxkZCCsICCsrGRkIGQgICAgrGQgIGQgICCsZCAgIGQgIKxkIKysZCAgrGQgICAgZCCsZGRkZGRkIKxkIKxkIKwgrGQgIKxkrCCsZCAgICAgZKxkZGQgICBkrGQgIGQgZGSsZKwgZCBkZKxkIGQgrGRkrGSsICBkIKysZCAgICAgICCsrCAgICAgIKysrCAgICAgrGQgIGQgICCsrCAgrCAgIKwgZCAgZCAgrCCsZCBkICCsICAgZGQgIKxkIGQgrCAgrGQgICAgZCCsIGQgICBkIKwgIGQgIGQgrGRkZCAgZCCsICAgZCBkIKwgIKxkIGQgrCAgICBkZCCsrGQgZGRkIKwgZGSsZGQgrGSsICCsZCCsICAgZKxkIKwgIKxkrGQgrKwgICAgrCCsIGQgIGSsIKxkIGQgrKwgrCBkICAgIGSsICBkICAgZKwgZKwgICBkrCAgIGQgIGSsZCCsrCAgZKysZGQgZCBkrCAgIKxkIGSsZGQgZKwgZKwgICAgIGRkrKwgrCAgZGSsIGQgZCBkZKxkIGRkZGRkrGQgIKwgrGSsICCsIGSsZKysICAgICCsrCAgZGQgIKysZGQgrCAgrKxkrCAgZCCsrCAgICCsIKysIKxkICBkrKwgIGRkIKysrCBkICBkrKys="
)

IQ2R_V2_D_BLOCK = 256
IQ2R_V2_GAIN_BYTES = 8  # two FP32 gains: even rows, odd rows
IQ2R_V2_EPS = 1e-15
# Codebook byte range the kernels' in-LDS copy formation reproduces exactly:
# normal E4M3 values 2**-6 (0x08) to 256 (0x78).
IQ2R_V2_CODEBOOK_MIN_BYTE = 0x08
IQ2R_V2_CODEBOOK_MAX_BYTE = 0x78
IQ2R_V2_CODEBOOK_MIN = 2.0**-6


@dataclass(frozen=True, slots=True)
class IQ2RV2Levels:
    """The 16 per-group levels ``multipliers[l % C] * 2**(l // C)``."""

    name: str
    multipliers: tuple[float, ...]

    @property
    def classes(self) -> int:
        return len(self.multipliers)

    @property
    def octaves(self) -> int:
        return 16 // self.classes

    def level(self, index: int) -> float:
        return self.multipliers[index % self.classes] * 2.0 ** (index // self.classes)


# Four quarter-octave mantissas over four octaves: the level set the kernels run.
IQ2R_V2_LEVELS_C4 = IQ2RV2Levels("c4e4q", tuple(2.0 ** (j / 4.0) for j in range(4)))


def iq2r_v2_format_name(levels: IQ2RV2Levels) -> str:
    return f"iq2r-512-fullsign-{levels.name}-e8m0d256-native-v2"


# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------


def iq2r_v2_d_columns(k: int) -> int:
    if k % IQ2R_V2_D_BLOCK:
        raise ValueError(f"IQ2R-v2 requires K divisible by {IQ2R_V2_D_BLOCK}, got {k}")
    return k // IQ2R_V2_D_BLOCK


# --------------------------------------------------------------------------
# Codebooks
# --------------------------------------------------------------------------


def _e4m3(x: Tensor) -> Tensor:
    return x.to(torch.float8_e4m3fn).float()


def iq2r_v2_seed_codebook(device: torch.device | str = "cpu") -> Tensor:
    """IQ2_XXS grid in native units (grid / 8), at 1x (0-255) and sqrt(2)x (256-511)."""

    grid = torch.tensor(list(_IQ2_XXS_GRID_BYTES), dtype=torch.float32, device=device)
    grid = grid.reshape(256, IQ2R_VECTOR_SIZE) * 0.125
    return _e4m3(torch.cat((grid, grid * math.sqrt(2.0)))).contiguous()


def iq2r_v2_check_codebook(codebook: Tensor) -> None:
    """Raise unless every entry of an E4M3 ``codebook`` (any shape) is a byte in
    ``IQ2R_V2_CODEBOOK_MIN_BYTE..IQ2R_V2_CODEBOOK_MAX_BYTE`` (see the module
    docstring)."""

    raw = codebook.float().to(torch.float8_e4m3fn).view(torch.uint8)
    low, high = int(raw.min()), int(raw.max())
    if low < IQ2R_V2_CODEBOOK_MIN_BYTE or high > IQ2R_V2_CODEBOOK_MAX_BYTE:
        raise ValueError(
            f"IQ2R-v2 codebook bytes span 0x{low:02x}..0x{high:02x}; the kernels "
            f"need 0x{IQ2R_V2_CODEBOOK_MIN_BYTE:02x}..0x{IQ2R_V2_CODEBOOK_MAX_BYTE:02x}"
        )


def iq2r_v2_codebook_copies(codebook: Tensor, levels: IQ2RV2Levels) -> Tensor:
    """``[C, ..., 512, 8]`` FP32 copies ``E4M3(codebook * m[c])``, the values the kernels read.

    Each product is formed in FP32 and rounded to E4M3 (round to nearest even),
    the same two steps as the kernels' conversion.
    """

    book = codebook.float()
    multipliers = torch.tensor(
        levels.multipliers, dtype=torch.float32, device=book.device
    )
    return _e4m3(book[None] * multipliers.view(-1, *([1] * book.dim())))


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------


def _cost_weights(blocks: Tensor, importance: Tensor) -> Tensor:
    sigma2 = blocks.square().mean(-1, keepdim=True)
    return importance * torch.sqrt(sigma2 + blocks.square())


def _make_qp_quants(x: Tensor, w: Tensor, nmax: int = 4) -> Tensor:
    """ggml's ``make_qp_quants`` per row of ``x`` (non-negative): least-squares scale."""

    mx = x.amax(-1)
    ok = mx >= IQ2R_V2_EPS
    mxs = torch.where(ok, mx, torch.ones_like(mx))
    iscale = nmax / mxs
    best = (
        w * (x - (1 / iscale)[:, None] * torch.round(iscale[:, None] * x)).square()
    ).sum(-1)
    for step in range(-4, 5):
        if step == 0:
            continue
        candidate = (0.1 * step + nmax) / mxs
        q = torch.round(candidate[:, None] * x).clamp(max=nmax)
        mse = (w * (x - (1 / candidate)[:, None] * q).square()).sum(-1)
        better = mse < best
        best = torch.where(better, mse, best)
        iscale = torch.where(better, candidate, iscale)
    q = torch.round(iscale[:, None] * x).clamp(max=nmax)
    sumlx = (w * x * q).sum(-1)
    suml2 = (w * q * q).sum(-1)
    for _ in range(5):
        for i in range(x.shape[1]):
            wi, xi, qi = w[:, i], x[:, i], q[:, i]
            slx = sumlx - wi * xi * qi
            sl2 = suml2 - wi * qi * qi
            valid = (slx > 0) & (sl2 > 0)
            nq = torch.round(
                xi * sl2 / torch.where(valid, slx, torch.ones_like(slx))
            ).clamp(max=nmax)
            slx2 = slx + wi * xi * nq
            sl22 = sl2 + wi * nq * nq
            accept = valid & (nq != qi) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
            q[:, i] = torch.where(accept, nq, qi)
            sumlx = torch.where(accept, slx2, sumlx)
            suml2 = torch.where(accept, sl22, suml2)
    positive = suml2 > 0
    scale = torch.where(
        positive, sumlx / torch.where(positive, suml2, torch.ones_like(suml2)), 0.0
    )
    return torch.where(ok, scale, torch.zeros_like(scale))


def _codeword_scores(
    vectors: Tensor, weights: Tensor, codebook: Tensor
) -> tuple[Tensor, Tensor]:
    """``A[v, c] = sum_j w c^2`` and ``B[v, c] = sum_j w x c``: cost(s) = s^2 A - 2 s B + const."""

    return weights @ codebook.square().t(), (weights * vectors) @ codebook.t()


def _group_scales(
    magnitudes: Tensor, weights: Tensor, codebook: Tensor, candidates: int = 6
) -> Tensor:
    """Per 32-weight group: the least-squares scale of the best of 13 candidates (steps 1-2)."""

    groups = magnitudes.reshape(-1, IQ2R_SCALE_BLOCK)
    group_weights = weights.reshape(-1, IQ2R_SCALE_BLOCK)
    zero = groups.amax(-1) < IQ2R_V2_EPS
    effective_max = 3.0 * _make_qp_quants(groups, group_weights)
    zero |= effective_max <= 0
    a, b = _codeword_scores(
        magnitudes.reshape(-1, IQ2R_VECTOR_SIZE),
        weights.reshape(-1, IQ2R_VECTOR_SIZE),
        codebook,
    )
    b2 = 2 * b

    def fit(scale: Tensor) -> tuple[Tensor, Tensor]:
        index = torch.argmin(scale.repeat_interleave(4)[:, None] * a - b2, dim=-1)
        q2 = a.gather(1, index[:, None]).reshape(-1, 4).sum(-1)
        qx = b.gather(1, index[:, None]).reshape(-1, 4).sum(-1)
        return q2, qx

    best = torch.zeros_like(effective_max)
    best_scale = torch.zeros_like(effective_max)
    for step in range(-candidates, candidates + 1):
        # antirez's inverse-scale steps around 5 grid units (40 native units / 8).
        scale = torch.where(zero, 0.0, effective_max / (5.0 + 0.1 * step))
        q2, qx = fit(scale)
        safe = torch.where(q2 > 0, q2, torch.ones_like(q2))
        criterion = torch.where(q2 > 0, qx * qx / safe, 0.0)
        better = (q2 > 0) & (criterion > best)
        best = torch.where(better, criterion, best)
        best_scale = torch.where(better, qx / safe, best_scale)
    q2, qx = fit(torch.where(zero, 0.0, best_scale))
    refit = torch.where(
        (q2 > 0) & ~zero, qx / torch.where(q2 > 0, q2, torch.ones_like(q2)), 0.0
    )
    return refit.clamp_min(0.0)


@dataclass
class IQ2RV2Encoding:
    """IQ2R-v2 encoding of whole 256-weight blocks (row-major logical order)."""

    indices: Tensor  # [..., 32 per block] int64
    negative: Tensor  # [..., 256 per block] bool
    levels: Tensor  # [..., 8 per block] uint8
    d: Tensor  # [..., 1 per block] uint8, E8M0
    reconstruction: Tensor  # [..., 256 per block] fp32, before the gain
    cost: float


@torch.no_grad()
def _encode_blocks(
    blocks: Tensor,
    block_importance: Tensor,
    codebook: Tensor,
    levels: IQ2RV2Levels,
    chunk_blocks: int,
) -> IQ2RV2Encoding:
    """Encode ``blocks [B, 256]`` with per-weight importance ``block_importance [B, 256]``."""

    device = blocks.device
    copies = iq2r_v2_codebook_copies(codebook, levels)
    classes, octaves = levels.classes, levels.octaves
    nominal_max = levels.level(15)
    total_blocks = blocks.shape[0]
    indices = torch.empty(total_blocks, 32, dtype=torch.int64, device=device)
    level_out = torch.empty(total_blocks, 8, dtype=torch.uint8, device=device)
    d_out = torch.empty(total_blocks, dtype=torch.uint8, device=device)
    reconstruction = torch.empty_like(blocks, dtype=torch.float32)
    cost = 0.0
    for start in range(0, total_blocks, chunk_blocks):
        stop = min(total_blocks, start + chunk_blocks)
        x = blocks[start:stop].float()
        weights = _cost_weights(x, block_importance[start:stop].float())
        magnitudes = x.abs()
        rows = stop - start
        group_scale = _group_scales(magnitudes, weights, copies[0]).reshape(rows, 8)
        vectors = magnitudes.reshape(-1, IQ2R_VECTOR_SIZE)
        vector_weights = weights.reshape(-1, IQ2R_VECTOR_SIZE)
        scores = [
            (a, 2 * b)
            for a, b in (
                _codeword_scores(vectors, vector_weights, copies[c])
                for c in range(classes)
            )
        ]
        # d: the E8M0 exponent putting the largest group at the top level, and its neighbours.
        top = (group_scale.amax(-1) / nominal_max).clamp_min(2.0**-126)
        centre = torch.floor(torch.log2(top))
        best_cost = torch.full((rows,), math.inf, device=device)
        best_d = torch.zeros(rows, device=device)
        best_level = torch.zeros(rows, 8, dtype=torch.uint8, device=device)
        best_index = torch.zeros(rows * 32, dtype=torch.int64, device=device)
        best_recon = torch.zeros(rows, IQ2R_V2_D_BLOCK, device=device)
        for offset in (-1.0, 0.0, 1.0):
            exponent = (centre + offset).clamp(-127, 127 - (octaves - 1))
            group_cost = torch.full((rows, 8), math.inf, device=device)
            group_level = torch.zeros(rows, 8, dtype=torch.uint8, device=device)
            group_index = torch.zeros(rows * 32, dtype=torch.int64, device=device)
            group_recon = torch.zeros(rows, 8, IQ2R_SCALE_BLOCK, device=device)
            for level in range(16):
                a, b2 = scores[level % classes]
                scale = torch.exp2(exponent + level // classes).repeat_interleave(32)
                index = torch.argmin(scale[:, None] * a - b2, dim=-1)
                recon = (copies[level % classes][index] * scale[:, None]).reshape(
                    rows, 8, IQ2R_SCALE_BLOCK
                )
                err = (
                    weights.reshape(rows, 8, -1)
                    * (recon - magnitudes.reshape(rows, 8, -1)).square()
                ).sum(-1)
                better = err < group_cost
                group_cost = torch.where(better, err, group_cost)
                group_level = torch.where(better, level, group_level)
                group_recon = torch.where(better[..., None], recon, group_recon)
                group_index = torch.where(
                    better.repeat_interleave(4, dim=1).reshape(-1), index, group_index
                )
            total = group_cost.sum(-1)
            better = total < best_cost
            best_cost = torch.where(better, total, best_cost)
            best_d = torch.where(better, exponent, best_d)
            best_level = torch.where(better[:, None], group_level, best_level)
            best_recon = torch.where(
                better[:, None], group_recon.reshape(rows, -1), best_recon
            )
            best_index = torch.where(
                better.repeat_interleave(32), group_index, best_index
            )
        indices[start:stop] = best_index.reshape(rows, 32)
        level_out[start:stop] = best_level
        d_out[start:stop] = (best_d + 127).to(torch.uint8)
        reconstruction[start:stop] = torch.where(x < 0, -best_recon, best_recon)
        cost += float(best_cost.sum(dtype=torch.float64))
    return IQ2RV2Encoding(indices, blocks < 0, level_out, d_out, reconstruction, cost)


@torch.no_grad()
def iq2r_v2_encode(
    weight: Tensor,
    importance: Tensor,
    codebook: Tensor,
    levels: IQ2RV2Levels = IQ2R_V2_LEVELS_C4,
    *,
    chunk_blocks: int = 4096,
) -> IQ2RV2Encoding:
    """Encode ``weight [N, K]`` (K % 256 == 0) with a native ``[512, 8]`` E4M3 codebook.

    Returns indices ``[N, K/8]``, negative ``[N, K]``, levels ``[N, K/32]``,
    d ``[N, K/256]`` and the reconstruction ``[N, K]`` (before the gain).
    """

    if weight.ndim != 2 or importance.shape != (weight.shape[1],):
        raise ValueError("weight and importance must have shapes [N,K] and [K]")
    n, k = weight.shape
    columns = iq2r_v2_d_columns(k)
    blocks = weight.float().reshape(-1, IQ2R_V2_D_BLOCK)
    block_importance = (
        importance.float()
        .reshape(1, columns, IQ2R_V2_D_BLOCK)
        .expand(n, -1, -1)
        .reshape(-1, IQ2R_V2_D_BLOCK)
    )
    enc = _encode_blocks(blocks, block_importance, codebook, levels, chunk_blocks)
    return IQ2RV2Encoding(
        indices=enc.indices.reshape(n, k // IQ2R_VECTOR_SIZE),
        negative=enc.negative.reshape(n, k),
        levels=enc.levels.reshape(n, k // IQ2R_SCALE_BLOCK),
        d=enc.d.reshape(n, columns),
        reconstruction=enc.reconstruction.reshape(n, k),
        cost=enc.cost,
    )


def iq2r_v2_gains(
    weight: Tensor,
    reconstruction: Tensor,
    target_gain: float = 0.98,
    *,
    interleaved: bool = False,
) -> tuple[float, float]:
    """Output gains (even rows, odd rows) giving ``<W, g*Wq> / <W, W> = target_gain``.

    ``interleaved`` (the gate0, up0, gate1, ... matrix) fits gate and up rows separately.
    """

    def gain(w: Tensor, q: Tensor) -> float:
        w, q = w.double(), q.double()
        return target_gain * float(w.square().sum() / (q * w).sum().clamp_min(1e-30))

    if not interleaved:
        g = gain(weight, reconstruction)
        return g, g
    return gain(weight[0::2], reconstruction[0::2]), gain(
        weight[1::2], reconstruction[1::2]
    )


# --------------------------------------------------------------------------
# Codebook learning
# --------------------------------------------------------------------------


@torch.no_grad()
def iq2r_v2_learn_codebook(
    weight: Tensor,
    importance: Tensor,
    levels: IQ2RV2Levels = IQ2R_V2_LEVELS_C4,
    *,
    rounds: int = 4,
    inner: int = 4,
    sample_vectors: int = 65536,
    seed: int = 0,
    chunk_blocks: int = 4096,
) -> tuple[Tensor, IQ2RV2Encoding]:
    """Learn one ``[512, 8]`` native E4M3 codebook for ``weight [N, K]``.

    Importance-weighted k-means on ``sample_vectors / 32`` random 256-weight
    blocks, alternating with the encoder. Each round encodes the sample, then
    runs ``inner`` Lloyd steps at its fixed per-group scales t (centroid =
    cost-weighted mean of t * |x| over t^2, so the update minimizes the
    encoder's own cost). Codewords stay within the seed's maximum, which keeps
    the scale search's reference point, and at or above
    ``IQ2R_V2_CODEBOOK_MIN`` (see the module docstring), and a round is kept
    only if the weighted cost of the whole matrix drops.

    Returns the codebook and ``weight``'s encoding with it, which the last kept
    round has already computed.
    """

    k = weight.shape[1]
    device = weight.device
    columns = iq2r_v2_d_columns(k)
    book = iq2r_v2_seed_codebook(device)
    pin = float(book.max())
    blocks = weight.float().reshape(-1, IQ2R_V2_D_BLOCK)
    block_importance = importance.float().reshape(columns, IQ2R_V2_D_BLOCK)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = min(blocks.shape[0], max(1, sample_vectors // 32))
    pick = torch.randperm(blocks.shape[0], generator=generator)[:count].to(device)
    sample = blocks[pick]
    sample_importance = block_importance[pick % columns]
    vectors = sample.abs().reshape(-1, IQ2R_VECTOR_SIZE)
    weights = _cost_weights(sample, sample_importance).reshape(-1, IQ2R_VECTOR_SIZE)
    multipliers = torch.tensor(levels.multipliers, device=device)

    def matrix_encode(b: Tensor) -> IQ2RV2Encoding:
        return iq2r_v2_encode(weight, importance, b, levels, chunk_blocks=chunk_blocks)

    best = matrix_encode(book)
    for _ in range(rounds):
        enc = _encode_blocks(sample, sample_importance, book, levels, chunk_blocks)
        level = enc.levels.long()
        # Scale each vector's plain codeword sees: 2^(d-127) * 2^(l//C) * m[l%C].
        t = (
            torch.exp2(enc.d.float()[:, None] - 127 + (level // levels.classes).float())
            * multipliers[level % levels.classes]
        )
        t = t.repeat_interleave(4, dim=1).reshape(-1)
        index = enc.indices.reshape(-1)
        candidate = book
        for step in range(inner):
            if step:
                a, b = _codeword_scores(vectors, weights, candidate)
                index = torch.argmin(t[:, None] * a - 2 * b, dim=-1)
            numerator = torch.zeros_like(candidate).index_add_(
                0, index, weights * t[:, None] * vectors
            )
            denominator = torch.zeros_like(candidate).index_add_(
                0, index, weights * t[:, None].square()
            )
            occupied = denominator > 0
            candidate = torch.where(
                occupied, numerator / torch.where(occupied, denominator, 1.0), candidate
            )
        candidate = _e4m3(candidate.clamp(IQ2R_V2_CODEBOOK_MIN, pin))
        encoding = matrix_encode(candidate)
        if encoding.cost >= best.cost:
            break
        book, best = candidate, encoding
    return book.contiguous(), best


# --------------------------------------------------------------------------
# Canonical records
# --------------------------------------------------------------------------
#
# Per 16-row N block and 128-wide K tile: 64 lanes of four 9-bit indices and
# 32 sign bits, plus a metadata byte per lane and atom whose low nibble holds
# the indices' ninth bits. Its high nibble holds
# ``exponent(b) * C + class(s(b))`` in lane group b, s = (0, 2, 1, 3): the
# lane that supplies block b's MFMA scale keeps b's exponent, and the class of
# the block a lane decodes codewords for sits in that lane or its neighbour
# group, one permlane16 swap away (see iq2r_levels in the kernels).

_PARTNER = (0, 2, 1, 3)


def _vector_offsets(n: int, k: int, device) -> tuple[Tensor, Tensor, Tensor]:
    """Byte offsets of every 8-vector's index byte and metadata byte, and its metadata bit.

    Shapes ``[N, K/8]``. A vector of row r and columns [8v, 8v+8) sits in lane
    ``g*16 + r%16`` as codeword ``j``, where its 32-block b and position p in the
    block give ``g = 2*(b%2) + p//2`` and ``j = 2*(b//2) + p%2``.
    """

    k_tiles = iq2r_k_tiles(k)
    row = torch.arange(n, device=device)[:, None]
    vec = torch.arange(k // IQ2R_VECTOR_SIZE, device=device)[None, :]
    n_block, rr = row // IQ2R_TILE_N, row % IQ2R_TILE_N
    k_tile, within = vec // 16, vec % 16
    block, position = within // 4, within % 4
    group = 2 * (block % 2) + position // 2
    word = 2 * (block // 2) + position % 2
    lane = group * 16 + rr
    tile = (
        (n_block // IQ2R_N_BLOCKS_PER_GROUP) * k_tiles + k_tile
    ) * IQ2R_N_BLOCKS_PER_GROUP + (n_block % IQ2R_N_BLOCKS_PER_GROUP)
    atom = (tile % IQ2R_N_BLOCKS_PER_GROUP) % IQ2R_ATOMS_PER_TRIPLET
    base = (tile // IQ2R_N_BLOCKS_PER_GROUP) * IQ2R_GROUP_BYTES + (
        (tile % IQ2R_N_BLOCKS_PER_GROUP) // IQ2R_ATOMS_PER_TRIPLET
    ) * IQ2R_TRIPLET_BYTES
    record = torch.where(
        atom < 2,
        base + lane * 2 * IQ2R_LANE_RECORD_BYTES + atom * IQ2R_LANE_RECORD_BYTES,
        base + IQ2R_ATOM_TWO_RECORDS_OFFSET + lane * IQ2R_ATOM_TWO_RECORD_BYTES,
    )
    metadata = (
        base
        + IQ2R_ATOM_TWO_RECORDS_OFFSET
        + lane * IQ2R_ATOM_TWO_RECORD_BYTES
        + (IQ2R_ATOM_TWO_METADATA_OFFSET + atom)
    )
    return record + word, metadata, word


def _level_offsets(n: int, k: int, device) -> Tensor:
    """``[N, K/32]`` byte offsets of each 32-group's level nibble (lane ``b*16 + r%16``)."""

    k_tiles = iq2r_k_tiles(k)
    row = torch.arange(n, device=device)[:, None]
    blk = torch.arange(k // IQ2R_SCALE_BLOCK, device=device)[None, :]
    n_block, rr = row // IQ2R_TILE_N, row % IQ2R_TILE_N
    k_tile, b = blk // 4, blk % 4
    lane = b * 16 + rr
    tile = (
        (n_block // IQ2R_N_BLOCKS_PER_GROUP) * k_tiles + k_tile
    ) * IQ2R_N_BLOCKS_PER_GROUP + (n_block % IQ2R_N_BLOCKS_PER_GROUP)
    atom = (tile % IQ2R_N_BLOCKS_PER_GROUP) % IQ2R_ATOMS_PER_TRIPLET
    base = (tile // IQ2R_N_BLOCKS_PER_GROUP) * IQ2R_GROUP_BYTES + (
        (tile % IQ2R_N_BLOCKS_PER_GROUP) // IQ2R_ATOMS_PER_TRIPLET
    ) * IQ2R_TRIPLET_BYTES
    return (
        base
        + IQ2R_ATOM_TWO_RECORDS_OFFSET
        + lane * IQ2R_ATOM_TWO_RECORD_BYTES
        + (IQ2R_ATOM_TWO_METADATA_OFFSET + atom)
    )


def _nibbles(levels: Tensor, classes: int) -> Tensor:
    """Levels ``[N, K/32]`` -> stored nibbles (exponent of b, class of s(b)) per 128-wide tile."""

    n = levels.shape[0]
    tiles = levels.to(torch.int64).reshape(n, -1, 4)
    partner = tiles[..., list(_PARTNER)]
    return ((tiles // classes) * classes + partner % classes).reshape(n, -1)


def iq2r_v2_write_data(
    indices: Tensor, negative: Tensor, levels: Tensor, level_set: IQ2RV2Levels
) -> Tensor:
    """Canonical data bytes of one matrix from ``indices [N, K/8]``, ``negative [N, K]``, ``levels [N, K/32]``."""

    n, k = negative.shape
    device = indices.device
    data = torch.zeros(iq2r_data_bytes(n, k), dtype=torch.int64, device=device)
    index_at, metadata_at, word = _vector_offsets(n, k, device)
    data.index_put_((index_at.flatten(),), (indices & 255).flatten())
    bits = negative.reshape(n, -1, IQ2R_VECTOR_SIZE).to(torch.int64)
    signs = (bits << torch.arange(IQ2R_VECTOR_SIZE, device=device)).sum(-1)
    data.index_put_(((index_at + 4).flatten(),), signs.flatten())
    # Ninth index bits (low nibble) and levels (high nibble) of the metadata bytes.
    data.index_put_(
        (metadata_at.flatten(),),
        (((indices >> 8) & 1) << word).flatten(),
        accumulate=True,
    )
    nibbles = _nibbles(levels, level_set.classes)
    data.index_put_(
        (_level_offsets(n, k, device).flatten(),),
        (nibbles << 4).flatten(),
        accumulate=True,
    )
    return data.to(torch.uint8)


def iq2r_v2_write_auxiliary(
    codebooks: Tensor, d: Tensor, gains: tuple[float, float]
) -> Tensor:
    """Canonical auxiliary bytes: ``codebooks [B, 512, 8]`` E4M3, ``d [N, K/256]``, two FP32 gains."""

    books = codebooks.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)
    gain_bytes = torch.tensor(gains, dtype=torch.float32).view(torch.uint8).to(d.device)
    return torch.cat((books.to(d.device), d.reshape(-1).to(torch.uint8), gain_bytes))


def iq2r_v2_read(
    data: Tensor,
    auxiliary: Tensor,
    n: int,
    k: int,
    level_set: IQ2RV2Levels,
    books: int = 1,
):
    """Inverse of the writers: ``(indices, negative, levels, codebooks, d, gains)``."""

    data = data.to(torch.int64)
    device = data.device
    index_at, metadata_at, word = _vector_offsets(n, k, device)
    indices = data[index_at] | (((data[metadata_at] >> word) & 1) << 8)
    signs = data[index_at + 4]
    negative = (
        ((signs[..., None] >> torch.arange(IQ2R_VECTOR_SIZE, device=device)) & 1)
        .bool()
        .reshape(n, k)
    )
    nibbles = (data[_level_offsets(n, k, device)] >> 4).reshape(n, -1, 4)
    classes = level_set.classes
    # s is its own inverse: class(b) is stored in group s(b).
    levels = (
        (nibbles // classes) * classes + nibbles[..., list(_PARTNER)] % classes
    ).reshape(n, -1)
    levels = levels.to(torch.uint8)
    book_bytes = books * IQ2R_CODEBOOK_BYTES
    codebooks = (
        auxiliary[:book_bytes]
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(books, IQ2R_CODEBOOK_ENTRIES, 8)
    )
    d = auxiliary[book_bytes : book_bytes + n * iq2r_v2_d_columns(k)].reshape(n, -1)
    gains = tuple(
        auxiliary[-IQ2R_V2_GAIN_BYTES:].contiguous().view(torch.float32).tolist()
    )
    return indices, negative, levels, codebooks, d, gains


@torch.no_grad()
def iq2r_v2_materialize(
    data: Tensor,
    auxiliary: Tensor,
    n: int,
    k: int,
    levels: IQ2RV2Levels,
    books: int = 1,
    *,
    apply_gain: bool = True,
) -> Tensor:
    """Reference decode of one canonical IQ2R-v2 matrix to FP32 ``[N, K]``.

    With two codebooks (the interleaved gate/up matrix) even rows use book 0 and
    odd rows book 1; the gains likewise.
    """

    indices, negative, level, codebooks, d, gains = iq2r_v2_read(
        data, auxiliary, n, k, levels, books
    )
    copies = torch.stack(
        [iq2r_v2_codebook_copies(codebooks[b], levels) for b in range(books)]
    )
    level = level.long()
    row_book = (torch.arange(n, device=data.device) % books)[:, None]
    group_class = level % levels.classes  # [N, K/32]
    vector_class = group_class.repeat_interleave(4, dim=1)
    codes = copies[row_book, vector_class, indices]  # [N, K/8, 8]
    exponent = (
        d.float().repeat_interleave(8, dim=1) - 127 + (level // levels.classes).float()
    )
    scale = torch.exp2(exponent).repeat_interleave(4, dim=1)[..., None]
    values = (codes * scale).reshape(n, k)
    values = torch.where(negative, -values, values)
    if apply_gain:
        gain = torch.tensor(gains, dtype=torch.float32, device=data.device)
        values = values * gain[torch.arange(n, device=data.device) % 2][:, None]
    return values
