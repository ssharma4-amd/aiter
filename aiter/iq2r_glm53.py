# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 IQ2R-v2 MoE on the packed layout (TP4 and TP8, gfx950).

256 routed experts with top-8 routing; the shared expert stays FP8 outside the
MoE. The packed layout is a byte-exact relayout of the canonical IQ2R-v2
records (``aiter.ops.iq2r_v2``):

* the four 9-bit (or 10-bit) indices of an atom pack as low32 + high4
  (+ high4 in a per-lane extra word for a 1,024-entry codebook);
* gate/up record signs transpose into four 8-bit planes, and the gate/up
  records are regrouped into quads of four 16-column blocks;
* the auxiliary holds the codebook copies, a per-lane d table and the gains.

Every transform is per record, so packing a full checkpoint and then taking a
tensor-parallel shard (``iq2r_glm53_slice_gate``, ``iq2r_glm53_v2_slice_gate_aux``,
``iq2r_slice_input_data``, ``iq2r_glm53_v2_slice_down_aux``) equals packing
that shard. ``iq2r_glm53_unpack_v2`` inverts the packing.

Launch choices per token count and codebook size come from
``aiter/configs/iq2r_glm53_tuned.csv`` (override with ``AITER_CONFIG_IQ2R_GLM53``).
Other token counts use the next tuned row up that can run them, else the
defaults in :func:`_default_config`; a codebook size without rows uses the
512-entry rows.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch import Tensor

from aiter import logger

from .jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG
from .jit.utils.chip_info import get_cu_num
from .jit.utils.chip_info import get_gfx_runtime as get_gfx
from .ops.iq2r_format import iq2r_data_bytes
from .ops.iq2r import (
    iq2r_glm53_down_out,
    iq2r_glm53_down_reduce_out,
    iq2r_glm53_down_route9_out,
    iq2r_glm53_gate_m1_out,
    iq2r_glm53_gate_out,
    iq2r_glm53_m1_route_quant_out,
    iq2r_glm53_prefill_sort_quant_out,
    iq2r_glm53_route_reduce_out,
    iq2r_glm53_sort_quant_out,
)

if TYPE_CHECKING:
    from .ops.iq2r_v2 import IQ2RV2Levels

HIDDEN = 6144
EXPERTS = 256
TOPK = 8
ROUTE_SLOTS = 9  # routes per token in the kernels' layout; slot 8 is empty
MAX_DECODE_TOKENS = 1024  # decode kernels cover M2..1024 (tuned rows may pick them)
DEFAULT_DECODE_TOKENS = 256  # untuned shapes switch to the prefill kernels above this
MAX_CHUNK_TOKENS = 4096
INTERMEDIATE_SIZES = (256, 512)  # TP8, TP4

# Stored slabs (aiter/ops/iq2r_format.py): K tiles per slab and slab bytes of
# a six-block group, for 512- and 1,024-entry codebooks. A packed gate/up quad
# (4 x 16 rows) holds 2304 bytes per K tile, plus for 1,024 entries a
# 12-byte word per lane and six-tile K slice (one decode-gate wave's).
_SLABS = {512: (1, 3584), 1024: (2, 7296)}
_QUAD_TILE_BYTES = {512: 2304, 1024: 2432}
_QUAD_SLICE_TILES = 6

GATE_KERNELS = {"nobarrier": 1, "prefill": 2}
# "single2": the TP4 single down built for two workgroups per CU.
DOWN_KERNELS = {"ordered": 1, "single": 2, "single2": 3}


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------
#
# Per atom (16 rows x 128 columns) and lane, the four indices pack as one
# integer ``i0 | i1 << B | i2 << 2B | i3 << 3B`` (B = 9 or 10 bits): its low 32
# bits replace the four index bytes, bits 32-35 the metadata byte's low nibble
# (which held the ninth bits) and, for 10-bit indices, bits 36-39 the atom's
# tenth-bit nibble (atoms 0-1: the atom-two record's fourth metadata byte;
# atom 2: the slab's byte of the lane).


def _entries(codebook_entries: int) -> int:
    if codebook_entries not in _SLABS:
        raise ValueError(f"IQ2R codebooks have 512 or 1024 entries, got {codebook_entries}")
    return codebook_entries


def _slabs(data: Tensor, logical_k: int, entries: int):
    """``(records [E, G, slabs, tiles per slab, 3584], ext [E, G, slabs, 2, 64] or None)``
    views of canonical (or index-packed) records; ``ext`` holds atom 2's tenth
    bits per triplet and lane."""
    slab_tiles, slab_bytes = _SLABS[entries]
    experts = data.shape[0]
    k_tiles = logical_k // 128
    if k_tiles % slab_tiles or data.shape[1] % (k_tiles // slab_tiles * slab_bytes):
        raise ValueError("unexpected IQ2R record layout")
    s = data.view(experts, -1, k_tiles // slab_tiles, slab_bytes)
    records = s[..., : slab_tiles * 3584].unflatten(-1, (slab_tiles, 3584))
    ext = None
    if entries == 1024:
        ext = s[..., slab_tiles * 3584 :].unflatten(-1, (2, 64))
    return records, ext


def _tenth(third: Tensor, ext: Tensor | None, tile: int, atom: int):
    """``(byte view [..., 64, 1], shift)`` of an atom's tenth-bit nibble, or None:
    atoms 0-1 in the atom-two record's fourth metadata byte, atom 2 in the
    slab's byte (``ext`` [..., 64] of the triplet)."""
    if ext is None:
        return None
    if atom < 2:
        return third[..., 11:12], 4 * atom
    return ext[..., None], 4 * tile


def _atoms(data: Tensor, logical_k: int, entries: int):
    """Yield (record, third, tenth, atom) per 8-byte atom record: ``tenth`` is the
    ``_tenth`` view of its tenth bits (None for 512 entries). All views share
    leading dims [E, G, slabs]."""
    records, ext = _slabs(data, logical_k, entries)
    for tile in range(records.shape[3]):
        for triplet in range(2):
            offset = triplet * 1792
            pair = records[:, :, :, tile, offset : offset + 1024].unflatten(-1, (64, 16))
            third = records[:, :, :, tile, offset + 1024 : offset + 1792].unflatten(
                -1, (64, 12)
            )
            bytes_ = ext[:, :, :, triplet] if ext is not None else None
            for atom in range(3):
                record = pair[..., atom * 8 : atom * 8 + 8] if atom < 2 else third[..., :8]
                yield record, third, _tenth(third, bytes_, tile, atom), atom


def _word(raw: Tensor) -> Tensor:
    """Little-endian integer of the last dim's bytes."""
    return sum(raw[..., i].to(torch.int64) << (8 * i) for i in range(raw.shape[-1]))


def _set_nibble(raw: Tensor, shift: int, nibble: Tensor) -> None:
    word = (_word(raw) & ~(15 << shift)) | (nibble << shift)
    for i in range(raw.shape[-1]):
        raw[..., i].copy_(((word >> (8 * i)) & 255).to(torch.uint8))


# The record transforms below rewrite ``data`` in place: each atom's new bits
# depend only on that atom's old bits, which are read first.


def _pack_indices(data: Tensor, logical_k: int, entries: int) -> None:
    bits = 10 if entries == 1024 else 9
    for record, third, tenth_at, atom in _atoms(data, logical_k, entries):
        metadata = third[..., 8 + atom].to(torch.int64)
        tenth = (_word(tenth_at[0]) >> tenth_at[1]) & 15 if tenth_at else None
        packed = torch.zeros_like(metadata)
        for word in range(4):
            index = record[..., word].to(torch.int64) | (((metadata >> word) & 1) << 8)
            if tenth is not None:
                index |= ((tenth >> word) & 1) << 9
            packed |= index << (word * bits)
        for byte in range(4):
            record[..., byte].copy_(((packed >> (8 * byte)) & 255).to(torch.uint8))
        third[..., 8 + atom].copy_(
            ((metadata & 240) | ((packed >> 32) & 15)).to(torch.uint8)
        )
        if tenth_at:
            _set_nibble(*tenth_at, (packed >> 36) & 15)


def _transpose_signs(data: Tensor, logical_k: int, entries: int) -> None:
    for record, *_ in _atoms(data, logical_k, entries):
        old = sum(record[..., 4 + j].to(torch.int64) << (8 * j) for j in range(4))
        new = torch.zeros_like(old)
        for bit in range(32):
            new |= ((old >> bit) & 1) << ((bit % 4) * 8 + bit // 4)
        for j in range(4):
            record[..., 4 + j].copy_(((new >> (8 * j)) & 255).to(torch.uint8))


def _quad_tiles(quad: Tensor, entries: int):
    """``(tiles [E, N/64, 48, 2304], ext [E, N/64, 8, 64, 12] or None)`` views of a
    packed gate/up stack: 1,024-entry quads store each six-tile K slice's tiles,
    then its lanes' 12-byte tenth-bit words (slice tile t's four quad nibbles in
    bits 16 t .. 16 t + 15, so dword t // 2 holds tiles 2 (t // 2) and + 1)."""
    experts = quad.shape[0]
    if entries == 512:
        return quad.view(experts, -1, 48, 2304), None
    slices = quad.view(experts, -1, 48 // _QUAD_SLICE_TILES, _QUAD_SLICE_TILES * 2432)
    tiles = slices[..., : _QUAD_SLICE_TILES * 2304].unflatten(-1, (_QUAD_SLICE_TILES, 2304))
    return tiles, slices[..., _QUAD_SLICE_TILES * 2304 :].unflatten(-1, (64, 12))


def _quad_map(data: Tensor, quad: Tensor, logical_n: int, entries: int, to_quad: bool):
    """Copy gate/up records between the canonical K=6144 records and quads of four
    16-column blocks (``to_quad``), or back. Every operand is a view, so the
    copies land in ``data`` or ``quad``."""
    records, ext = _slabs(data, HIDDEN, entries)
    tiles, quad_ext = _quad_tiles(quad, entries)
    slab_tiles = records.shape[3]
    for block in range(logical_n // 16):
        atom = block % 3
        offset = (block % 6) // 3 * 1792
        slot = block % 4
        for tile in range(slab_tiles):
            source = records[:, block // 6, :, tile]  # [E, slabs, 3584]
            if entries == 512:
                target = tiles[:, block // 4]  # [E, 48 K tiles, 2304]
            else:
                # K tile 2 s + tile: six-tile slice s // 3, position 2 (s % 3) + tile
                source = source.unflatten(1, (8, 3))
                target = tiles[:, block // 4, :, tile::2]  # [E, 8, 3, 2304]
            third = source[..., offset + 1024 : offset + 1792].unflatten(-1, (64, 12))
            if atom < 2:
                record = source[..., offset : offset + 1024].unflatten(-1, (64, 16))
                record = record[..., atom * 8 : atom * 8 + 8]
            else:
                record = third[..., :8]
            pair_base = (slot // 2) * 1024
            pair = target[..., pair_base : pair_base + 1024].unflatten(-1, (64, 16))
            pair = pair[..., slot % 2 * 8 : slot % 2 * 8 + 8]
            metadata = target[..., 2048:2304].unflatten(-1, (64, 4))[..., slot]
            if to_quad:
                pair.copy_(record)
                metadata.copy_(third[..., 8 + atom])
            else:
                record.copy_(pair)
                third[..., 8 + atom].copy_(metadata)
            if ext is None:
                continue
            # [E, 8 slices, 3 slabs, 64]
            slab_bytes = ext[:, block // 6, :, (block % 6) // 3].unflatten(1, (8, 3))
            for position in range(3):
                raw, shift = _tenth(third[:, :, position], slab_bytes[:, :, position], tile, atom)
                # [E, 8, 64, 12]: K tile 2 position + tile's 16 bits are in
                # dword position, at bit 16 tile.
                quad_words = quad_ext[:, block // 4, ..., 4 * position : 4 * position + 4]
                quad_shift = 16 * tile + 4 * slot
                if to_quad:
                    _set_nibble(quad_words, quad_shift, (_word(raw) >> shift) & 15)
                else:
                    _set_nibble(raw, shift, (_word(quad_words) >> quad_shift) & 15)


def _quad_gate(data: Tensor, logical_n: int, entries: int) -> Tensor:
    """Regroup K=6144 gate/up records into packed quads (``_quad_tiles``)."""
    blocks = logical_n // 16
    slab_tiles, slab_bytes = _SLABS[entries]
    groups = data.shape[1] // (48 // slab_tiles * slab_bytes)
    if logical_n % 64 or (blocks + 5) // 6 != groups:
        raise ValueError("gate/up N must be a multiple of 64 matching the data")
    quad = torch.zeros(
        (data.shape[0], iq2r_glm53_gate_bytes(logical_n, entries)),
        dtype=torch.uint8,
        device=data.device,
    )
    _quad_map(data, quad, logical_n, entries, to_quad=True)
    return quad


# The packed auxiliary is
#   [C * entries * 8]   codebook copies E4M3(codebook * m[c])
#   [d table]           E8M0 d, one uint32 per (row block, K256, lane row):
#                       gate/up [N/64][K/256][16][4 quad atoms],
#                       down    [N/48][K/256][16][3 triplet atoms + 1 pad]
#   [8]                 two FP32 output gains (even rows, odd rows)
# so a lane loads its atoms' d bytes for one K256 with a single dword.


def _pack_v2_d(d: Tensor, atoms: int) -> Tensor:
    """``d [E, N, K/256]`` -> ``[E, N/(16*atoms), K/256, 16, 4]`` bytes (pad byte 127)."""
    experts, n, columns = d.shape
    blocks = n // (16 * atoms)
    grouped = d.reshape(experts, blocks, atoms, 16, columns).permute(0, 1, 4, 3, 2)
    out = torch.full(
        (experts, blocks, columns, 16, 4), 127, dtype=torch.uint8, device=d.device
    )
    out[..., :atoms] = grouped
    return out


def iq2r_glm53_v2_aux_bytes(
    n: int, k: int, classes: int, *, gate: bool, codebook_entries: int = 512
) -> int:
    """Bytes per expert of a packed IQ2R-v2 auxiliary tensor (gate/up or down)."""
    atoms = 4 if gate else 3
    entries = _entries(codebook_entries)
    return classes * entries * 8 + (n // (16 * atoms)) * (k // 256) * 64 + 8


def iq2r_glm53_v2_codebook_entries(
    auxiliary_bytes: int, n: int, k: int, classes: int, *, gate: bool
) -> int:
    """Codebook size (512 or 1,024) of a packed auxiliary of ``auxiliary_bytes`` bytes."""
    for entries in _SLABS:
        if auxiliary_bytes == iq2r_glm53_v2_aux_bytes(
            n, k, classes, gate=gate, codebook_entries=entries
        ):
            return entries
    raise ValueError(
        f"packed IQ2R-v2 auxiliary of {auxiliary_bytes} bytes matches no codebook "
        f"size for N={n}, K={k}"
    )


def _canonical_entries(auxiliary: Tensor, n: int, k: int) -> int:
    """Codebook size of canonical auxiliary rows ``[E, entries * 8 + N*K/256 + 8]``."""
    for entries in _SLABS:
        if auxiliary.shape[1] == entries * 8 + n * (k // 256) + 8:
            return entries
    raise ValueError("canonical IQ2R-v2 auxiliary size does not match N, K")


@torch.no_grad()
def iq2r_glm53_pack_v2(
    gate_up_data: Tensor,
    gate_up_auxiliary: Tensor,
    down_data: Tensor,
    down_auxiliary: Tensor,
    *,
    intermediate_size: int,
    levels: IQ2RV2Levels | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Relayout one GLM-5.3 IQ2R-v2 MoE layer (full checkpoint or a TP shard).

    Inputs are canonical IQ2R-v2 buffers (``aiter.ops.iq2r_v2``), one codebook
    per matrix (gate and up share one), with 512 or 1,024 entries (read from
    the auxiliary size; both matrices must agree); ``levels`` defaults to the
    kernels' level set. Returns
    ``(gate_quad, gate_auxiliary, down_data, down_auxiliary)``.
    """
    from .ops.iq2r_v2 import (
        IQ2R_V2_LEVELS_C4,
        iq2r_v2_check_codebook,
        iq2r_v2_codebook_copies,
    )

    levels = levels or IQ2R_V2_LEVELS_C4
    entries = _canonical_entries(gate_up_auxiliary, 2 * intermediate_size, HIDDEN)
    if _canonical_entries(down_auxiliary, HIDDEN, intermediate_size) != entries:
        raise ValueError("gate/up and down codebooks must have the same size")

    def aux(auxiliary: Tensor, n: int, k: int, atoms: int) -> Tensor:
        experts = auxiliary.shape[0]
        book_bytes = entries * 8
        codebooks = auxiliary[:, :book_bytes].view(torch.float8_e4m3fn).float()
        # Most kernels form copies 1..C-1 from copy 0, exact only in the
        # checked byte range.
        iq2r_v2_check_codebook(codebooks)
        copies = iq2r_v2_codebook_copies(codebooks.reshape(-1, entries, 8), levels)
        copies = copies.permute(1, 0, 2, 3)  # [E, C, entries, 8]
        d = auxiliary[:, book_bytes : book_bytes + n * (k // 256)].reshape(
            experts, n, -1
        )
        return torch.cat(
            (
                copies.to(torch.float8_e4m3fn).view(torch.uint8).reshape(experts, -1),
                _pack_v2_d(d, atoms).reshape(experts, -1),
                auxiliary[:, -8:],
            ),
            dim=1,
        ).contiguous()

    gate = gate_up_data.clone(memory_format=torch.contiguous_format)
    _pack_indices(gate, HIDDEN, entries)
    _transpose_signs(gate, HIDDEN, entries)
    quad = _quad_gate(gate, 2 * intermediate_size, entries)
    del gate
    down = down_data.clone(memory_format=torch.contiguous_format)
    _pack_indices(down, intermediate_size, entries)
    gate_aux = aux(gate_up_auxiliary, 2 * intermediate_size, HIDDEN, 4)
    down_aux = aux(down_auxiliary, HIDDEN, intermediate_size, 3)
    return quad, gate_aux, down, down_aux


# Inverses of the record transforms above, for iq2r_glm53_unpack_v2.


def _unpack_indices(data: Tensor, logical_k: int, entries: int) -> None:
    bits = 10 if entries == 1024 else 9
    for record, third, tenth_at, atom in _atoms(data, logical_k, entries):
        metadata = third[..., 8 + atom].to(torch.int64)
        packed = sum(record[..., j].to(torch.int64) << (8 * j) for j in range(4))
        packed = packed | ((metadata & 15) << 32)
        if tenth_at:
            packed = packed | (((_word(tenth_at[0]) >> tenth_at[1]) & 15) << 36)
        ninth = torch.zeros_like(metadata)
        tenth = torch.zeros_like(metadata)
        for word in range(4):
            index = (packed >> (word * bits)) & ((1 << bits) - 1)
            record[..., word].copy_((index & 255).to(torch.uint8))
            ninth |= ((index >> 8) & 1) << word
            tenth |= ((index >> 9) & 1) << word
        third[..., 8 + atom].copy_(((metadata & 240) | ninth).to(torch.uint8))
        if tenth_at:
            _set_nibble(*tenth_at, tenth)


def _untranspose_signs(data: Tensor, logical_k: int, entries: int) -> None:
    for record, *_ in _atoms(data, logical_k, entries):
        new = sum(record[..., 4 + j].to(torch.int64) << (8 * j) for j in range(4))
        old = torch.zeros_like(new)
        for bit in range(32):
            old |= ((new >> ((bit % 4) * 8 + bit // 4)) & 1) << bit
        for j in range(4):
            record[..., 4 + j].copy_(((old >> (8 * j)) & 255).to(torch.uint8))


def _unquad_gate(quad: Tensor, logical_n: int, entries: int) -> Tensor:
    """Inverse of ``_quad_gate``: packed quads -> K=6144 records.

    The padding N blocks of the last 6-block group (``_quad_gate`` drops them)
    come back as zeros, as ``iq2r_v2_write_data`` writes them.
    """
    blocks = logical_n // 16
    if logical_n % 64 or quad.shape[1] != iq2r_glm53_gate_bytes(logical_n, entries):
        raise ValueError("packed gate/up size does not match N")
    data = torch.zeros(
        (quad.shape[0], iq2r_data_bytes(logical_n, HIDDEN, entries)),
        dtype=torch.uint8,
        device=quad.device,
    )
    _quad_map(data, quad, logical_n, entries, to_quad=False)
    return data


def _unpack_v2_d(table: Tensor, n: int, atoms: int) -> Tensor:
    """Inverse of ``_pack_v2_d``: ``[E, N/(16*atoms), K/256, 16, 4]`` -> ``[E, N, K/256]``."""
    experts, _, columns = table.shape[:3]
    return table[..., :atoms].permute(0, 1, 4, 3, 2).reshape(experts, n, columns)


@torch.no_grad()
def iq2r_glm53_unpack_v2(
    gate_quad: Tensor,
    gate_auxiliary: Tensor,
    down_data: Tensor,
    down_auxiliary: Tensor,
    *,
    intermediate_size: int,
    levels: IQ2RV2Levels | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Inverse of :func:`iq2r_glm53_pack_v2`: packed buffers (as stored in a
    ``glm53-packed-v2`` or ``glm53-packed-v3`` checkpoint, full or a TP shard)
    -> canonical ``(gate_up_data, gate_up_auxiliary, down_data, down_auxiliary)``
    for ``iq2r_v2_materialize``.

    The canonical auxiliary keeps codebook copy 0 (the codebook); copies 1..C-1
    are recomputed by packing, so ``pack(unpack(x)) == x`` holds whenever the
    stored copies are the ones packing computes.
    """
    from .ops.iq2r_v2 import IQ2R_V2_LEVELS_C4

    classes = (levels or IQ2R_V2_LEVELS_C4).classes
    entries = iq2r_glm53_v2_codebook_entries(
        gate_auxiliary.shape[1], 2 * intermediate_size, HIDDEN, classes, gate=True
    )

    def aux(packed: Tensor, n: int, k: int, atoms: int) -> Tensor:
        experts, columns = packed.shape[0], k // 256
        blocks = n // (16 * atoms)
        books = classes * entries * 8
        if packed.shape[1] != books + blocks * columns * 64 + 8:
            raise ValueError("packed IQ2R-v2 auxiliary size does not match N, K")
        table = packed[:, books:-8].reshape(experts, blocks, columns, 16, 4)
        d = _unpack_v2_d(table, n, atoms)
        return torch.cat(
            (packed[:, : entries * 8], d.reshape(experts, -1), packed[:, -8:]), dim=1
        ).contiguous()

    gate = _unquad_gate(gate_quad, 2 * intermediate_size, entries)
    _untranspose_signs(gate, HIDDEN, entries)
    _unpack_indices(gate, HIDDEN, entries)
    down = down_data.clone(memory_format=torch.contiguous_format)
    _unpack_indices(down, intermediate_size, entries)
    gate_aux = aux(gate_auxiliary, 2 * intermediate_size, HIDDEN, 4)
    down_aux = aux(down_auxiliary, HIDDEN, intermediate_size, 3)
    return gate.contiguous(), gate_aux, down, down_aux


def iq2r_glm53_v2_slice_gate_aux(
    aux: Tensor,
    start: int,
    length: int,
    full_n: int,
    classes: int,
    codebook_entries: int = 512,
) -> Tensor:
    """Output-row slice of a packed v2 gate/up auxiliary (64-row aligned)."""
    if start % 64 or length % 64:
        raise ValueError("packed gate/up slices must be aligned to 64 columns")
    books = classes * _entries(codebook_entries) * 8
    table = aux[:, books:-8].reshape(aux.shape[0], full_n // 64, -1)
    picked = table[:, start // 64 : (start + length) // 64].reshape(aux.shape[0], -1)
    return torch.cat((aux[:, :books], picked, aux[:, -8:]), dim=1).contiguous()


def iq2r_glm53_v2_slice_down_aux(
    aux: Tensor,
    start: int,
    length: int,
    full_k: int,
    classes: int,
    codebook_entries: int = 512,
) -> Tensor:
    """Input-column slice of a packed v2 down auxiliary (256-aligned).

    The down data slices like IQ2R's (``iq2r_slice_input_data``).
    """
    if start % 256 or length % 256:
        raise ValueError("IQ2R-v2 down slices must be aligned to 256 columns")
    books = classes * _entries(codebook_entries) * 8
    table = aux[:, books:-8].reshape(aux.shape[0], HIDDEN // 48, full_k // 256, 64)
    picked = table[:, :, start // 256 : (start + length) // 256].reshape(
        aux.shape[0], -1
    )
    return torch.cat((aux[:, :books], picked, aux[:, -8:]), dim=1).contiguous()


def iq2r_glm53_gate_bytes(logical_n: int, codebook_entries: int = 512) -> int:
    """Bytes per expert of a packed GLM-5.3 gate/up matrix (K=6144)."""
    if logical_n % 64:
        raise ValueError("packed gate/up N must be a multiple of 64")
    return logical_n // 64 * 48 * _QUAD_TILE_BYTES[_entries(codebook_entries)]


def iq2r_glm53_slice_gate(
    gate: Tensor, start: int, length: int, codebook_entries: int = 512
) -> Tensor:
    """Output-column slice [start, start+length) of a packed gate/up matrix.

    Both bounds must be multiples of 64 so the slice consists of whole quads.
    """
    quad_bytes = 48 * _QUAD_TILE_BYTES[_entries(codebook_entries)]
    if start % 64 or length % 64 or gate.shape[1] % quad_bytes:
        raise ValueError("packed gate/up slices must be aligned to 64 columns")
    quads = gate.view(gate.shape[0], -1, quad_bytes)
    if (start + length) // 64 > quads.shape[1]:
        raise ValueError("packed gate/up slice out of range")
    picked = quads[:, start // 64 : (start + length) // 64]
    return picked.reshape(gate.shape[0], -1).clone()


# --------------------------------------------------------------------------
# Tuned dispatch
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IQ2RGlm53Config:
    gate_kernel: str
    gate_grid: int
    down_kernel: str
    down_grid: int
    down_chunks: int


def _default_config(tokens: int) -> IQ2RGlm53Config:
    if tokens > DEFAULT_DECODE_TOKENS:
        return IQ2RGlm53Config("prefill", 2, "prefill", 0, 1 if tokens <= 2560 else 2)
    down = "route9" if tokens in (1, 2, 4) else "single"
    return IQ2RGlm53Config("nobarrier", 2, down, 4, 1)


def _config_problem(
    config: IQ2RGlm53Config, tokens: int, intermediate_size: int
) -> str | None:
    """Why ``config`` cannot run ``tokens`` tokens as written, or None.

    The fields each path reads: M1 only the down kernel (route9); decode
    (gate ``nobarrier``, M2..1024) all but ``down_chunks``, and ``down_grid``
    only for a non-route9 down; prefill (gate and down ``prefill``) the gate
    grid and ``down_chunks``.
    """
    if tokens == 1:
        return None if config.down_kernel == "route9" else "M1 needs down route9"
    if config.gate_kernel not in GATE_KERNELS:
        return f"unknown gate kernel {config.gate_kernel!r}"
    if not 1 <= config.gate_grid <= 8:
        return "gate_grid must be 1..8"
    if config.gate_kernel == "prefill":
        if config.down_kernel != "prefill":
            return "the prefill gate runs only with down prefill"
        return None if config.down_chunks in (1, 2) else "down_chunks must be 1 or 2"
    if tokens > MAX_DECODE_TOKENS:
        return f"decode kernels cover M2..{MAX_DECODE_TOKENS}"
    if config.down_kernel == "route9":
        return None if tokens in (2, 4) else "route9 covers M1, M2 and M4"
    if config.down_kernel not in DOWN_KERNELS:
        return f"unknown decode down kernel {config.down_kernel!r}"
    if config.down_kernel == "ordered" and intermediate_size != 256:
        return "the ordered down kernel is TP8 (intermediate 256) only"
    if config.down_kernel == "single2" and intermediate_size != 512:
        return "the single2 down kernel is TP4 (intermediate 512) only"
    return None if 1 <= config.down_grid <= 16 else "down_grid must be 1..16"


_TUNED: dict[str, dict] = {}


def _tuned_table(tuned_file: str) -> dict:
    """(gfx, cu_num, intermediate, codebook) -> [(token, config)] sorted by token.

    Rows without a ``codebook`` column are 512-entry rows.
    """
    if tuned_file not in _TUNED:
        table = pd.read_csv(tuned_file).drop_duplicates()
        if "codebook" not in table.columns:
            table = table.assign(codebook=512)
        # Other model families may share the file; keep only this shape.
        table = table[
            (table.model_dim == HIDDEN)
            & (table.expert == EXPERTS)
            & (table.topk == TOPK)
        ]
        keys = ["gfx", "cu_num", "token", "inter_dim", "codebook"]
        duplicate = table[table.duplicated(keys, keep=False)]
        if len(duplicate):
            raise ValueError(
                f"{tuned_file}: more than one GLM-5.3 IQ2R row for\n"
                f"{duplicate[keys].drop_duplicates().to_string(index=False)}"
            )
        shapes: dict = {}
        for row in table.sort_values("token").itertuples(index=False):
            config = IQ2RGlm53Config(
                str(row.gate_kernel),
                int(row.gate_grid),
                str(row.down_kernel),
                int(row.down_grid),
                int(row.down_chunks),
            )
            problem = _config_problem(config, int(row.token), int(row.inter_dim))
            if problem is not None:
                raise ValueError(
                    f"{tuned_file}: M={row.token} I={row.inter_dim} {config}: {problem}"
                )
            key = (row.gfx, int(row.cu_num), int(row.inter_dim), int(row.codebook))
            shapes.setdefault(key, []).append((int(row.token), config))
        _TUNED[tuned_file] = shapes
    return _TUNED[tuned_file]


@functools.lru_cache(maxsize=1024)
def iq2r_glm53_config(
    tokens: int, intermediate_size: int, codebook_entries: int = 512
) -> IQ2RGlm53Config:
    """Config of the smallest tuned M >= ``tokens`` that can run ``tokens``.

    Tuned rows are powers of two; rounding up skips rows that cannot run the
    real M (M3 skips the route9 M4 row).
    """
    tuned_file = AITER_CONFIGS.AITER_CONFIG_IQ2R_GLM53_FILE
    table = _tuned_table(tuned_file)
    shape = (get_gfx(), get_cu_num(), intermediate_size)
    rows = table.get((*shape, codebook_entries)) or table.get((*shape, 512))
    for tuned_tokens, config in rows or ():
        if tuned_tokens >= tokens and not _config_problem(
            config, tokens, intermediate_size
        ):
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    f"IQ2R GLM-5.3 M={tokens} I={intermediate_size} "
                    f"codebook {codebook_entries}: tuned "
                    f"M={tuned_tokens} {config} from {tuned_file}"
                )
            return config
    return _default_config(tokens)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _task_capacity(routes: int, rows: int) -> int:
    # At most one partial task per routed expert (invalid routes get none); the
    # kernels check the same bound.
    return (routes + rows - 1) // rows + min(routes, EXPERTS)


@dataclass(slots=True)
class IQ2RGlm53Workspace:
    """Fixed-capacity buffers shared by every MoE layer on one device.

    Every launch writes these buffers, so use one workspace per stream.
    """

    max_tokens: int
    intermediate_size: int
    route_input_fp8: Tensor
    route_input_scales: Tensor
    intermediate_fp8: Tensor
    intermediate_scales: Tensor
    route_output: Tensor
    gather_indices: Tensor
    scatter_indices: Tensor
    tasks: Tensor
    task_count: Tensor
    gate_tasks: Tensor
    gate_task_count: Tensor
    sort_scratch: Tensor

    @classmethod
    def allocate(
        cls, max_tokens: int, intermediate_size: int, *, device: torch.device | str
    ) -> IQ2RGlm53Workspace:
        if intermediate_size not in INTERMEDIATE_SIZES:
            raise ValueError(
                "GLM-5.3 IQ2R supports intermediate 256 (TP8) or 512 (TP4)"
            )
        # Longer batches run in MAX_CHUNK_TOKENS pieces.
        max_tokens = min(max(max_tokens, 1), MAX_CHUNK_TOKENS)
        routes = max_tokens * ROUTE_SLOTS
        rows = max(max_tokens, ROUTE_SLOTS)  # M1 quantizes each of its route slots
        decode_routes = min(max_tokens, MAX_DECODE_TOKENS) * ROUTE_SLOTS
        i32 = {"dtype": torch.int32, "device": device}
        u8 = {"dtype": torch.uint8, "device": device}
        fp8 = {"dtype": torch.float8_e4m3fn, "device": device}
        return cls(
            max_tokens=max_tokens,
            intermediate_size=intermediate_size,
            route_input_fp8=torch.empty((rows, HIDDEN), **fp8),
            route_input_scales=torch.empty((rows, HIDDEN // 32), **u8),
            intermediate_fp8=torch.empty((routes, intermediate_size), **fp8),
            intermediate_scales=torch.empty((routes, intermediate_size // 32), **u8),
            route_output=torch.empty(
                (routes, HIDDEN), dtype=torch.bfloat16, device=device
            ),
            gather_indices=torch.empty((routes,), **i32),
            scatter_indices=torch.empty((routes,), **i32),
            tasks=torch.empty(
                (max(_task_capacity(routes, 64), _task_capacity(decode_routes, 32)), 3),
                **i32,
            ),
            task_count=torch.empty((1,), **i32),
            gate_tasks=torch.empty((_task_capacity(decode_routes, 16), 3), **i32),
            gate_task_count=torch.empty((1,), **i32),
            sort_scratch=torch.empty(((routes + 255) // 256 * 512 + 1024,), **i32),
        )


def iq2r_glm53_moe_out(
    hidden_states: Tensor,
    gate_up_data: Tensor,
    gate_up_auxiliary: Tensor,
    down_data: Tensor,
    down_auxiliary: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    output: Tensor,
    workspace: IQ2RGlm53Workspace,
    config: IQ2RGlm53Config | None = None,
) -> None:
    """Routed GLM-5.3 MoE on packed weights: ``output = sum_k w_k * expert_k(x)``.

    ``hidden_states`` is BF16 [tokens, 6144]; ``topk_ids``/``topk_weights``
    are int32/float32 [tokens, 8] over the 256 routed experts; the weights come
    from :func:`iq2r_glm53_pack_v2`. Routes with an id outside 0..255
    contribute nothing. ``config`` overrides the tuned launch choice (used by
    the tuner); it must fit ``tokens``, and ``tokens`` must fit the workspace.
    """
    # The gate and down kernel bodies are built for gfx950 only and are empty
    # on other targets, which would leave ``output`` unwritten.
    gfx = get_gfx()
    if gfx != "gfx950":
        raise RuntimeError(f"GLM-5.3 IQ2R MoE requires gfx950, not {gfx}")
    tokens = hidden_states.shape[0]
    if tokens > workspace.max_tokens:
        if config is not None:
            # Each chunk needs its own launch choice (the tail may be M1).
            raise ValueError(
                f"an explicit config runs one launch, but {tokens} tokens exceed "
                f"the workspace's {workspace.max_tokens}"
            )
        for begin in range(0, tokens, workspace.max_tokens):
            rows = slice(begin, min(begin + workspace.max_tokens, tokens))
            iq2r_glm53_moe_out(
                hidden_states[rows],
                gate_up_data,
                gate_up_auxiliary,
                down_data,
                down_auxiliary,
                topk_weights[rows],
                topk_ids[rows],
                output[rows],
                workspace,
            )
        return
    if tokens == 0:
        return
    if topk_ids.shape != (tokens, TOPK) or topk_weights.shape != (tokens, TOPK):
        raise ValueError("GLM-5.3 IQ2R expects top-8 routing [tokens, 8]")
    if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous float32")
    if config is None:
        entries = iq2r_glm53_v2_codebook_entries(
            gate_up_auxiliary.shape[1],
            2 * workspace.intermediate_size,
            HIDDEN,
            4,
            gate=True,
        )
        config = iq2r_glm53_config(tokens, workspace.intermediate_size, entries)
    else:
        problem = _config_problem(config, tokens, workspace.intermediate_size)
        if problem is not None:
            raise ValueError(f"{config} cannot run M={tokens}: {problem}")
    routes = tokens * ROUTE_SLOTS
    ids = topk_ids.reshape(-1)
    gather = workspace.gather_indices[:routes]
    scatter = workspace.scatter_indices[:routes]
    tasks = workspace.tasks
    count = workspace.task_count
    intermediate = workspace.intermediate_fp8[:routes]
    intermediate_scales = workspace.intermediate_scales[:routes]
    quant_rows = ROUTE_SLOTS if tokens == 1 else tokens
    quant = workspace.route_input_fp8[:quant_rows]
    quant_scales = workspace.route_input_scales[:quant_rows]

    if tokens == 1:
        # One-row tasks; each route gets its own copy of the quantized row.
        tasks = tasks[:routes]
        iq2r_glm53_m1_route_quant_out(
            hidden_states, ids, scatter, tasks, count, quant, quant_scales
        )
        iq2r_glm53_gate_m1_out(
            quant, quant_scales, gate_up_data, gate_up_auxiliary, tasks, count,
            intermediate, intermediate_scales,
        )  # fmt: skip
    elif config.gate_kernel != "prefill":
        # One launch sorts routes into M32 down / M16 gate tasks and quantizes
        # each token row once; the gate gathers rows through ``gather``.
        tasks = tasks[: _task_capacity(routes, 32)]
        gate_tasks = workspace.gate_tasks[: _task_capacity(routes, 16)]
        iq2r_glm53_sort_quant_out(
            hidden_states, ids, gather, scatter, tasks, count, quant, quant_scales,
            gate_tasks, workspace.gate_task_count,
        )  # fmt: skip
        iq2r_glm53_gate_out(
            quant, quant_scales, gate_up_data, gate_up_auxiliary, gate_tasks,
            workspace.gate_task_count, gather, intermediate, intermediate_scales,
            GATE_KERNELS[config.gate_kernel], config.gate_grid,
        )  # fmt: skip
    else:
        tasks = tasks[: _task_capacity(routes, 64)]
        iq2r_glm53_prefill_sort_quant_out(
            hidden_states, ids, gather, scatter, tasks, count, quant, quant_scales,
            workspace.sort_scratch,
        )  # fmt: skip
        iq2r_glm53_gate_out(
            quant, quant_scales, gate_up_data, gate_up_auxiliary, tasks, count,
            gather, intermediate, intermediate_scales, GATE_KERNELS["prefill"],
            config.gate_grid,
        )  # fmt: skip
        iq2r_glm53_down_reduce_out(
            intermediate, intermediate_scales, down_data, down_auxiliary, tasks,
            count, workspace.route_output[:routes], scatter, topk_weights, output,
            config.down_chunks,
        )  # fmt: skip
        return

    if config.down_kernel == "route9":
        # Token-owned down projection with the weighted top-8 sum fused in.
        iq2r_glm53_down_route9_out(
            intermediate, intermediate_scales, down_data, down_auxiliary, ids,
            scatter, topk_weights, output,
        )  # fmt: skip
        return
    route_output = workspace.route_output[:routes]
    iq2r_glm53_down_out(
        intermediate, intermediate_scales, down_data, down_auxiliary, tasks, count,
        route_output, DOWN_KERNELS[config.down_kernel], config.down_grid,
    )  # fmt: skip
    iq2r_glm53_route_reduce_out(route_output, topk_weights, scatter, output)


__all__ = [
    "IQ2RGlm53Config",
    "IQ2RGlm53Workspace",
    "iq2r_glm53_config",
    "iq2r_glm53_gate_bytes",
    "iq2r_glm53_moe_out",
    "iq2r_glm53_pack_v2",
    "iq2r_glm53_slice_gate",
    "iq2r_glm53_unpack_v2",
    "iq2r_glm53_v2_aux_bytes",
    "iq2r_glm53_v2_codebook_entries",
    "iq2r_glm53_v2_slice_down_aux",
    "iq2r_glm53_v2_slice_gate_aux",
]
