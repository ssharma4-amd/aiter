# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 IQ2R-v2 MoE on the packed layout (TP4 and TP8, gfx950).

256 routed experts with top-8 routing; the shared expert stays FP8 outside the
MoE. The packed layout is a byte-exact relayout of the canonical IQ2R-v2
records (``aiter.ops.iq2r_v2``):

* the four 9-bit indices of an atom pack as low32 + high4;
* gate/up record signs transpose into four 8-bit planes, and the gate/up
  records are regrouped into quads of four 16-column blocks;
* the auxiliary holds the codebook copies, a per-lane d table and the gains.

Every transform is per record, so packing a full checkpoint and then taking a
tensor-parallel shard (``iq2r_glm53_slice_gate``, ``iq2r_glm53_v2_slice_gate_aux``,
``iq2r_slice_input_data``, ``iq2r_glm53_v2_slice_down_aux``) equals packing
that shard. ``iq2r_glm53_unpack_v2`` inverts the packing.

Launch choices per token count come from ``aiter/configs/iq2r_glm53_tuned.csv``
(override with ``AITER_CONFIG_IQ2R_GLM53``). Other token counts use the next
tuned row up that can run them, else the defaults in :func:`_default_config`.
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

_RECORD_BYTES = 3584
_QUAD_BYTES = 48 * 2304

GATE_KERNELS = {"nobarrier": 1, "prefill": 2}
DOWN_KERNELS = {"ordered": 1, "single": 2}


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def _geometry(data: Tensor, logical_k: int) -> tuple[int, int]:
    experts = data.shape[0]
    k_tiles = logical_k // 128
    if data.shape[1] % (k_tiles * _RECORD_BYTES):
        raise ValueError("unexpected IQ2R record layout")
    return experts, k_tiles


def _atoms(src: Tensor, dst: Tensor, experts: int, k_tiles: int):
    """Yield (record, dst_record, third, dst_third, atom) views per 8-byte atom."""
    s = src.reshape(experts, -1, k_tiles, _RECORD_BYTES)
    d = dst.reshape_as(s)
    for triplet in range(2):
        offset = triplet * 1792
        pair = s[..., offset : offset + 1024].reshape(experts, -1, k_tiles, 64, 16)
        third = s[..., offset + 1024 : offset + 1792].reshape(
            experts, -1, k_tiles, 64, 12
        )
        dst_pair = d[..., offset : offset + 1024].reshape_as(pair)
        dst_third = d[..., offset + 1024 : offset + 1792].reshape_as(third)
        for atom in range(3):
            record = pair[..., atom * 8 : atom * 8 + 8] if atom < 2 else third[..., :8]
            out = (
                dst_pair[..., atom * 8 : atom * 8 + 8]
                if atom < 2
                else dst_third[..., :8]
            )
            yield record, out, third, dst_third, atom


# The record transforms below rewrite ``data`` in place: each atom's new bytes
# depend only on that atom's old bytes, which are read first.


def _pack_indices(data: Tensor, logical_k: int) -> None:
    experts, k_tiles = _geometry(data, logical_k)
    for record, out, third, dst_third, atom in _atoms(data, data, experts, k_tiles):
        metadata = third[..., 8 + atom].to(torch.int64)
        packed = torch.zeros_like(metadata)
        for word in range(4):
            index = record[..., word].to(torch.int64) | (((metadata >> word) & 1) << 8)
            packed |= index << (word * 9)
        for byte in range(4):
            out[..., byte].copy_(((packed >> (8 * byte)) & 255).to(torch.uint8))
        dst_third[..., 8 + atom].copy_(
            ((metadata & 240) | ((packed >> 32) & 15)).to(torch.uint8)
        )


def _transpose_signs(data: Tensor, logical_k: int) -> None:
    experts, k_tiles = _geometry(data, logical_k)
    for record, out, _, _, _ in _atoms(data, data, experts, k_tiles):
        old = sum(record[..., 4 + j].to(torch.int64) << (8 * j) for j in range(4))
        new = torch.zeros_like(old)
        for bit in range(32):
            new |= ((old >> bit) & 1) << ((bit % 4) * 8 + bit // 4)
        for j in range(4):
            out[..., 4 + j].copy_(((new >> (8 * j)) & 255).to(torch.uint8))


def _quad_gate(data: Tensor, logical_n: int) -> Tensor:
    """Regroup K=6144 gate/up records into [experts, N/64, 48, 2304]."""
    groups = data.shape[1] // (48 * _RECORD_BYTES)
    blocks = logical_n // 16
    if logical_n % 64 or (blocks + 5) // 6 != groups:
        raise ValueError("gate/up N must be a multiple of 64 matching the data")
    experts = data.shape[0]
    source = data.reshape(experts, groups, 48, _RECORD_BYTES)
    target = torch.empty(
        (experts, blocks // 4, 48, 2304), dtype=torch.uint8, device=data.device
    )
    for block in range(blocks):
        atom = block % 3
        offset = (block % 6) // 3 * 1792
        third = source[:, block // 6, :, offset + 1024 : offset + 1792].reshape(
            experts, 48, 64, 12
        )
        if atom < 2:
            pair = source[:, block // 6, :, offset : offset + 1024].reshape(
                experts, 48, 64, 16
            )
            record = pair[..., atom * 8 : atom * 8 + 8]
        else:
            record = third[..., :8]
        slot = block % 4
        pair_base = (slot // 2) * 1024
        pair_out = target[:, block // 4, :, pair_base : pair_base + 1024].view(
            experts, 48, 64, 16
        )
        pair_out[..., slot % 2 * 8 : slot % 2 * 8 + 8].copy_(record)
        metadata = target[:, block // 4, :, 2048:2304].view(experts, 48, 64, 4)
        metadata[..., slot].copy_(third[..., 8 + atom])
    return target.flatten(1).contiguous()


# The packed auxiliary is
#   [C * 4096]          codebook copies E4M3(codebook * m[c])
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


def iq2r_glm53_v2_aux_bytes(n: int, k: int, classes: int, *, gate: bool) -> int:
    """Bytes per expert of a packed IQ2R-v2 auxiliary tensor (gate/up or down)."""
    atoms = 4 if gate else 3
    return classes * 4096 + (n // (16 * atoms)) * (k // 256) * 64 + 8


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
    per matrix (gate and up share one); ``levels`` defaults to the
    kernels' level set. Returns
    ``(gate_quad, gate_auxiliary, down_data, down_auxiliary)``.
    """
    from .ops.iq2r_v2 import (
        IQ2R_V2_LEVELS_C4,
        iq2r_v2_check_codebook,
        iq2r_v2_codebook_copies,
    )

    levels = levels or IQ2R_V2_LEVELS_C4

    def aux(auxiliary: Tensor, n: int, k: int, atoms: int) -> Tensor:
        experts = auxiliary.shape[0]
        codebooks = auxiliary[:, :4096].view(torch.float8_e4m3fn).float()
        # Most kernels form copies 1..C-1 from copy 0 in LDS, exact only in
        # the checked byte range.
        iq2r_v2_check_codebook(codebooks)
        copies = iq2r_v2_codebook_copies(codebooks.reshape(-1, 512, 8), levels)
        copies = copies.permute(1, 0, 2, 3)  # [E, C, 512, 8]
        d = auxiliary[:, 4096 : 4096 + n * (k // 256)].reshape(experts, n, -1)
        return torch.cat(
            (
                copies.to(torch.float8_e4m3fn).view(torch.uint8).reshape(experts, -1),
                _pack_v2_d(d, atoms).reshape(experts, -1),
                auxiliary[:, -8:],
            ),
            dim=1,
        ).contiguous()

    gate = gate_up_data.clone(memory_format=torch.contiguous_format)
    _pack_indices(gate, HIDDEN)
    _transpose_signs(gate, HIDDEN)
    quad = _quad_gate(gate, 2 * intermediate_size)
    del gate
    down = down_data.clone(memory_format=torch.contiguous_format)
    _pack_indices(down, intermediate_size)
    gate_aux = aux(gate_up_auxiliary, 2 * intermediate_size, HIDDEN, 4)
    down_aux = aux(down_auxiliary, HIDDEN, intermediate_size, 3)
    return quad, gate_aux, down, down_aux


# Inverses of the record transforms above, for iq2r_glm53_unpack_v2.


def _unpack_indices(data: Tensor, logical_k: int) -> None:
    experts, k_tiles = _geometry(data, logical_k)
    for record, out, third, dst_third, atom in _atoms(data, data, experts, k_tiles):
        metadata = third[..., 8 + atom].to(torch.int64)
        packed = sum(record[..., j].to(torch.int64) << (8 * j) for j in range(4))
        packed = packed | ((metadata & 15) << 32)
        ninth = torch.zeros_like(metadata)
        for word in range(4):
            index = (packed >> (word * 9)) & 511
            out[..., word].copy_((index & 255).to(torch.uint8))
            ninth |= (index >> 8) << word
        dst_third[..., 8 + atom].copy_(((metadata & 240) | ninth).to(torch.uint8))


def _untranspose_signs(data: Tensor, logical_k: int) -> None:
    experts, k_tiles = _geometry(data, logical_k)
    for record, out, _, _, _ in _atoms(data, data, experts, k_tiles):
        new = sum(record[..., 4 + j].to(torch.int64) << (8 * j) for j in range(4))
        old = torch.zeros_like(new)
        for bit in range(32):
            old |= ((new >> ((bit % 4) * 8 + bit // 4)) & 1) << bit
        for j in range(4):
            out[..., 4 + j].copy_(((old >> (8 * j)) & 255).to(torch.uint8))


def _unquad_gate(quad: Tensor, logical_n: int) -> Tensor:
    """Inverse of ``_quad_gate``: ``[experts, N/64, 48, 2304]`` -> K=6144 records.

    The padding N blocks of the last 6-block group (``_quad_gate`` drops them)
    come back as zeros, as ``iq2r_v2_write_data`` writes them.
    """
    blocks = logical_n // 16
    if logical_n % 64 or quad.shape[1] != (blocks // 4) * 48 * 2304:
        raise ValueError("packed gate/up size does not match N")
    experts, groups = quad.shape[0], (blocks + 5) // 6
    source = quad.reshape(experts, blocks // 4, 48, 2304)
    target = torch.zeros(
        (experts, groups, 48, _RECORD_BYTES), dtype=torch.uint8, device=quad.device
    )
    for block in range(blocks):
        atom = block % 3
        offset = (block % 6) // 3 * 1792
        third = target[:, block // 6, :, offset + 1024 : offset + 1792].view(
            experts, 48, 64, 12
        )
        if atom < 2:
            pair = target[:, block // 6, :, offset : offset + 1024].view(
                experts, 48, 64, 16
            )
            record = pair[..., atom * 8 : atom * 8 + 8]
        else:
            record = third[..., :8]
        slot = block % 4
        pair_base = (slot // 2) * 1024
        pair_in = source[:, block // 4, :, pair_base : pair_base + 1024].reshape(
            experts, 48, 64, 16
        )
        record.copy_(pair_in[..., slot % 2 * 8 : slot % 2 * 8 + 8])
        metadata = source[:, block // 4, :, 2048:2304].reshape(experts, 48, 64, 4)
        third[..., 8 + atom].copy_(metadata[..., slot])
    return target.flatten(1)


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
    ``glm53-packed-v2`` checkpoint, full or a TP shard) -> canonical
    ``(gate_up_data, gate_up_auxiliary, down_data, down_auxiliary)`` for
    ``iq2r_v2_materialize``.

    The canonical auxiliary keeps codebook copy 0 (the codebook); copies 1..C-1
    are recomputed by packing, so ``pack(unpack(x)) == x`` holds whenever the
    stored copies are the ones packing computes.
    """
    from .ops.iq2r_v2 import IQ2R_V2_LEVELS_C4

    classes = (levels or IQ2R_V2_LEVELS_C4).classes

    def aux(packed: Tensor, n: int, k: int, atoms: int) -> Tensor:
        experts, columns = packed.shape[0], k // 256
        blocks = n // (16 * atoms)
        if packed.shape[1] != classes * 4096 + blocks * columns * 64 + 8:
            raise ValueError("packed IQ2R-v2 auxiliary size does not match N, K")
        table = packed[:, classes * 4096 : -8].reshape(experts, blocks, columns, 16, 4)
        d = _unpack_v2_d(table, n, atoms)
        return torch.cat(
            (packed[:, :4096], d.reshape(experts, -1), packed[:, -8:]), dim=1
        ).contiguous()

    gate = _unquad_gate(gate_quad, 2 * intermediate_size)
    _untranspose_signs(gate, HIDDEN)
    _unpack_indices(gate, HIDDEN)
    down = down_data.clone(memory_format=torch.contiguous_format)
    _unpack_indices(down, intermediate_size)
    gate_aux = aux(gate_auxiliary, 2 * intermediate_size, HIDDEN, 4)
    down_aux = aux(down_auxiliary, HIDDEN, intermediate_size, 3)
    return gate.contiguous(), gate_aux, down, down_aux


def iq2r_glm53_v2_slice_gate_aux(
    aux: Tensor, start: int, length: int, full_n: int, classes: int
) -> Tensor:
    """Output-row slice of a packed v2 gate/up auxiliary (64-row aligned)."""
    if start % 64 or length % 64:
        raise ValueError("packed gate/up slices must be aligned to 64 columns")
    books = classes * 4096
    table = aux[:, books:-8].reshape(aux.shape[0], full_n // 64, -1)
    picked = table[:, start // 64 : (start + length) // 64].reshape(aux.shape[0], -1)
    return torch.cat((aux[:, :books], picked, aux[:, -8:]), dim=1).contiguous()


def iq2r_glm53_v2_slice_down_aux(
    aux: Tensor, start: int, length: int, full_k: int, classes: int
) -> Tensor:
    """Input-column slice of a packed v2 down auxiliary (256-aligned).

    The down data slices like IQ2R's (``iq2r_slice_input_data``).
    """
    if start % 256 or length % 256:
        raise ValueError("IQ2R-v2 down slices must be aligned to 256 columns")
    books = classes * 4096
    table = aux[:, books:-8].reshape(aux.shape[0], HIDDEN // 48, full_k // 256, 64)
    picked = table[:, :, start // 256 : (start + length) // 256].reshape(
        aux.shape[0], -1
    )
    return torch.cat((aux[:, :books], picked, aux[:, -8:]), dim=1).contiguous()


def iq2r_glm53_gate_bytes(logical_n: int) -> int:
    """Bytes per expert of a packed GLM-5.3 gate/up matrix (K=6144)."""
    if logical_n % 64:
        raise ValueError("packed gate/up N must be a multiple of 64")
    return logical_n // 64 * _QUAD_BYTES


def iq2r_glm53_slice_gate(gate: Tensor, start: int, length: int) -> Tensor:
    """Output-column slice [start, start+length) of a packed gate/up matrix.

    Both bounds must be multiples of 64 so the slice consists of whole quads.
    """
    if start % 64 or length % 64 or gate.shape[1] % _QUAD_BYTES:
        raise ValueError("packed gate/up slices must be aligned to 64 columns")
    quads = gate.view(gate.shape[0], -1, _QUAD_BYTES)
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
    return None if 1 <= config.down_grid <= 16 else "down_grid must be 1..16"


_TUNED: dict[str, dict] = {}


def _tuned_table(tuned_file: str) -> dict:
    """(gfx, cu_num, intermediate) -> [(token, config)] sorted by token."""
    if tuned_file not in _TUNED:
        table = pd.read_csv(tuned_file).drop_duplicates()
        # Other model families may share the file; keep only this shape.
        table = table[
            (table.model_dim == HIDDEN)
            & (table.expert == EXPERTS)
            & (table.topk == TOPK)
        ]
        keys = ["gfx", "cu_num", "token", "inter_dim"]
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
            key = (row.gfx, int(row.cu_num), int(row.inter_dim))
            shapes.setdefault(key, []).append((int(row.token), config))
        _TUNED[tuned_file] = shapes
    return _TUNED[tuned_file]


@functools.lru_cache(maxsize=1024)
def iq2r_glm53_config(tokens: int, intermediate_size: int) -> IQ2RGlm53Config:
    """Config of the smallest tuned M >= ``tokens`` that can run ``tokens``.

    Tuned rows are powers of two; rounding up skips rows that cannot run the
    real M (M3 skips the route9 M4 row).
    """
    tuned_file = AITER_CONFIGS.AITER_CONFIG_IQ2R_GLM53_FILE
    rows = _tuned_table(tuned_file).get((get_gfx(), get_cu_num(), intermediate_size))
    for tuned_tokens, config in rows or ():
        if tuned_tokens >= tokens and not _config_problem(
            config, tokens, intermediate_size
        ):
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    f"IQ2R GLM-5.3 M={tokens} I={intermediate_size}: tuned "
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
        config = iq2r_glm53_config(tokens, workspace.intermediate_size)
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
    "iq2r_glm53_v2_slice_down_aux",
    "iq2r_glm53_v2_slice_gate_aux",
]
