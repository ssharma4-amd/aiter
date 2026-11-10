# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Storage layout of AITER's native-basis IQ2R weights.

CPU-only, with no compiled extension, so checkpoint tooling can size and
slice IQ2R byte buffers before touching a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

IQ2R_ACTIVATION_BASIS = "native"

IQ2R_CODEBOOK_ENTRIES = 512
IQ2R_VECTOR_SIZE = 8
IQ2R_SCALE_BLOCK = 32
IQ2R_TILE_N = 16
IQ2R_TILE_K = 128
IQ2R_LANE_RECORD_BYTES = 8
IQ2R_LANES_PER_TILE = 64
IQ2R_N_BLOCKS_PER_GROUP = 6
IQ2R_ATOMS_PER_TRIPLET = 3
IQ2R_TRIPLETS_PER_GROUP = IQ2R_N_BLOCKS_PER_GROUP // IQ2R_ATOMS_PER_TRIPLET
IQ2R_ATOM_PAIR_RECORDS_BYTES = IQ2R_LANES_PER_TILE * 2 * IQ2R_LANE_RECORD_BYTES
IQ2R_ATOM_TWO_RECORD_BYTES = 12
IQ2R_ATOM_TWO_RECORDS_OFFSET = IQ2R_ATOM_PAIR_RECORDS_BYTES
IQ2R_ATOM_TWO_METADATA_OFFSET = IQ2R_LANE_RECORD_BYTES
IQ2R_TRIPLET_BYTES = IQ2R_ATOM_PAIR_RECORDS_BYTES + (
    IQ2R_LANES_PER_TILE * IQ2R_ATOM_TWO_RECORD_BYTES
)
IQ2R_GROUP_BYTES = IQ2R_TRIPLETS_PER_GROUP * IQ2R_TRIPLET_BYTES
IQ2R_CODEBOOK_BYTES = IQ2R_CODEBOOK_ENTRIES * IQ2R_VECTOR_SIZE


def _require_int(name: str, value: int, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def iq2r_k_tiles(k: int) -> int:
    """Number of stored K=128 tiles for a logical contraction width."""

    _require_int("k", k, positive=True)
    if k % IQ2R_SCALE_BLOCK:
        raise ValueError(f"IQ2R requires K divisible by {IQ2R_SCALE_BLOCK}, got {k}")
    return (k + IQ2R_TILE_K - 1) // IQ2R_TILE_K


def iq2r_n_blocks(n: int) -> int:
    """Number of logical N=16 blocks."""

    _require_int("n", n, positive=True)
    if n % IQ2R_TILE_N:
        raise ValueError(f"IQ2R requires N divisible by {IQ2R_TILE_N}, got {n}")
    return n // IQ2R_TILE_N


def iq2r_physical_n_blocks(n: int) -> int:
    """N-block count rounded to the six-block physical storage group."""

    logical = iq2r_n_blocks(n)
    return (
        (logical + IQ2R_N_BLOCKS_PER_GROUP - 1) // IQ2R_N_BLOCKS_PER_GROUP
    ) * IQ2R_N_BLOCKS_PER_GROUP


def iq2r_data_bytes(n: int, k: int) -> int:
    physical_n_blocks = iq2r_physical_n_blocks(n)
    return (
        physical_n_blocks
        // IQ2R_N_BLOCKS_PER_GROUP
        * iq2r_k_tiles(k)
        * IQ2R_GROUP_BYTES
    )


@dataclass(frozen=True, slots=True)
class IQ2RMetadata:
    """Shape of one IQ2R expert projection (``N`` outputs, ``K`` inputs)."""

    logical_n: int
    logical_k: int

    def __post_init__(self) -> None:
        # Applies the type and divisibility checks.
        iq2r_data_bytes(self.logical_n, self.logical_k)

    @property
    def n_blocks(self) -> int:
        return iq2r_n_blocks(self.logical_n)

    @property
    def physical_n_blocks(self) -> int:
        return iq2r_physical_n_blocks(self.logical_n)

    @property
    def k_tiles(self) -> int:
        return iq2r_k_tiles(self.logical_k)

    @property
    def data_bytes(self) -> int:
        return iq2r_data_bytes(self.logical_n, self.logical_k)


def iq2r_slice_input_data(
    data: Tensor,
    metadata: IQ2RMetadata,
    start: int,
    length: int,
) -> Tensor:
    """Extract an exact contiguous K-tile slice of stacked IQ2R data."""

    _validate_byte_matrix("data", data, metadata.data_bytes)
    _require_int("start", start)
    _require_int("length", length, positive=True)
    if (
        start < 0
        or start % IQ2R_TILE_K
        or length % IQ2R_TILE_K
        or start + length > metadata.logical_k
    ):
        raise ValueError(
            f"IQ2R input slice must be in range and aligned to {IQ2R_TILE_K} columns"
        )
    target = IQ2RMetadata(logical_n=metadata.logical_n, logical_k=length)
    experts = data.shape[0]
    groups = metadata.physical_n_blocks // IQ2R_N_BLOCKS_PER_GROUP
    source = data.view(experts, groups, metadata.k_tiles, IQ2R_GROUP_BYTES)
    tile_start = start // IQ2R_TILE_K
    # ``contiguous()`` may return the original narrow view when its strides are
    # already dense, retaining a non-zero storage offset.  IQ2R consumers treat
    # byte zero as the first packed tile, so force a fresh zero-offset storage.
    return (
        source[:, :, tile_start : tile_start + target.k_tiles]
        .clone(memory_format=torch.contiguous_format)
        .view(experts, target.data_bytes)
    )


def _validate_byte_matrix(name: str, value: Tensor, expected_bytes: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.uint8:
        raise TypeError(f"{name} must have dtype uint8, got {value.dtype}")
    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [experts,{expected_bytes}], got {tuple(value.shape)}"
        )
    if value.shape[1] != expected_bytes:
        raise ValueError(
            f"{name} has {value.shape[1]} bytes per expert, expected {expected_bytes}"
        )
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


__all__ = [name for name in globals() if name.startswith("IQ2R_")] + [
    "IQ2RMetadata",
    "iq2r_data_bytes",
    "iq2r_k_tiles",
    "iq2r_n_blocks",
    "iq2r_physical_n_blocks",
    "iq2r_slice_input_data",
]
