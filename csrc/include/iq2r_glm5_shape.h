// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Model shapes of the GLM-5 IQ2R packed MoE kernels (csrc/kernels/iq2r). Every
// kernel is instantiated per shape, so a shape's constants fold as literals.
//
// Hidden is the gate/up K and the down N. The down weights are stored in
// 48-row triplets; DownRows pads Hidden to a whole number of the tiles the
// down kernels use, and the rows past Hidden are computed and not stored.
// Clamp selects GLM-5.3-Flash's SwiGLU, silu(min(gate, L)) * clamp(up, -L, L)
// with L = SwigluLimit; without it the gate computes silu(gate) * up. The
// fused shared expert uses the same SwiGLU as the routed experts.
//
// The decode gates split K into eight slices, one per wave: SliceTiles K128
// tiles each. A 1,024-entry gate/up quad stores each slice's tiles, then a
// tenth-bit word of ExtLaneBytes per lane (aiter/iq2r_glm53.py,
// iq2r_glm53_pack), so the packed gate/up layout depends on the shape.

namespace aiter {

template <int Hidden_, int Experts_, int DownRows_, bool Clamp_>
struct Glm5MoeShape
{
    static constexpr int Hidden   = Hidden_;
    static constexpr int Experts  = Experts_;
    static constexpr int DownRows = DownRows_;
    static constexpr bool Clamp   = Clamp_;
    static constexpr float SwigluLimit = 10.0f;
    // E8M0 scales of an activation row, K128 tiles of the hidden size (also
    // the 128-column tiles of a hidden-wide output) and its K256 columns of d
    // words.
    static constexpr int HiddenScales = Hidden / 32;
    static constexpr int HiddenTiles  = Hidden / 128;
    static constexpr int HiddenK256   = Hidden / 256;
    // K128 tiles per decode-gate wave, and the 1,024-entry tenth-bit bytes a
    // lane holds per slice (four bits per atom and tile).
    static constexpr int SliceTiles   = HiddenTiles / 8;
    static constexpr int ExtLaneBytes = 2 * SliceTiles;
    static_assert(Hidden % 2048 == 0 && SliceTiles % 2 == 0 && SliceTiles <= 6,
                  "eight K slices of whole K256 columns; a slice's tenth bits fit one 12-byte load");
    static_assert(DownRows % 768 == 0 && DownRows >= Hidden,
                  "the down kernels tile N by 384 and 768 columns");
    static_assert(Experts < 512, "the route sorts keep 512 buckets: the experts and invalid routes");
};

// GLM-5.3: hidden 6144, 256 routed experts, SiLU(gate) * up.
using Glm53Shape = Glm5MoeShape<6144, 256, 6144, false>;
// GLM-5.3-Flash: hidden 4096, 288 routed experts, SwiGLU clamped at 10; down
// rows padded to 4608 (96 triplets).
using Glm5NextShape = Glm5MoeShape<4096, 288, 4608, true>;

} // namespace aiter
