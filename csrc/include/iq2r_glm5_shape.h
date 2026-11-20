// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Model shapes of the GLM-5 IQ2R packed MoE kernels. Every kernel is
// instantiated per shape, so a shape's constants fold as before.
//
// Hidden is the gate/up K and the down N. The down weights are stored in
// 48-row triplets; DownRows pads Hidden to a whole number of the tiles the
// down kernels use, and the rows past Hidden are computed and not stored.
// Clamp selects GLM-5.3-Flash's SwiGLU, silu(min(gate, L)) * clamp(up, -L, L)
// with L = SwigluLimit; without it the gate computes silu(gate) * up. The
// shared expert uses the same SwiGLU as the routed experts.
//
// The decode gates split K into eight slices, one per wave: SliceTiles K
// tiles each. A 1,024-entry gate/up quad stores each slice's tiles, then a
// tenth-bit word of 2 * SliceTiles bytes per lane (iq2r_glm53_pack_v2).

namespace aiter {

template <int Hidden_, int Experts_, int DownRows_, bool Clamp_>
struct Glm5MoeShape
{
    static constexpr int Hidden   = Hidden_;
    static constexpr int Experts  = Experts_;
    static constexpr int DownRows = DownRows_;
    static constexpr bool Clamp   = Clamp_;
    static constexpr int TopK       = 8;
    static constexpr int RouteSlots = 9; // route = token * 9 + k; slot 8 is the shared expert's
    static constexpr int Groups     = Hidden / 32;  // MXFP8 scale groups per row
    static constexpr int KTiles     = Hidden / 128; // gate/up K tiles
    static constexpr int K256       = Hidden / 256;
    static constexpr int SliceTiles = KTiles / 8;   // K tiles per decode-gate wave
    static constexpr int ExtLaneBytes = 2 * SliceTiles; // 1,024 entries: tenth bits per lane and slice
    static constexpr float SwigluLimit = 10.0f;
    static_assert(Hidden % 2048 == 0 && SliceTiles % 2 == 0 && SliceTiles <= 6,
                  "eight K slices of whole K256 columns; tenth bits fit a 12-byte load");
    static_assert(DownRows % 768 == 0 && DownRows >= Hidden);
    static_assert(Experts < 512, "the sorts keep 512 buckets: the experts and invalid routes");
};

// GLM-5.3: hidden 6144, 256 routed experts, SiLU(gate) * up.
using Glm53Shape = Glm5MoeShape<6144, 256, 6144, false>;
// GLM-5.3-Flash: hidden 4096, 288 routed experts, clamped SwiGLU (limit 10);
// down rows padded to 4608 (96 triplets).
using Glm5NextShape = Glm5MoeShape<4096, 288, 4608, true>;

} // namespace aiter
