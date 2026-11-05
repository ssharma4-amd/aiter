// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "quant_amax_floor.h"
#include "iq2r.h"
#include "mx_quant_utils.h"
#include "opus/opus.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <initializer_list>
#include <type_traits>

namespace aiter {
namespace {

constexpr int kCodebookBytes       = 4096;
constexpr int kTileN               = 16;
constexpr int kTileK               = 128;
constexpr int kScaleBlock          = 32;
constexpr int kLaneRecordBytes     = 8;
constexpr int kAtomsPerTriplet     = 3;
constexpr int kNBlocksPerGroup     = 6;
constexpr int kAtomTwoRecordBytes  = 12;
constexpr int kAtomTwoOffset       = 1024;
constexpr int kAtomTwoMetadata     = 8;
constexpr int kTripletBytes        = 1792;
constexpr int kGroupBytes          = 3584;

__device__ __forceinline__ int64_t physical_tile(int n_block,
                                                  int k_tile,
                                                  int k_tiles)
{
    return (static_cast<int64_t>(n_block / kNBlocksPerGroup) * k_tiles + k_tile) *
               kNBlocksPerGroup +
           n_block % kNBlocksPerGroup;
}

__device__ __forceinline__ int64_t triplet_base(int64_t tile)
{
    const int block_in_group = static_cast<int>(tile % kNBlocksPerGroup);
    return (tile / kNBlocksPerGroup) * kGroupBytes +
           (block_in_group / kAtomsPerTriplet) * kTripletBytes;
}

__device__ __forceinline__ int64_t lane_record_offset(int64_t tile, int lane)
{
    const int atom = static_cast<int>(tile % kNBlocksPerGroup) % kAtomsPerTriplet;
    const int64_t base = triplet_base(tile);
    return atom < 2 ? base + lane * (2 * kLaneRecordBytes) + atom * kLaneRecordBytes
                    : base + kAtomTwoOffset + lane * kAtomTwoRecordBytes;
}

__device__ __forceinline__ int64_t metadata_offset(int64_t tile, int lane)
{
    return triplet_base(tile) + kAtomTwoOffset + lane * kAtomTwoRecordBytes +
           kAtomTwoMetadata;
}

__device__ __forceinline__ uint32_t load_u32(const uint8_t* pointer)
{
    return *reinterpret_cast<const uint32_t*>(pointer);
}

__device__ __forceinline__ void wave_min(float& error, int& index)
{
#pragma unroll
    for(int offset = 32; offset > 0; offset >>= 1)
    {
        const float other_error = __shfl_down(error, offset);
        const int other_index = __shfl_down(index, offset);
        if(other_error < error || (other_error == error && other_index < index))
        {
            error = other_error;
            index = other_index;
        }
    }
}

__global__ __launch_bounds__(64) void iq2r_encode_assign_kernel(
    const float* __restrict__ weight,
    const float* __restrict__ importance,
    const float* __restrict__ codebook,
    float codebook_max,
    int N,
    int storage_K,
    int valid_K,
    int exponent_radius,
    uint16_t* __restrict__ indices,
    uint8_t* __restrict__ data,
    uint8_t* __restrict__ scales)
{
#if defined(__gfx950__)
    const int logical = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int blocks_per_row = valid_K / kScaleBlock;
    const int row = logical / blocks_per_row;
    const int block_k = logical % blocks_per_row;
    if(row >= N)
        return;
    const int k_base = block_k * kScaleBlock;
    const float value =
        lane < kScaleBlock ? weight[static_cast<int64_t>(row) * storage_K + k_base + lane]
                           : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for(int offset = 32; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down(maximum, offset));
    maximum = __shfl(maximum, 0);
    const int center = static_cast<int>(nearbyintf(
        log2f(fmaxf(maximum, 0x1p-126f) / fmaxf(codebook_max, 1.0e-30f))));

    float best_total = INFINITY;
    int best_exponent = 0;
    int best_indices[4] = {0, 0, 0, 0};
    const int candidate_count = 2 * exponent_radius + 1;
    for(int candidate = 0; candidate < candidate_count; ++candidate)
    {
        const int exponent = min(127, max(-126, center + candidate - exponent_radius));
        const float scale = exp2f(static_cast<float>(exponent));
        float total = 0.0f;
        int candidate_indices[4] = {0, 0, 0, 0};
#pragma unroll
        for(int group = 0; group < 4; ++group)
        {
            float local_error = INFINITY;
            int local_index = 0;
#pragma unroll
            for(int slot = 0; slot < 8; ++slot)
            {
                const int codebook_index = lane + slot * 64;
                float error = 0.0f;
#pragma unroll
                for(int element = 0; element < 8; ++element)
                {
                    const int k = k_base + group * 8 + element;
                    const float delta =
                        fabsf(weight[static_cast<int64_t>(row) * storage_K + k]) -
                        codebook[codebook_index * 8 + element] * scale;
                    error = fmaf(delta * delta, importance[k], error);
                }
                if(error < local_error)
                {
                    local_error = error;
                    local_index = codebook_index;
                }
            }
            wave_min(local_error, local_index);
            if(lane == 0)
            {
                total += local_error;
                candidate_indices[group] = local_index;
            }
        }
        if(lane == 0 && total < best_total)
        {
            best_total = total;
            best_exponent = exponent;
#pragma unroll
            for(int group = 0; group < 4; ++group)
                best_indices[group] = candidate_indices[group];
        }
    }
    if(lane != 0)
        return;

    const int row_in_tile = row % kTileN;
    const int n_block = row / kTileN;
    const int k_tile = block_k / 4;
    const int logical_block = block_k % 4;
    const int k_tiles = storage_K / kTileK;
    const int64_t tile = physical_tile(n_block, k_tile, k_tiles);
    // An all-zero block is exact with codeword 0 at any scale. Scale byte 0
    // marks it and keeps it out of its band's base exponent.
    scales[tile * 64 + logical_block * 16 + row_in_tile] =
        maximum == 0.0f ? 0 : static_cast<uint8_t>(best_exponent + 127);
#pragma unroll
    for(int group = 0; group < 4; ++group)
    {
        const int physical_group = 2 * (logical_block % 2) + group / 2;
        const int physical_slot = 2 * (logical_block / 2) + group % 2;
        const int physical_lane = physical_group * 16 + row_in_tile;
        indices[(tile * 64 + physical_lane) * 4 + physical_slot] =
            static_cast<uint16_t>(best_indices[group]);
        uint8_t sign = 0;
#pragma unroll
        for(int element = 0; element < 8; ++element)
        {
            const float item = weight[static_cast<int64_t>(row) * storage_K +
                                      k_base + group * 8 + element];
            sign |= static_cast<uint8_t>(signbit(item)) << element;
        }
        data[lane_record_offset(tile, physical_lane) + 4 + physical_slot] = sign;
    }
#endif
}

__global__ __launch_bounds__(256) void iq2r_encode_base_kernel(
    const uint8_t* __restrict__ scales,
    int N,
    int storage_K,
    int valid_K,
    uint8_t* __restrict__ base_exponents)
{
#if defined(__gfx950__)
    const int n_block = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int n_blocks = N / kTileN;
    if(n_block >= n_blocks)
    {
        if(thread == 0)
            base_exponents[n_block] = 127;
        return;
    }
    const int k_tiles = storage_K / kTileK;
    const int valid_blocks = valid_K / kScaleBlock;
    uint32_t minimum = 255;
    for(int item = thread; item < valid_blocks * 16; item += blockDim.x)
    {
        const int block_k = item / 16;
        const int row_in_tile = item % 16;
        const int k_tile = block_k / 4;
        const int logical_block = block_k % 4;
        const int64_t tile = physical_tile(n_block, k_tile, k_tiles);
        const uint32_t scale = scales[tile * 64 + logical_block * 16 + row_in_tile];
        minimum = min(minimum, scale == 0 ? 255u : scale);
    }
#pragma unroll
    for(int offset = 32; offset > 0; offset >>= 1)
        minimum = min(minimum, __shfl_down(minimum, offset));
    __shared__ uint32_t wave_minimum[4];
    if((thread & 63) == 0)
        wave_minimum[thread / 64] = minimum;
    __syncthreads();
    if(thread < 64)
    {
        minimum = thread < 4 ? wave_minimum[thread] : 255;
#pragma unroll
        for(int offset = 32; offset > 0; offset >>= 1)
            minimum = min(minimum, __shfl_down(minimum, offset));
        if(thread == 0)  // 255: every block of the band is zero
            base_exponents[n_block] = static_cast<uint8_t>(minimum == 255 ? 127 : minimum);
    }
#endif
}

__global__ __launch_bounds__(256) void iq2r_encode_pack_kernel(
    const uint16_t* __restrict__ indices,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ base_exponents,
    int64_t triplets,
    int N,
    int storage_K,
    int valid_K,
    uint8_t* __restrict__ data,
    int32_t* __restrict__ scale_delta_overflow)
{
#if defined(__gfx950__)
    const int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(item >= triplets * 64)
        return;
    const int64_t triplet_index = item / 64;
    const int lane = static_cast<int>(item % 64);
    const int k_tiles = storage_K / kTileK;
    const int64_t group_k = triplet_index / 2;
    const int triplet_in_group = static_cast<int>(triplet_index % 2);
    const int n_group = static_cast<int>(group_k / k_tiles);
    const int k_tile = static_cast<int>(group_k % k_tiles);
    const int64_t base = group_k * kGroupBytes + triplet_in_group * kTripletBytes;
    uint32_t metadata = 0;
#pragma unroll
    for(int atom = 0; atom < kAtomsPerTriplet; ++atom)
    {
        const int64_t tile = group_k * kNBlocksPerGroup +
                             triplet_in_group * kAtomsPerTriplet + atom;
        uint32_t high_bits = 0;
#pragma unroll
        for(int slot = 0; slot < 4; ++slot)
        {
            const uint32_t value = indices[(tile * 64 + lane) * 4 + slot];
            data[lane_record_offset(tile, lane) + slot] = static_cast<uint8_t>(value);
            high_bits |= ((value >> 8) & 1u) << slot;
        }
        metadata |= high_bits << (atom * 8);
        const int n_block = n_group * kNBlocksPerGroup +
                            triplet_in_group * kAtomsPerTriplet + atom;
        const int block_k = k_tile * 4 + lane / 16;
        if(n_block < N / kTileN && block_k < valid_K / kScaleBlock)
        {
            const int scale = scales[tile * 64 + lane];
            const int base_exponent = base_exponents[n_block];
            const int delta = scale - base_exponent;
            if(delta > 15)
                atomicExch(scale_delta_overflow, 1);
            metadata |= static_cast<uint32_t>(min(max(delta, 0), 15))
                        << (atom * 8 + 4);
        }
    }
    const int64_t destination = base + kAtomTwoOffset + lane * kAtomTwoRecordBytes +
                                kAtomTwoMetadata;
    data[destination] = static_cast<uint8_t>(metadata);
    data[destination + 1] = static_cast<uint8_t>(metadata >> 8);
    data[destination + 2] = static_cast<uint8_t>(metadata >> 16);
#endif
}

__device__ __forceinline__ float decode_fp8(uint8_t bits)
{
    const opus::fp8_t value = __builtin_bit_cast(opus::fp8_t, bits);
    return opus::fp8_to_fp32(value);
}

__global__ void iq2r_materialize_kernel(const uint8_t* __restrict__ data,
                                        const uint8_t* __restrict__ auxiliary,
                                        float* __restrict__ output,
                                        int N,
                                        int K)
{
#if defined(__gfx950__)
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(N) * K;
    if(index >= elements)
        return;

    const int row = static_cast<int>(index / K);
    const int column = static_cast<int>(index % K);
    const int n_block = row / kTileN;
    const int row_in_block = row % kTileN;
    const int k_tile = column / kTileK;
    const int local_k = column % kTileK;
    const int logical_block = local_k / kScaleBlock;
    const int within_block = local_k % kScaleBlock;
    const int lane_group = (logical_block % 2) * 2 + within_block / 16;
    const int lane = lane_group * 16 + row_in_block;
    const int fragment_index = (logical_block / 2) * 16 + within_block % 16;
    const int codeword = fragment_index / 8;
    const int element = fragment_index % 8;
    const int k_tiles = (K + kTileK - 1) / kTileK;
    const int64_t tile = physical_tile(n_block, k_tile, k_tiles);
    const uint8_t* record = data + lane_record_offset(tile, lane);
    const uint32_t metadata = load_u32(data + metadata_offset(tile, lane));
    const int atom = n_block % kAtomsPerTriplet;
    const int high_bits = (metadata >> (atom * 8)) & 0x0f;
    const int codebook_index = record[codeword] |
                               (((high_bits >> codeword) & 1) << 8);
    const uint8_t magnitude = auxiliary[codebook_index * 8 + element];
    const uint8_t signed_value = magnitude ^
                                 (((record[4 + codeword] >> element) & 1) << 7);
    const int scale_lane = logical_block * 16 + row_in_block;
    const uint32_t scale_metadata =
        load_u32(data + metadata_offset(tile, scale_lane));
    const int delta = (scale_metadata >> (atom * 8 + 4)) & 0x0f;
    const int exponent = static_cast<int>(auxiliary[kCodebookBytes + n_block]) + delta;
    const float scale = exponent == 0 ? 0.0f : ldexpf(1.0f, exponent - 127);
    output[index] = decode_fp8(signed_value) * scale;
#endif
}

struct IQ2RCompressedTriplet
{
    uint4 paired;
    uint4 third;
};

union IQ2RActivationFragment
{
    opus::i32x8_t words;
    uint8_t bytes[32];
};

template<int N>
__device__ __forceinline__ void iq2r_wait_vmcnt()
{
    static_assert(N >= 0 && N <= 63);
    constexpr unsigned waitcnt = 0x0f70u | static_cast<unsigned>(N & 0x0f) |
                                 (static_cast<unsigned>(N & 0x30) << 10);
    __builtin_amdgcn_s_waitcnt(waitcnt);
}

template<int AccumulatorBase>
__device__ __forceinline__ void iq2r_triplet_mfma(
    const opus::i32x8_t& activation,
    const opus::i32x8_t* weights,
    opus::vector_t<float, 4>* accumulators,
    uint32_t scale_a,
    uint32_t scale_b)
{
    auto mma = opus::mfma<opus::fp8_t, opus::fp8_t, opus::fp32_t, 16, 16, 128>{};
    accumulators[AccumulatorBase] =
        mma(activation,
            weights[0],
            accumulators[AccumulatorBase],
            scale_a,
            scale_b,
            opus::number<0>{},
            opus::number<0>{});
    accumulators[AccumulatorBase + 1] =
        mma(activation,
            weights[1],
            accumulators[AccumulatorBase + 1],
            scale_a,
            scale_b,
            opus::number<0>{},
            opus::number<1>{});
    accumulators[AccumulatorBase + 2] =
        mma(activation,
            weights[2],
            accumulators[AccumulatorBase + 2],
            scale_a,
            scale_b,
            opus::number<0>{},
            opus::number<2>{});
}

#define GLM53_X_NTT 0
__device__ __forceinline__ IQ2RCompressedTriplet
iq2r_load_compact_triplet(const uint8_t* data, int source_base, int lane)
{
    IQ2RCompressedTriplet result;
    const uint8_t* third = data + source_base + kAtomTwoOffset +
                           lane * kAtomTwoRecordBytes;
#if GLM53_X_NTT
    // Weights stream once: non-temporal, like iq2r_scheduled_load_compact_uniform.
    result.paired = __builtin_bit_cast(uint4, __builtin_nontemporal_load(
        reinterpret_cast<const opus::i32x4_t*>(data + source_base + lane * 16)));
    result.third.x = __builtin_nontemporal_load(reinterpret_cast<const uint32_t*>(third));
    result.third.y = __builtin_nontemporal_load(reinterpret_cast<const uint32_t*>(third + 4));
    result.third.z = __builtin_nontemporal_load(reinterpret_cast<const uint32_t*>(third + 8));
#else
    result.paired = *reinterpret_cast<const uint4*>(
        data + source_base + lane * 16);
    result.third.x = load_u32(third);
    result.third.y = load_u32(third + 4);
    result.third.z = load_u32(third + 8);
#endif
    result.third.w = 0;
    return result;
}

int64_t expected_data_bytes(int64_t n, int64_t k)
{
    const int64_t n_blocks = n / kTileN;
    const int64_t physical_n_blocks = ((n_blocks + 5) / 6) * 6;
    const int64_t k_tiles = (k + kTileK - 1) / kTileK;
    return (physical_n_blocks / 6) * k_tiles * kGroupBytes;
}

int64_t expected_auxiliary_bytes(int64_t n)
{
    const int64_t n_blocks = n / kTileN;
    const int64_t physical_n_blocks = ((n_blocks + 5) / 6) * 6;
    return kCodebookBytes + physical_n_blocks + 3;
}

void validate_weights(const aiter_tensor_t& data,
                      const aiter_tensor_t& auxiliary,
                      int64_t logical_n,
                      int64_t logical_k)
{
    AITER_CHECK(data.is_gpu() && auxiliary.is_gpu(), "IQ2R weights must be GPU tensors");
    AITER_CHECK(data.device_id == auxiliary.device_id,
                "IQ2R data and auxiliary must be on the same device");
    AITER_CHECK(data.dtype() == AITER_DTYPE_u8 && auxiliary.dtype() == AITER_DTYPE_u8,
                "IQ2R data and auxiliary must be uint8");
    AITER_CHECK(data.dim() == 2 && auxiliary.dim() == 2 &&
                    data.size(0) == auxiliary.size(0),
                "IQ2R data and auxiliary must be matching [experts,bytes]");
    AITER_CHECK(data.is_contiguous() && auxiliary.is_contiguous(),
                "IQ2R data and auxiliary must be contiguous");
    AITER_CHECK(logical_n > 0 && logical_n % 16 == 0 && logical_k > 0 &&
                    logical_k % 32 == 0,
                "IQ2R requires N%16==0 and K%32==0");
    AITER_CHECK(data.size(1) == expected_data_bytes(logical_n, logical_k),
                "IQ2R data byte count does not match logical dimensions");
    AITER_CHECK(auxiliary.size(1) == expected_auxiliary_bytes(logical_n),
                "IQ2R auxiliary byte count does not match logical dimensions");
}

// GLM-5.3 IQ2R packed MoE kernels.

constexpr int kQuadBytes=2304;

struct IQ2RCompressedQuad { uint4 first; uint4 second; uint32_t metadata; };

__device__ __forceinline__ void iq2r_issue_quad(
    opus::gmem<uint8_t>& buffer,uint8_t* cache,int lane,int source_base)
{
    const int base=__builtin_amdgcn_readfirstlane(source_base);
    buffer.template async_load<16>(cache,lane*16,base,opus::number<0>{},opus::number<0>{});
    buffer.template async_load<16>(cache+1024,lane*16,base+1024,opus::number<0>{},opus::number<0>{});
    buffer.template async_load<4>(cache+2048,lane*4,base+2048,opus::number<0>{},opus::number<0>{});
    asm volatile("" ::: "memory");
}

__device__ __forceinline__ IQ2RCompressedQuad iq2r_read_quad(const uint8_t* cache,int lane)
{
    return {*reinterpret_cast<const uint4*>(cache+lane*16),
            *reinterpret_cast<const uint4*>(cache+1024+lane*16),
            *reinterpret_cast<const uint32_t*>(cache+2048+lane*4)};
}

using iq2r_scheduled_i32x3 = int __attribute__((ext_vector_type(3)));

__device__ iq2r_scheduled_i32x3 iq2r_scheduled_load_dwordx3(opus::i32x4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.v3i32");

__device__ opus::i32x4_t iq2r_scheduled_load_dwordx4(opus::i32x4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ __forceinline__ IQ2RCompressedTriplet iq2r_scheduled_load_compact_uniform(
    const uint8_t* data, int data_bytes, int base, int lane)
{
    const uint64_t address = reinterpret_cast<uint64_t>(data);
    const opus::i32x4_t rsrc = {
        static_cast<int>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address))),
        static_cast<int>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address >> 32))),
        __builtin_amdgcn_readfirstlane(data_bytes),
        static_cast<int>(opus::buffer_default_config())};
    const int scalar_base = __builtin_amdgcn_readfirstlane(base);
    IQ2RCompressedTriplet out;
    out.paired = __builtin_bit_cast(uint4, iq2r_scheduled_load_dwordx4(rsrc, lane*16, scalar_base, 2));
    const auto third = iq2r_scheduled_load_dwordx3(rsrc, lane*12+kAtomTwoOffset, scalar_base, 2);
    out.third = make_uint4(third[0], third[1], third[2], 0);
    asm volatile("" ::: "memory");
    return out;
}

#define GLM53_X_NT 2
__device__ int iq2r_scheduled_buffer_load1(opus::i32x4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ unsigned char iq2r_buffer_load_u8(opus::i32x4_t, int, int, int)
    __asm("llvm.amdgcn.raw.buffer.load.i8");

// Buffer descriptor of ``bytes`` bytes at ``p`` (wave-uniform).
__device__ __forceinline__ opus::i32x4_t iq2r_buffer_rsrc(const void* p, int bytes)
{
    const uint64_t address=reinterpret_cast<uint64_t>(p);
    return {static_cast<int>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address))),
            static_cast<int>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address>>32))),
            __builtin_amdgcn_readfirstlane(bytes),
            static_cast<int>(opus::buffer_default_config())};
}

__device__ __forceinline__ IQ2RCompressedQuad iq2r_scheduled_load_quad(
    const uint8_t* data, int data_bytes, int base, int lane)
{
    const uint64_t address=reinterpret_cast<uint64_t>(data);
    const opus::i32x4_t rsrc={
        static_cast<int>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address))),
        static_cast<int>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address>>32))),
        __builtin_amdgcn_readfirstlane(data_bytes),
        static_cast<int>(opus::buffer_default_config())};
    const int sb=__builtin_amdgcn_readfirstlane(base);
    IQ2RCompressedQuad out;
    out.first=__builtin_bit_cast(uint4,iq2r_scheduled_load_dwordx4(rsrc,lane*16,sb,GLM53_X_NT));
    out.second=__builtin_bit_cast(uint4,iq2r_scheduled_load_dwordx4(rsrc,lane*16+1024,sb,GLM53_X_NT));
    out.metadata=iq2r_scheduled_buffer_load1(rsrc,lane*4+2048,sb,GLM53_X_NT);
    asm volatile("" ::: "memory");
    return out;
}

// IQ2R-v2 (aiter/ops/iq2r_v2.py, iq2r_glm53_pack_v2). Classes == 0 is IQ2R:
// a metadata high nibble is an exponent delta on the N block's base byte.
// Classes > 0 is IQ2R-v2: ``bases`` holds the lane's own E8M0 d bytes for the
// current K256 (one per atom), and the high nibble of lane group g is
// exponent(block g) * Classes + class(block s(g)), s = (0, 2, 1, 3). Gate/up
// kernels keep two books (gate rows, up rows), the up copies after the gate's. The MFMA
// scale is d + exponent; codewords come from codebook copy ``class`` (copies
// 4 KB apart in LDS). A data lane's codewords 0-1 belong to block g/2 and 2-3
// to block g/2 + 2: with s, those classes sit in lane groups 2*(g/2) and
// 2*(g/2) + 1, which one permlane16 swap of the metadata word delivers.
struct IQ2RLevels { uint32_t low, high; };

#define GLM53_X_LVM 1
#define GLM53_X_R9SB 1
#define GLM53_X_R9AX 0
#define GLM53_X_R9AL 1
#define GLM53_X_R9SS 1
#define GLM53_X_R9LATE 0
#define GLM53_X_R9LAT 1
#define GLM53_X_R9GRP 0
#define GLM53_X_PRB 0
#define GLM53_X_TS 0
#define GLM53_X_R9PIPE 1
#define GLM53_X_R9X 1
#define GLM53_X_R9GEN 3
#define GLM53_X_GPERM 1
#define GLM53_X_PFA 1
#define GLM53_X_GG 2
#define GLM53_X_CHUNK 1
#define GLM53_X_STG4 0
#define GLM53_X_STG0 0
#define GLM53_X_PF 0
#define GLM53_X_GEB 0
#define GLM53_X_GTS 0
#define GLM53_X_XQ 0
#define GLM53_X_PALIAS 0
#define GLM53_X_OPSEL 1
#define GLM53_X_UNR 1
#define GLM53_X_GEF 0
template<int Classes>
constexpr int iq2r_log2_classes() { return Classes==8?3:Classes==4?2:Classes==2?1:0; }

// Bytes of class << 4 | book << (4 + log2 Classes) per atom, for the blocks of
// a data lane's codewords 0-1 (low) and 2-3 (high). ``book_bits`` selects the
// up codebook in gate/up lanes (iq2r_book_bits). Shifted left by 8 such a byte
// is the byte offset of the codebook copy, see iq2r_copy_offset.
template<int Classes>
__device__ __forceinline__ IQ2RLevels iq2r_levels(uint32_t metadata,uint32_t book_bits=0u)
{
    if constexpr(Classes==0) return {0u,0u};
    else
    {
        constexpr uint32_t mask=static_cast<uint32_t>(Classes-1)*0x10101010u;
        // permlane16_swap(x, x) -> rows (x0,x0,x2,x2), (x1,x1,x3,x3).
        if constexpr(GLM53_X_LVM)
        {
            // Mask before the swap (book_bits depends on lane % 16 only, which
            // the swap keeps): one mask for both rows.
            const uint32_t bits=(metadata&mask)|book_bits;
            const auto rows=__builtin_amdgcn_permlane16_swap(bits,bits,false,true);
            return {static_cast<uint32_t>(rows[0]),static_cast<uint32_t>(rows[1])};
        }
        const auto rows=__builtin_amdgcn_permlane16_swap(metadata,metadata,false,true);
        return {(static_cast<uint32_t>(rows[0])&mask)|book_bits,(static_cast<uint32_t>(rows[1])&mask)|book_bits};
    }
}

template<int Classes>
__device__ __forceinline__ uint32_t iq2r_book_bits(int lane_row)
{
    if constexpr(Classes==0) return 0u;
    else return (lane_row&1)?(0x01010101u<<(4+iq2r_log2_classes<Classes>())):0u;
}

// Byte offset of the codebook copy for ``atom``: one v_perm moves the atom's
// byte to byte 1 (x 256) and zeroes the rest.
template<int Classes>
__device__ __forceinline__ uint32_t iq2r_copy_offset(uint32_t levels,int atom,uint32_t high=0u)
{
    if constexpr(Classes==0) return 0u;
    else return __builtin_amdgcn_perm(high,levels,0x0c04000cu|(static_cast<uint32_t>(atom)<<8));
}

// E8M0 weight scales of the lane's own 32-blocks, one byte per atom. IQ2R:
// base + delta per atom (see iq2r_scale_b). IQ2R-v2: d + exponent, added
// byte-wise for all atoms at once (no byte overflows: d <= 254 - exponent).
template<int Classes>
__device__ __forceinline__ uint32_t iq2r_scale_bytes(uint32_t bases,uint32_t metadata)
{
    constexpr int shift=4+iq2r_log2_classes<Classes>();
    constexpr uint32_t mask=static_cast<uint32_t>((16>>iq2r_log2_classes<Classes>())-1)*0x01010101u;
    return bases+((metadata>>shift)&mask);
}

// E8M0 weight scale of the lane's own 32-block of ``atom``.
template<int Classes>
__device__ __forceinline__ uint32_t iq2r_scale_b(uint32_t bases,uint32_t metadata,int atom)
{
    if constexpr(Classes==0)
        return ((bases>>(atom*8))&255u)+((metadata>>(atom*8+4))&15u);
    else
        return (iq2r_scale_bytes<Classes>(bases,metadata)>>(atom*8))&255u;
}

// Codebook words a packed auxiliary holds: one 512-entry book, or for IQ2R-v2
// Classes copies of each of Books books.
template<int Classes,int Books>
constexpr int iq2r_codebook_words() { return (Classes==0?1:Classes*Books)*512; }

// Stage an expert's codebook (IQ2R-v2: Classes copies per book) in LDS with
// a global -> LDS DMA: 1 KB per wave instruction, no VGPRs, overlapped with
// the loads issued before the caller's s_waitcnt vmcnt(0) + barrier.
template<int Words,int Waves>
__device__ __forceinline__ void iq2r_stage_codebook_async(uint64_t* lds,const uint8_t* aux,int wave,int lane)
{
    constexpr int Chunks=Words*8/1024;
    opus::gmem<uint8_t> buffer(aux,static_cast<unsigned int>(Words*8));
    wave=__builtin_amdgcn_readfirstlane(wave);
#pragma unroll
    for(int i=0;i<(Chunks+Waves-1)/Waves;++i)
    {
        const int chunk=wave+i*Waves;
        if(Chunks%Waves==0 || chunk<Chunks)
            buffer.template async_load<16>(reinterpret_cast<uint8_t*>(lds)+chunk*1024,lane*16,chunk*1024,
                                           opus::number<0>{},opus::number<0>{});
    }
    asm volatile("" ::: "memory");
}

// IQ2R-v2: copy 0 of a book is the codebook itself (m[0] == 1), so a wave can
// load that 4 KB alone and form copies 1..C-1 = E4M3(FP32(codebook) * m[c]) in
// LDS with the packer's two steps (FP32 product, round to nearest even). The
// conversion does not saturate, so this needs products below 464: codebook
// entries are at most 256. base[i] holds bytes (i*64+lane)*16 of copy 0.
template<int Classes>
__device__ __forceinline__ void iq2r_v2_form_copies(uint64_t* lds,const uint4 (&base)[4],int lane)
{
    static_assert(Classes==4,"c4e4q multipliers 2^(c/4)");
    const uint32_t m_bits[4]={0x3f800000u,0x3f9837f0u,0x3fb504f3u,0x3fd744fdu};
    uint4* out=reinterpret_cast<uint4*>(lds);
#pragma unroll
    for(int i=0;i<4;++i)
    {
        // keeps the conversions after the caller's later loads are issued
        uint32_t w[4]={base[i].x,base[i].y,base[i].z,base[i].w};
        asm volatile("" : "+v"(w[0]),"+v"(w[1]),"+v"(w[2]),"+v"(w[3]));
        out[i*64+lane]=make_uint4(w[0],w[1],w[2],w[3]);
        if constexpr(GLM53_X_GPERM)
        {
            // normal bytes 0x08..0x78: E4M3(b * 2^(c/4)) = b + delta_c[mantissa]
            uint32_t m[4],r1[4],r2[4],r3[4];
#pragma unroll
            for(int d=0;d<4;++d)
            {
                m[d]=w[d]&0x07070707u;
                r1[d]=w[d]+0x02020202u;
                r2[d]=w[d]+__builtin_amdgcn_perm(0x04040404u,0x05040403u,m[d]);
                r3[d]=w[d]+__builtin_amdgcn_perm(0x06060606u,0x06060605u,m[d]);
            }
            out[256+i*64+lane]=make_uint4(r1[0],r1[1],r1[2],r1[3]);
            out[512+i*64+lane]=make_uint4(r2[0],r2[1],r2[2],r2[3]);
            out[768+i*64+lane]=make_uint4(r3[0],r3[1],r3[2],r3[3]);
            __builtin_amdgcn_sched_barrier(0);
            continue;
        }
        opus::vector_t<float,2> v[8];
#pragma unroll
        for(int d=0;d<4;++d)
        {
            v[2*d]=__builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w[d]),false);
            v[2*d+1]=__builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w[d]),true);
        }
        uint32_t r[4]={w[0],w[1],w[2],w[3]};  // each conversion overwrites both halves
#pragma unroll
        for(int c=1;c<Classes;++c)
        {
            const float mc=__builtin_bit_cast(float,m_bits[c]);
            const opus::vector_t<float,2> m2={mc,mc};
#pragma unroll
            for(int d=0;d<4;++d)
            {
                const opus::vector_t<float,2> lo=v[2*d]*m2,hi=v[2*d+1]*m2;
                int x=__builtin_amdgcn_cvt_pk_fp8_f32(lo[0],lo[1],static_cast<int>(r[d]),false);
                r[d]=static_cast<uint32_t>(__builtin_amdgcn_cvt_pk_fp8_f32(hi[0],hi[1],x,true));
            }
            out[c*256+i*64+lane]=make_uint4(r[0],r[1],r[2],r[3]);
            __builtin_amdgcn_sched_barrier(0);  // one chunk's copy at a time: few live VGPRs
        }
    }
}

// Copies 0..3 of chunk ``index`` (16 bytes of each 4 KB copy) of a c4e4q
// book from copy 0: a normal E4M3 byte b (0x08..0x78) times 2^(c/4) rounds
// to b + delta_c[b & 7], and no byte carries into the next.
__device__ __forceinline__ void iq2r_v2_form_chunk(uint4* book,int index,uint4 base)
{
    const uint32_t w[4]={base.x,base.y,base.z,base.w};
    uint32_t r1[4],r2[4],r3[4];
    // materialized here rather than hoisted into VGPRs held by the caller's loops
    uint32_t k2=0x05040403u,k3=0x06060605u;
    asm volatile("" : "+v"(k2),"+v"(k3));
#pragma unroll
    for(int d=0;d<4;++d)
    {
        const uint32_t m=w[d]&0x07070707u;
        r1[d]=w[d]+0x02020202u;
        r2[d]=w[d]+__builtin_amdgcn_perm(0x04040404u,k2,m);
        r3[d]=w[d]+__builtin_amdgcn_perm(0x06060606u,k3,m);
    }
    book[index]=base;
    book[256+index]=make_uint4(r1[0],r1[1],r1[2],r1[3]);
    book[512+index]=make_uint4(r2[0],r2[1],r2[2],r2[3]);
    book[768+index]=make_uint4(r3[0],r3[1],r3[2],r3[3]);
}

// Gate/up books in LDS from their copies 0 (2 x 4 KB, 8 chunks of 1 KB):
// load first, form after the caller's other loads are issued.
template<int Waves>
__device__ __forceinline__ void iq2r_v2_gate_base_load(uint4 (&base)[8/Waves],const uint8_t* aux,int wave,int lane)
{
#pragma unroll
    for(int j=0;j<8/Waves;++j)
    {
        const int c=wave+j*Waves;
        base[j]=reinterpret_cast<const uint4*>(aux)[(c>>2)*1024+(c&3)*64+lane];
    }
    asm volatile("" ::: "memory");
}

template<int Waves>
__device__ __forceinline__ void iq2r_v2_gate_form(uint64_t* lds,const uint4 (&base)[8/Waves],int wave,int lane)
{
#pragma unroll
    for(int j=0;j<8/Waves;++j)
    {
        const int c=wave+j*Waves;
        iq2r_v2_form_chunk(reinterpret_cast<uint4*>(lds)+(c>>2)*1024,(c&3)*64+lane,base[j]);
    }
}

// Gate/up copies 0 (2 x 4 KB) DMA'd into an 8 KB LDS stage; chunk c lands
// at stage[c * 64 + lane], read back by the thread that loaded it.
template<int Waves>
__device__ __forceinline__ void iq2r_v2_gate_base_dma(uint4* stage,const uint8_t* aux,int wave,int lane)
{
    opus::gmem<uint8_t> buffer(aux,static_cast<unsigned int>(20*1024));
    wave=__builtin_amdgcn_readfirstlane(wave);
#pragma unroll
    for(int j=0;j<8/Waves;++j)
    {
        const int c=wave+j*Waves;
        buffer.template async_load<16>(reinterpret_cast<uint8_t*>(stage)+c*1024,lane*16,((c>>2)*16+(c&3))*1024,
                                       opus::number<0>{},opus::number<0>{});
    }
    asm volatile("" ::: "memory");
}

// IQ2R-v2 output gains (even rows, odd rows): the auxiliary's last 8 bytes.
template<int Classes>
__device__ __forceinline__ float iq2r_gain(const uint8_t* aux,int auxiliary_bytes,int row)
{
    if constexpr(Classes==0) return 1.0f;
    else return reinterpret_cast<const float*>(aux+auxiliary_bytes-8)[row&1];
}

// Both gains of a workgroup-uniform expert: a scalar load, so they stay in
// SGPRs until the epilogue selects the lane's row (iq2r_row_gain).
template<int Classes>
__device__ __forceinline__ float2 iq2r_gains(const uint8_t* aux,int auxiliary_bytes)
{
    if constexpr(Classes==0) return make_float2(1.0f,1.0f);
    else return *reinterpret_cast<const float2*>(aux+auxiliary_bytes-8);
}

__device__ __forceinline__ float iq2r_row_gain(float2 gains,int row) { return (row&1)?gains.y:gains.x; }

// IQ2R-v2 d word of one lane: the table follows the codebook copies,
// [blocks][K/256][16][4] bytes.
template<int Classes,int Books>
__device__ __forceinline__ const uint32_t* iq2r_v2_bases_at(const uint8_t* aux,int block,int columns,int column,int lane_row)
{
    return reinterpret_cast<const uint32_t*>(aux+iq2r_codebook_words<Classes,Books>()*8)
        +(block*columns+column)*16+lane_row;
}

template<int Classes,int Books>
__device__ __forceinline__ uint32_t iq2r_v2_bases(const uint8_t* aux,int block,int columns,int column,int lane_row)
{
    return *iq2r_v2_bases_at<Classes,Books>(aux,block,columns,column,lane_row);
}

// Packed gate records: 9-bit packed indices, sign bit planes and a sign-free
// codebook (see iq2r_glm53_pack in aiter/iq2r_glm53.py).
template<int MAtoms,int Classes=0>
__device__ __forceinline__ void glm53_quad_decode_mfma(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m);

template<int NT,int Classes=0>
__global__ __launch_bounds__(512,1) void glm53_gate_m1_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int MAtoms=1;
    constexpr int K=6144, Rows=16*MAtoms;
    constexpr int BookWords=iq2r_codebook_words<Classes,2>();
    struct Storage {
        alignas(16) uint64_t codebook[BookWords];
        alignas(16) opus::vector_t<float,4> partial[8][4][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*NT;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int task=work%num_tasks;
        const int n_tile=work/num_tasks;
        const int row_begin=tasks[task*3];
        const int row_end=min(row_begin+Rows,tasks[task*3]+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting the codebook and partials.
        // The codebook DMA, scales, gain and first weights load together.
        [[maybe_unused]] uint4 gate_base[1];
        if constexpr(Classes>0 && GLM53_X_GG) iq2r_v2_gate_base_load<8>(gate_base,aux,wave,lane);
        else iq2r_stage_codebook_async<BookWords,8>(shared.codebook,aux,wave,lane);
        const int nbase=n_tile*4;
        uint32_t bases=0;
        if constexpr(Classes==0)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        }
        // IQ2R-v2: gate rows use book 0, up rows book 1; d words per K256.
        const uint64_t* lane_book=shared.codebook;
        uint32_t d_words[3]={};
        if constexpr(Classes>0)
        {
#pragma unroll
            for(int c=0;c<3;++c) d_words[c]=iq2r_v2_bases<Classes,2>(aux,n_tile,24,wave*3+c,lane_row);
        }
        const float gain=iq2r_gain<Classes>(aux,auxiliary_bytes,lane_row);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
        if constexpr(Classes>0 && GLM53_X_GG) iq2r_v2_gate_form<8>(shared.codebook,gate_base,wave,lane);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
#pragma unroll 6
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            const auto compressed=pending;
            if(iteration<5)
            {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            glm53_quad_decode_mfma<MAtoms,Classes>(compressed,lane_book,Classes==0?bases:d_words[iteration/2],a,sa,accumulators,MAtoms);
        }
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                shared.partial[wave][atom][lane]=accumulators[m][atom];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=shared.partial[source][atom][lane][wave];
                    values[atom]=__bfloat162float(__float2bfloat16(value*gain));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(lane_row%2==0)
                {
                    float activated[4];float abs_max=aiter::kAmaxFloor;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*(NT*32)+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*NT+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

#ifndef GLM53_GATE_BOOKS
#define GLM53_GATE_BOOKS 2
#endif
#ifndef GLM53_GATE_DOUBLE
#define GLM53_GATE_DOUBLE 0
#endif
#ifndef GLM53_X_NOCOPY
#define GLM53_X_NOCOPY 0
#endif
#ifndef GLM53_X_NOD
#define GLM53_X_NOD 0
#endif
#ifndef GLM53_X_ORDER
#define GLM53_X_ORDER 0
#endif
#ifndef GLM53_X_DPRE
#define GLM53_X_DPRE 0
// GLM53_X_BLK: a v2 gate CTA takes a contiguous run of its XCD's items, so
// consecutive items share a task (and expert) and its codebook.
#define GLM53_X_BLK 0
// GLM53_X_NOFORM: timing probe only, the v2 gate book is never staged.
#define GLM53_X_NOFORM 0
// GLM53_X_LDSB: waves 4-7 DMA the next expert's gate/up copies 0 into an
// 8 KB LDS stage during the K loop and form its book in the epilogue.
#define GLM53_X_LDSB 0
// GLM53_X_ABUF: the v2 gate K loop loads activations, their scales and the
// d words through buffer descriptors with the K offset in an SGPR, and keeps
// the quad offset scalar: no per-iteration 64-bit address arithmetic.
#define GLM53_X_ABUF 1
// GLM53_X_PF2: the v2 gate K loop keeps two quads in flight, so a quad load
// has a whole iteration before the carried lookups of its first atom.
#define GLM53_X_PF2 0
#endif
#ifndef GLM53_X_DBC
// GLM53_X_DBC: the v2 gate loads a wave's 48 d words once per item (lane
// 16c+r: column c, row r) and permlane swaps broadcast each column to all four
// rows, instead of a d load every K iteration. Written, never built or timed.
#define GLM53_X_DBC 0
#define GLM53_X_R9T 4
#define GLM53_X_R9W 0
#define GLM53_X_R9ROT 0
#define GLM53_X_R9VG 0
#define GLM53_X_R9ND 0
#define GLM53_X_R9SLP 0
#define GLM53_X_SGN 0
#define GLM53_X_R9RFL 1
#define GLM53_X_R9DROP 0
#define GLM53_X_R12 0
#define GLM53_X_R9IDLE 0
#define GLM53_X_R9BAR 0
#define GLM53_X_R9HELP 0
#define GLM53_X_R9EB 0
#define GLM53_X_R9SR 1
#define GLM53_X_R9ADDR 1
#define GLM53_X_RB 1
#define GLM53_X_R9STG 0
#define GLM53_X_R9ORD 0
#define GLM53_X_R9PRO 1
#define GLM53_X_R9S8 0
#define GLM53_X_S8W 0
#define GLM53_X_S8ND 0
#define GLM53_X_S8IL 1
#define GLM53_X_S8NB 0
#define GLM53_X_S8KO 1
#define GLM53_X_T8KO 1
#define GLM53_X_DND 0
#define GLM53_X_A32 1
#define GLM53_X_T8LV 1
#define GLM53_X_S8OCC 3
#endif

__device__ __forceinline__ int glm53_remap8(int index,int total)
{
 const int per=(total+7)/8,tall=total%8==0?8:total%8;
 const int xcd=index%8,local=index/8;
 return xcd<tall?xcd*per+local:tall*per+(xcd-tall)*(per-1)+local;
}

// Prefill gate: waves cover N, while a shared K128 activation tile is reused
// across the CTA. K is summed as eight K768 partials, like the decode gate.
template<int Waves,int XCD,int TaskGroup,int Unroll,int NT,int Classes=0>
__global__ __launch_bounds__(64*Waves,1) void glm53_gate_prefill_kernel(
    const opus::fp8_t* __restrict__ activations,const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int Rows=64,MAtoms=4,Threads=Waves*64;
    constexpr int BookWords=iq2r_codebook_words<Classes,2>();
    struct Storage {
        alignas(16) uint64_t codebook[BookWords];
        int row_ids[Rows];
        uint8_t scales[2*Rows*4];
        union {
            alignas(16) uint8_t input[2*Rows*128];
            alignas(16) __hip_bfloat16 gate[16][Waves*64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(NT/Waves);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(NT/Waves);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(NT/Waves)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(NT/Waves));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile*Waves+wave;
        const int begin=tasks[task*3],count=tasks[task*3+1],expert=tasks[task*3+2];
        if(begin<0 || count<=0 || count>64 || begin+count>routes || expert<0 || expert>=257)continue;
        const int active_m=(count+15)/16;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(expert!=previous_expert)
        {
            // IQ2R-v2's 32 KB lands by the first K step's vmcnt(0) + barrier.
            if constexpr(Classes==0 || !GLM53_X_PFA) { for(int i=linear;i<BookWords;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i]; }
            else iq2r_stage_codebook_async<BookWords,Waves>(shared.codebook,aux,wave,lane);
            previous_expert=expert;
        }
        if(linear<Rows)shared.row_ids[linear]=gather[begin+min(linear,count-1)]/9;
        __syncthreads();
        uint32_t bases=0;
        if constexpr(Classes==0)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_tile*4+atom])<<(atom*8);
        }
        const uint64_t* lane_book=shared.codebook;
        opus::gmem<uint8_t> a_buffer(reinterpret_cast<const uint8_t*>(activations),(routes/9)*6144);
        opus::gmem<uint8_t> s_buffer(scales,(routes/9)*192);
        constexpr int Copies=Rows*128/(Threads*16);
        int source_a[Copies];
#pragma unroll
        for(int copy=0;copy<Copies;++copy)
        {
            const int offset=(linear+copy*Threads)*16;
            const int r=offset/128,col=(offset%128)^(((r>>1)&7)<<4);
            source_a[copy]=shared.row_ids[r]*6144+col;
        }
        const int source_s=shared.row_ids[lane]*192;
        auto issue_a=[&](int kt,int slot)
        {
#pragma unroll
            for(int copy=0;copy<Copies;++copy)
            {
                // The LDS base is uniform within a wave. Hardware adds
                // lane*16; apply the row swizzle in the global source column.
                const int destination=slot*Rows*128+(wave*64+copy*Threads)*16;
                a_buffer.template async_load<16>(shared.reuse.input+destination,source_a[copy]+kt*128);
            }
            if(wave==0)
                s_buffer.template async_load<4>(shared.scales+slot*Rows*4,source_s+kt*4);
            asm volatile("" ::: "memory");
        };
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,n_tile*48*kQuadBytes,lane);
        // IQ2R-v2 d words load one K256 ahead, with the records.
        uint32_t pending_d=0;
        if constexpr(Classes>0) pending_d=iq2r_v2_bases<Classes,2>(aux,n_tile,24,0,lane_row);
        issue_a(0,0);
        opus::vector_t<float,4> total[MAtoms][4]={};
#pragma unroll 1
        for(int partition=0;partition<8;++partition)
        {
            opus::vector_t<float,4> partial[MAtoms][4]={};
#pragma unroll Unroll
            for(int iter=0;iter<6;++iter)
            {
                const int kt=partition*6+iter;
                const int slot=kt&1;
                const IQ2RCompressedQuad compressed=pending;
                // The previous iteration issued the current A/scales. All
                // waves must finish consuming the older slot before reuse.
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                __syncthreads();
                IQ2RActivationFragment a[MAtoms];uint32_t sa[MAtoms];
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    const int r=m*16+lane_row,col=lane_group*16;
                    *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(shared.reuse.input+slot*Rows*128+((r*128+col)^(((r>>1)&7)<<4)));
                    *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(shared.reuse.input+slot*Rows*128+((r*128+col+64)^(((r>>1)&7)<<4)));
                    sa[m]=shared.scales[slot*Rows*4+r*4+lane_group];
                }
                if constexpr(Classes>0) bases=pending_d;
                if(kt<47)
                {
                    issue_a(kt+1,1-slot);
                    pending=iq2r_scheduled_load_quad(data,data_bytes,(n_tile*48+kt+1)*kQuadBytes,lane);
                    if constexpr(Classes>0)
                        if((kt&1)==1) pending_d=iq2r_v2_bases<Classes,2>(aux,n_tile,24,(kt+1)/2,lane_row);
                }
                glm53_quad_decode_mfma<MAtoms,Classes>(compressed,lane_book,bases,a,sa,partial,active_m);
            }
#pragma unroll
            for(int m=0;m<MAtoms;++m)
#pragma unroll
                for(int atom=0;atom<4;++atom)
                    total[m][atom]+=partial[m][atom];
        }
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
#pragma unroll
            for(int atom=0;atom<4;++atom)
#pragma unroll
                for(int component=0;component<4;++component)
                    shared.reuse.gate[lane_group*4+component][wave*64+atom*16+lane_row]=__float2bfloat16(total[m][atom][component]*iq2r_gain<Classes>(aux,auxiliary_bytes,lane_row));
            __syncthreads();
            const int local_row=linear/(Waves*4),quad=(linear/4)%Waves,qlane=linear%4;
            const int row=begin+m*16+local_row;
            float values[8];float abs_max=aiter::kAmaxFloor;
#pragma unroll
            for(int element=0;element<8;++element)
            {
                const int col=quad*64+(qlane*8+element)*2;
                const float gate=__bfloat162float(shared.reuse.gate[local_row][col]);
                const float up=__bfloat162float(shared.reuse.gate[local_row][col+1]);
                values[element]=__bfloat162float(__float2bfloat16((gate/(1.0f+__expf(-gate)))*(up+0.0f)));
                abs_max=fmaxf(abs_max,fabsf(values[element]));
            }
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,1));
            abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
            if(row<begin+count)
            {
                const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                const float inverse=1.0f/bs.dq_scale;
                opus::vector_t<opus::fp8_t,8> q;
#pragma unroll
                for(int element=0;element<8;++element)q[element]=opus::fp32_to_fp8(values[element]*inverse);
                const int out_quad=output_tile*Waves+quad;
                *reinterpret_cast<opus::vector_t<opus::fp8_t,8>*>(output+static_cast<int64_t>(row)*(NT*32)+out_quad*32+qlane*8)=q;
                if(qlane==0)output_scales[static_cast<int64_t>(row)*NT+out_quad]=bs.byte;
            }
            __syncthreads();
        }
    }
#endif
}

// Row-predicated output store without a branch. A row past row_count gets an
// offset beyond the buffer's num_records (0x3fffffff bf16 is 0x7ffffffe bytes),
// so the buffer store's range check drops it.
//
// Keep these stores branch-free. The ROCm 7.2 compiler (LLVM 22) does not mark
// the gfx950 scaled MFMAs (v_mfma_scale_f32_*) convergent, so machine-sink moves
// an MFMA whose result is only used under a lane-dependent if into that branch,
// together with its operand setup. The setup then runs for the active lanes
// only, but the MFMA reads all 64 and the output turns NaN. Fixed upstream in
// llvm/llvm-project#178627.
__device__ __forceinline__ int glm53_row_offset(bool valid,int offset)
{
    return valid?offset:0x3fffffff;
}

// Codebook bytes are sign-free (bit 7 clear), so OR applies the sign planes
// and folds into a single v_and_or_b32.
__device__ __forceinline__ uint64_t glm53_apply_signs(uint64_t magnitude,uint32_t signs)
{
    constexpr uint32_t spread=0x10204080u,mask=0x80808080u;
    const uint32_t low=static_cast<uint32_t>(magnitude)|(((signs&15u)*spread)&mask);
    const uint32_t high=static_cast<uint32_t>(magnitude>>32)|((((signs>>4)&15u)*spread)&mask);
    return static_cast<uint64_t>(high)<<32|low;
}

// Weight bases of a down triplet. IQ2R: the three N blocks' base bytes.
// IQ2R-v2: the lane's d word of every K256 column (low dword column 0,
// high dword column 1), see iq2r_kt_bases.
template<int Classes,int KT>
__device__ __forceinline__ uint64_t iq2r_triplet_bases(const uint8_t* aux,int n_block,int lane_row)
{
    if constexpr(Classes==0)
    {
        uint32_t bases=0;
#pragma unroll
        for(int atom=0;atom<3;++atom)bases|=static_cast<uint32_t>(aux[kCodebookBytes+n_block+atom])<<(atom*8);
        return bases;
    }
    else
    {
        constexpr int columns=KT/2;
        uint64_t bases=iq2r_v2_bases<Classes,1>(aux,n_block/3,columns,0,lane_row);
        if constexpr(columns>1)
            bases|=static_cast<uint64_t>(iq2r_v2_bases<Classes,1>(aux,n_block/3,columns,1,lane_row))<<32;
        return bases;
    }
}

template<int Classes>
__device__ __forceinline__ uint32_t iq2r_kt_bases(uint64_t bases,int kt)
{
    return Classes==0?static_cast<uint32_t>(bases):static_cast<uint32_t>(bases>>((kt>>1)*32));
}

// IQ2R-v2 decode in two halves, so a caller can read the next K tile's
// codewords from LDS before it applies this tile's signs.
template<int Atom,int Classes>
__device__ __forceinline__ void glm53_gather_atom(const IQ2RCompressedTriplet& compressed,const uint64_t* codebook,
                                                  const IQ2RLevels& levels,uint32_t book_high,uint64_t (&raw)[4])
{
    static_assert(GLM53_X_RB);
    const uint32_t index_lows=Atom==0?compressed.paired.x:Atom==1?compressed.paired.z:compressed.third.x;
    const uint32_t index_highs=(compressed.third.z>>(Atom*8))&0x0fu;
#pragma unroll
    for(int codeword=0;codeword<4;++codeword)
    {
        const int codebook_index=codeword<3?((index_lows>>(codeword*9))&511u):((index_lows>>27)|(index_highs<<5));
        const uint32_t book_byte=iq2r_copy_offset<Classes>(codeword<2?levels.low:levels.high,Atom,book_high)
                                 |(static_cast<uint32_t>(codebook_index)<<3);
        raw[codeword]=*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(codebook)+book_byte);
    }
}

template<int Atom>
__device__ __forceinline__ opus::i32x8_t glm53_sign_atom(const IQ2RCompressedTriplet& compressed,const uint64_t (&raw)[4])
{
    const uint32_t signs=Atom==0?compressed.paired.y:Atom==1?compressed.paired.w:compressed.third.y;
    union { opus::i32x8_t words; uint64_t codewords[4]; } decoded;
#pragma unroll
    for(int codeword=0;codeword<4;++codeword)
        decoded.codewords[codeword]=glm53_apply_signs(raw[codeword],(signs>>(codeword*8))&255u);
    return decoded.words;
}

template<int Atom,int Classes=0>
__device__ __forceinline__ void glm53_decode_packed_atom(
    const IQ2RCompressedTriplet& compressed,
    const uint64_t* codebook,
    uint32_t packed_bases,
    opus::i32x8_t& weight_fragment,
    uint32_t& scale_b,
    uint32_t book_bits=0u,
    uint32_t book_high=0u,
    const IQ2RLevels* given_levels=nullptr,
    const uint32_t* given_scales=nullptr)
{
    static_assert(Atom >= 0 && Atom < kAtomsPerTriplet);
    const uint32_t index_lows = Atom == 0 ? compressed.paired.x
                                : Atom == 1 ? compressed.paired.z
                                            : compressed.third.x;
    const uint32_t signs = Atom == 0 ? compressed.paired.y
                           : Atom == 1 ? compressed.paired.w
                                       : compressed.third.y;
    const uint32_t metadata = compressed.third.z;
    const uint32_t index_highs = (metadata >> (Atom * 8)) & 0x0fu;
    const IQ2RLevels levels = given_levels ? *given_levels : iq2r_levels<Classes>(metadata,book_bits);
    union
    {
        opus::i32x8_t words;
        uint64_t codewords[4];
    } decoded;
#if GLM53_X_SGN==1
    const uint32_t nibbles_low=signs&0x0f0f0f0fu,nibbles_high=(signs>>4)&0x0f0f0f0fu;
#endif
#pragma unroll
    for(int codeword = 0; codeword < 4; ++codeword)
    {
        const int codebook_index =
            codeword<3?((index_lows>>(codeword*9))&511u):((index_lows>>27)|(index_highs<<5));
#if GLM53_X_RB
        // copy offset (bits 12+) | index * 8 (bits 3-11): disjoint, one v_lshl_or / v_and_or
        const uint32_t book_byte=iq2r_copy_offset<Classes>(codeword < 2 ? levels.low : levels.high, Atom, book_high)
                                 |(static_cast<uint32_t>(codebook_index)<<3);
        const uint64_t* book=reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(codebook)+book_byte);
        constexpr int book_index=0;
#else
        const uint64_t* book =
            codebook + iq2r_copy_offset<Classes>(codeword < 2 ? levels.low : levels.high, Atom, book_high) / 8;
        const int book_index=codebook_index;
#endif
#if GLM53_X_SGN==1
        // n * 0x204081 puts bit j of a nibble n at bit 8 j (the four shifted
        // copies do not overlap): a full-rate 24-bit multiply.
        const uint32_t low=__umul24((nibbles_low>>(codeword*8))&255u,0x204081u);
        const uint32_t high=__umul24((nibbles_high>>(codeword*8))&255u,0x204081u);
        decoded.codewords[codeword]=book[book_index]
            |(static_cast<uint64_t>((high<<7)&0x80808080u)<<32)|((low<<7)&0x80808080u);
#elif GLM53_X_SGN==2
        // The nibble moved to bits 7-10 times 0x204081 lands bit j at 8 j + 7.
        // Both operands fit 24 bits: a full-rate v_mul_u32_u24 (asm, since
        // LLVM folds a later shift back into a quarter-rate v_mul_lo_u32).
        const auto spread=[](uint32_t n){uint32_t r;asm("v_mul_u32_u24 %0, 0x204081, %1":"=v"(r):"v"(n));return r;};
        const uint32_t low=codeword==0?(signs<<7)&0x780u:(signs>>(codeword*8-7))&0x780u;
        const uint32_t high=codeword==0?(signs<<3)&0x780u:(signs>>(codeword*8-3))&0x780u;
        decoded.codewords[codeword]=book[book_index]
            |(static_cast<uint64_t>(spread(high)&0x80808080u)<<32)|(spread(low)&0x80808080u);
#else
        if constexpr(Classes>0 && GLM53_X_PRB==1) decoded.codewords[codeword]=book[book_index];  // timing probe only
        else if constexpr(Classes>0 && GLM53_X_PRB==2)  // timing probe only
            decoded.codewords[codeword]=glm53_apply_signs(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(book+book_index))*0x0101010101010101ull,(signs>>(codeword*8))&255u);
        else
        decoded.codewords[codeword] =
            glm53_apply_signs(book[book_index],(signs>>(codeword*8))&255u);
#endif
    }
    weight_fragment = decoded.words;
    scale_b = Classes>0 && given_scales ? (*given_scales>>(Atom*8))&255u : iq2r_scale_b<Classes>(packed_bases, metadata, Atom);
    if constexpr(Classes>0 && GLM53_X_DND)
    {
        weight_fragment=opus::i32x8_t{static_cast<int>(compressed.paired.x),static_cast<int>(compressed.paired.y),static_cast<int>(compressed.paired.z),static_cast<int>(compressed.paired.w),
                                      static_cast<int>(compressed.third.x),static_cast<int>(compressed.third.y),static_cast<int>(compressed.third.z^Atom),static_cast<int>(packed_bases)};
        scale_b=packed_bases>>(8*Atom);
    }
}

// TP4 (K=512) down: each wave covers KT K128 tiles. Each K tile keeps its own
// zero-initialized accumulator and the partials add in order ((k0+k1)+k2)+k3.
template<int Atom,int KT,int MAtoms,int Classes=0>
__device__ __forceinline__ void glm53_down_tp4_atom(
    const IQ2RCompressedTriplet (&compressed)[KT],const uint64_t* codebook,uint64_t bases,
    const uint8_t* input,const uint8_t* scales,int active_m,int lane_row,int lane_group,
    opus::vector_t<float,4> (&sum)[MAtoms])
{
    constexpr int K=KT*128;
    // Issue every K tile's MFMA before the first addition so no add waits on
    // the MFMA it follows; the ((k0+k1)+k2)+k3 order is unchanged.
    opus::vector_t<float,4> part[KT][MAtoms];
    if constexpr(Classes>0 && GLM53_X_S8IL)
    {
#pragma unroll
        for(int kt=0;kt<KT;++kt)
        {
            opus::i32x8_t w;uint32_t sb;
            glm53_decode_packed_atom<Atom,Classes>(compressed[kt],codebook,iq2r_kt_bases<Classes>(bases,kt),w,sb);
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if((MAtoms>1 || !GLM53_X_S8NB) && m>=active_m)continue;
                const int r=m*16+lane_row,col=kt*128+lane_group*16;
                IQ2RActivationFragment a;
                *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(input+((r*K+col)^((r&15)<<4)));
                *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(input+((r*K+col+64)^((r&15)<<4)));
                const uint32_t sa=scales[r*(K/32)+kt*4+lane_group];
                part[kt][m]=opus::vector_t<float,4>{};
                auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                part[kt][m]=mma(w,a.words,part[kt][m],sb,sa);
            }
        }
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if((MAtoms>1 || !GLM53_X_S8NB) && m>=active_m)continue;
            sum[m]=part[0][m];
#pragma unroll
            for(int kt=1;kt<KT;++kt)
#pragma unroll
                for(int c=0;c<4;++c)sum[m][c]+=part[kt][m][c];
        }
        return;
    }
    opus::i32x8_t w[KT];uint32_t sb[KT];
#pragma unroll
    for(int kt=0;kt<KT;++kt)
    {
        if constexpr(Classes>0 && GLM53_X_S8ND)
        {
            const auto& c=compressed[kt];const uint32_t kb=iq2r_kt_bases<Classes>(bases,kt);
            w[kt]=opus::i32x8_t{static_cast<int>(c.paired.x),static_cast<int>(c.paired.y),static_cast<int>(c.paired.z),static_cast<int>(c.paired.w),
                                static_cast<int>(c.third.x),static_cast<int>(c.third.y),static_cast<int>(c.third.z^Atom),static_cast<int>(kb)};
            sb[kt]=kb>>(8*Atom);
        }
        else glm53_decode_packed_atom<Atom,Classes>(compressed[kt],codebook,iq2r_kt_bases<Classes>(bases,kt),w[kt],sb[kt]);
    }
#pragma unroll
    for(int kt=0;kt<KT;++kt)
    {
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
            const int r=m*16+lane_row,col=kt*128+lane_group*16;
            IQ2RActivationFragment a;
            *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(input+((r*K+col)^((r&15)<<4)));
            *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(input+((r*K+col+64)^((r&15)<<4)));
            const uint32_t sa=scales[r*(K/32)+kt*4+lane_group];
            part[kt][m]=opus::vector_t<float,4>{};
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            part[kt][m]=mma(w[kt],a.words,part[kt][m],sb[kt],sa);
        }
    }
#pragma unroll
    for(int m=0;m<MAtoms;++m)
    {
        if(m>=active_m)continue;
        sum[m]=part[0][m];
#pragma unroll
        for(int kt=1;kt<KT;++kt)
#pragma unroll
            for(int c=0;c<4;++c)sum[m][c]+=part[kt][m][c];
    }
}

// Packed triplet source offset; with n_block%3==0 it is affine in the K tile,
// equal to triplet_base(physical_tile(n_block,k_tile,k_tiles)).
template<int KT>
__device__ __forceinline__ int glm53_ws_source_base(int n_block,int k_tile)
{
    const uint32_t n=static_cast<uint32_t>(n_block);
    return static_cast<int>(((n/kNBlocksPerGroup)*KT+static_cast<uint32_t>(k_tile))*kGroupBytes+
                            ((n%kNBlocksPerGroup)/kAtomsPerTriplet)*kTripletBytes);
}

// IQ2R-v2 TP4 down, K tiles outermost: the three atoms of a triplet share
// each K tile's activation fragment, metadata swap and scale bytes. Each
// atom's sum still accumulates ((k0+k1)+k2)+k3 from zero-initialized MFMAs.
template<int KT,int MAtoms,int Classes>
__device__ __forceinline__ void glm53_down_tp4_triplet(
    const IQ2RCompressedTriplet (&compressed)[KT],const uint64_t* codebook,uint64_t bases,
    const uint8_t* input,const uint8_t* scales,int active_m,int lane_row,int lane_group,
    opus::vector_t<float,4> (&sum)[3][MAtoms])
{
    constexpr int K=KT*128;
#pragma unroll
    for(int kt=0;kt<KT;++kt)
    {
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
            const int r=m*16+lane_row,col=kt*128+lane_group*16;
            IQ2RActivationFragment a;
            *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(input+((r*K+col)^((r&15)<<4)));
            *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(input+((r*K+col+64)^((r&15)<<4)));
            const uint32_t sa=scales[r*(K/32)+kt*4+lane_group];
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
#define GLM53_TP4_TRIPLET_ATOM(ATOM) \
            { \
                opus::i32x8_t w;uint32_t sb; \
                glm53_decode_packed_atom<ATOM,Classes>(compressed[kt],codebook,iq2r_kt_bases<Classes>(bases,kt),w,sb); \
                const opus::vector_t<float,4> part=mma(w,a.words,opus::vector_t<float,4>{},sb,sa); \
                if(kt==0)sum[ATOM][m]=part; \
                else \
                    _Pragma("unroll") \
                    for(int c=0;c<4;++c)sum[ATOM][m][c]+=part[c]; \
            }
            GLM53_TP4_TRIPLET_ATOM(0) GLM53_TP4_TRIPLET_ATOM(1) GLM53_TP4_TRIPLET_ATOM(2)
#undef GLM53_TP4_TRIPLET_ATOM
        }
    }
}

template<int Waves,int MAtoms,int Groups,int XCD,int TaskGroup,int KT,int Occupancy=1,int Classes=0>
__global__ __launch_bounds__(64*Waves,Occupancy) void glm53_down_tp4_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    __hip_bfloat16* __restrict__ output,
    int M,int expert_count,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int K=KT*128,Rows=16*MAtoms,Columns=48*Waves*Groups,Threads=64*Waves;
    struct Storage {
        alignas(16) uint64_t codebook[iq2r_codebook_words<Classes,1>()];
        alignas(16) uint8_t input[Rows*K];
        uint8_t scales[Rows*(K/32)];
    };
    __shared__ Storage shared;
    // A wave-uniform wave index keeps the single-atom variant within 168 VGPRs
    // (three workgroups per CU) without spilling.
    const int lane=threadIdx.x,wave=__builtin_amdgcn_readfirstlane(threadIdx.y),linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    opus::gmem<uint8_t> a_buffer(activations,static_cast<unsigned int>(M*K));
    opus::gmem<opus::bf16_t> out_buffer(output,static_cast<unsigned int>(M*6144*sizeof(__hip_bfloat16)));
    opus::gmem<uint8_t> scale_buffer(activation_scales,static_cast<unsigned int>(M*(K/32)));
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(6144/Columns);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(6144/Columns);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(6144/Columns)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(6144/Columns));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile;
        const int begin=tasks[task*3],count=tasks[task*3+1],expert=tasks[task*3+2];
        if(begin<0 || count<=0 || begin+count>M || expert<0 || expert>=expert_count)continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(previous_expert!=expert)
        {
            if constexpr(Classes==0) { for(int i=linear;i<512;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i]; }
            else iq2r_stage_codebook_async<(GLM53_X_S8W>0?GLM53_X_S8W*512:iq2r_codebook_words<Classes,1>()),Waves>(shared.codebook,aux,wave,lane);
            previous_expert=expert;
        }
        for(int row_base=begin;row_base<begin+count;row_base+=Rows)
        {
            const int row_count=min(Rows,begin+count-row_base);
            const int active_m=(row_count+15)/16;
#pragma unroll
            for(int copy=0;copy<(Rows*K+Threads*16-1)/(Threads*16);++copy)
            {
                const int offset=(linear+copy*Threads)*16;
                const int r=offset/K,col=(offset%K)^((r&15)<<4);
                const int src=(row_base+min(r,row_count-1))*K+col;
                const int dest=(wave*64+copy*Threads)*16;
                if(dest<Rows*K)a_buffer.template async_load<16>(shared.input+dest,src);
            }
            if(linear<Rows*(K/128))
            {
                const int r=linear/(K/128),col=(linear%(K/128))*4;
                scale_buffer.template async_load<4>(shared.scales+wave*64*4,(row_base+min(r,row_count-1))*(K/32)+col);
            }
            IQ2RCompressedTriplet pending[KT];
            uint64_t pending_bases=0;
            {
                const int nb=n_tile*(Columns/16)+wave*3;
                pending_bases=iq2r_triplet_bases<Classes,KT>(aux,nb,lane_row);
#pragma unroll
                for(int kt=0;kt<KT;++kt)pending[kt]=iq2r_load_compact_triplet(data,Classes>0 && GLM53_X_A32?glm53_ws_source_base<KT>(nb,kt):static_cast<int>(triplet_base(physical_tile(nb,kt,KT))),lane);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
#pragma unroll 1
            for(int group=0;group<Groups;++group)
            {
                const int n_block=n_tile*(Columns/16)+wave*3+group*Waves*3;
                const uint64_t bases=pending_bases;
                IQ2RCompressedTriplet compressed[KT];
#pragma unroll
                for(int kt=0;kt<KT;++kt)compressed[kt]=pending[kt];
                if(group+1<Groups)
                {
                    const int nb=n_block+Waves*3;
                    pending_bases=iq2r_triplet_bases<Classes,KT>(aux,nb,lane_row);
#pragma unroll
                    for(int kt=0;kt<KT;++kt)pending[kt]=iq2r_load_compact_triplet(data,Classes>0 && GLM53_X_A32?glm53_ws_source_base<KT>(nb,kt):static_cast<int>(triplet_base(physical_tile(nb,kt,KT))),lane);
                    asm volatile("" ::: "memory");
                }
#define GLM53_DOWN_TP4_ATOM(ATOM) \
                { \
                    opus::vector_t<float,4> sum[MAtoms]; \
                    glm53_down_tp4_atom<ATOM,KT,MAtoms,Classes>(compressed,shared.codebook,bases,shared.input,shared.scales,active_m,lane_row,lane_group,sum); \
                    _Pragma("unroll") \
                    for(int m=0;m<MAtoms;++m) \
                    { \
                        if(m>=active_m)continue; \
                        const int out_r=m*16+lane_row; \
                        const int out_col=n_block*16+ATOM*16+lane_group*4; \
                        opus::vector_t<opus::bf16_t,4> packed; \
                        _Pragma("unroll") \
                        for(int c=0;c<4;++c)packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(sum[m][c]*iq2r_gain<Classes>(aux,auxiliary_bytes,0))); \
                        out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,(row_base+out_r)*6144+out_col)); \
                    } \
                }
                if constexpr(Classes>0 && GLM53_X_S8KO)
                {
                    opus::vector_t<float,4> sums[3][MAtoms];
                    glm53_down_tp4_triplet<KT,MAtoms,Classes>(compressed,shared.codebook,bases,shared.input,shared.scales,active_m,lane_row,lane_group,sums);
#pragma unroll
                    for(int atom=0;atom<3;++atom)
#pragma unroll
                    for(int m=0;m<MAtoms;++m)
                    {
                        if(m>=active_m)continue;
                        const int out_r=m*16+lane_row;
                        const int out_col=n_block*16+atom*16+lane_group*4;
                        opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                        for(int c=0;c<4;++c)packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(sums[atom][m][c]*iq2r_gain<Classes>(aux,auxiliary_bytes,0)));
                        out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,(row_base+out_r)*6144+out_col));
                    }
                }
                else { GLM53_DOWN_TP4_ATOM(0) GLM53_DOWN_TP4_ATOM(1) GLM53_DOWN_TP4_ATOM(2) }
#undef GLM53_DOWN_TP4_ATOM
            }
            __syncthreads();
        }
    }
#endif
}


// TP4 (K=512) weight-stationary down: one 64-row task tile stays in LDS while
// each wave decodes a K128 weight triplet once and applies it to every row
// atom. K tiles are outermost, so each output still accumulates
// ((k0+k1)+k2)+k3 from zero-initialized per-tile MFMAs, as in glm53_down_tp4_kernel.
template<int Waves,int Groups,int XCD,int TaskGroup,int KT,int Chunks,int Occupancy,int Classes=0>
__global__ __launch_bounds__(64*Waves,Occupancy) void glm53_down_prefill_tp4_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    __hip_bfloat16* __restrict__ output,
    int M,int expert_count,int data_bytes,int auxiliary_bytes,int chunk)
{
#if defined(__gfx950__)
    constexpr int MAtoms=4,K=KT*128,Rows=16*MAtoms,Columns=48*Waves*Groups,Threads=64*Waves;
    struct Storage {
        alignas(16) uint64_t codebook[iq2r_codebook_words<Classes,1>()];
        alignas(16) uint8_t input[Rows*K];
        uint8_t scales[Rows*(K/32)];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    opus::gmem<uint8_t> a_buffer(activations,static_cast<unsigned int>(M*K));
    opus::gmem<opus::bf16_t> out_buffer(output,static_cast<unsigned int>(M*6144*sizeof(__hip_bfloat16)));
    opus::gmem<uint8_t> scale_buffer(activation_scales,static_cast<unsigned int>(M*(K/32)));
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(6144/Columns/Chunks);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(6144/Columns/Chunks);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(6144/Columns/Chunks)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(6144/Columns/Chunks));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile+chunk*(6144/Columns/Chunks);
        const int begin=tasks[task*3],count=tasks[task*3+1],expert=tasks[task*3+2];
        if(begin<0 || count<=0 || begin+count>M || expert<0 || expert>=expert_count)continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(previous_expert!=expert)
        {
            if constexpr(Classes==0) { for(int i=linear;i<512;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i]; }
            else iq2r_stage_codebook_async<iq2r_codebook_words<Classes,1>(),Waves>(shared.codebook,aux,wave,lane);
            previous_expert=expert;
        }
        for(int row_base=begin;row_base<begin+count;row_base+=Rows)
        {
            const int row_count=min(Rows,begin+count-row_base);
            const int active_m=(row_count+15)/16;
#pragma unroll
            for(int copy=0;copy<(Rows*K+Threads*16-1)/(Threads*16);++copy)
            {
                const int offset=(linear+copy*Threads)*16;
                const int r=offset/K,col=(offset%K)^((r&15)<<4);
                const int src=(row_base+min(r,row_count-1))*K+col;
                const int dest=(wave*64+copy*Threads)*16;
                if(dest<Rows*K)a_buffer.template async_load<16>(shared.input+dest,src);
            }
#pragma unroll
            for(int part=0;part<(Rows*(K/128)+Threads-1)/Threads;++part)
            if(linear+part*Threads<Rows*(K/128))
            {
                const int r=(linear+part*Threads)/(K/128),col=((linear+part*Threads)%(K/128))*4;
                scale_buffer.template async_load<4>(shared.scales+(wave*64+part*Threads)*4,(row_base+min(r,row_count-1))*(K/32)+col);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
#pragma unroll 1
            for(int group=0;group<Groups;++group)
            {
                const int n_block=n_tile*(Columns/16)+wave*3+group*Waves*3;
                const uint64_t bases=iq2r_triplet_bases<Classes,KT>(aux,n_block,lane_row);
                opus::vector_t<float,4> sum[MAtoms][3];
                // Rolled K loop: one compressed K tile live plus a one-ahead
                // load, keeping VGPRs low enough for more resident waves.
                IQ2RCompressedTriplet current=iq2r_load_compact_triplet(data,glm53_ws_source_base<KT>(n_block,0),lane);
#pragma unroll 1
                for(int kt=0;kt<KT;++kt)
                {
                    IQ2RCompressedTriplet next=current;
                    if(kt+1<KT)next=iq2r_load_compact_triplet(data,glm53_ws_source_base<KT>(n_block,kt+1),lane);
                    opus::i32x8_t w[3];uint32_t sb[3];
                    glm53_decode_packed_atom<0,Classes>(current,shared.codebook,iq2r_kt_bases<Classes>(bases,kt),w[0],sb[0]);
                    glm53_decode_packed_atom<1,Classes>(current,shared.codebook,iq2r_kt_bases<Classes>(bases,kt),w[1],sb[1]);
                    glm53_decode_packed_atom<2,Classes>(current,shared.codebook,iq2r_kt_bases<Classes>(bases,kt),w[2],sb[2]);
#pragma unroll
                    for(int m=0;m<MAtoms;++m)
                    {
                        if(m>=active_m)continue;
                        const int r=m*16+lane_row,col=kt*128+lane_group*16;
                        IQ2RActivationFragment a;
                        *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*K+col)^((r&15)<<4)));
                        *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*K+col+64)^((r&15)<<4)));
                        const uint32_t sa=shared.scales[r*(K/32)+kt*4+lane_group];
#pragma unroll
                        for(int atom=0;atom<3;++atom)
                        {
                            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                            opus::vector_t<float,4> part{};
                            part=mma(w[atom],a.words,part,sb[atom],sa);
                            if(kt==0)sum[m][atom]=part;
                            else
#pragma unroll
                                for(int c=0;c<4;++c)sum[m][atom][c]+=part[c];
                        }
                    }
                    current=next;
                }
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int out_r=m*16+lane_row;
#pragma unroll
                    for(int atom=0;atom<3;++atom)
                    {
                        const int out_col=n_block*16+atom*16+lane_group*4;
                        opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                        for(int c=0;c<4;++c)packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(sum[m][atom][c]*iq2r_gain<Classes>(aux,auxiliary_bytes,0)));
                        out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,((out_col/384)*M+row_base+out_r)*384+out_col%384));
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

// TP8 (K=256) down: waves cover independent N tiles and share one A tile
// cached per CTA. Each output is the sum of two zero-initialized K128 MFMAs.
// Chunks splits N across launches; Tiled stores [N/Columns][M][Columns] for
// glm53_reduce_chunk_kernel, otherwise rows are [M][6144].
template<int Waves,int MAtoms,int Groups,int XCD,int TaskGroup,int Chunks,bool Tiled,int Classes=0>
__global__ __launch_bounds__(64*Waves,1) void glm53_down_prefill_tp8_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    __hip_bfloat16* __restrict__ output,
    int M,int expert_count,int data_bytes,int auxiliary_bytes,int chunk)
{
#if defined(__gfx950__)
    constexpr int Rows=16*MAtoms,Columns=48*Waves*Groups,Threads=64*Waves;
    struct Storage {
        alignas(16) uint64_t codebook[iq2r_codebook_words<Classes,1>()];
        alignas(16) uint8_t input[Rows*256];
        uint8_t scales[Rows*8];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    opus::gmem<uint8_t> a_buffer(activations,static_cast<unsigned int>(M*256));
    // Descriptor bounds are bytes; typed offsets below are BF16 elements.
    opus::gmem<opus::bf16_t> out_buffer(output,static_cast<unsigned int>(M*6144*sizeof(__hip_bfloat16)));
    opus::gmem<uint8_t> scale_buffer(activation_scales,static_cast<unsigned int>(M*8));
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(6144/Columns/Chunks);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(6144/Columns/Chunks);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(6144/Columns/Chunks)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(6144/Columns/Chunks));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile+chunk*(6144/Columns/Chunks);
        const int begin=tasks[task*3],count=tasks[task*3+1],expert=tasks[task*3+2];
        if(begin<0 || count<=0 || begin+count>M || expert<0 || expert>=expert_count)continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(previous_expert!=expert)
        {
            if constexpr(Classes==0) { for(int i=linear;i<512;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i]; }
            else iq2r_stage_codebook_async<iq2r_codebook_words<Classes,1>(),Waves>(shared.codebook,aux,wave,lane);
            previous_expert=expert;
        }
        for(int row_base=begin;row_base<begin+count;row_base+=Rows)
        {
            const int row_count=min(Rows,begin+count-row_base);
            const int active_m=(row_count+15)/16;
#pragma unroll
            for(int copy=0;copy<(Rows*256+Threads*16-1)/(Threads*16);++copy)
            {
                const int offset=(linear+copy*Threads)*16;
                const int r=offset/256,col=(offset%256)^((r&15)<<4);
                const int src=(row_base+min(r,row_count-1))*256+col;
                const int dest=(wave*64+copy*Threads)*16;
                if(dest<Rows*256)a_buffer.template async_load<16>(shared.input+dest,src);
            }
            if(linear<Rows*2)
            {
                const int r=linear/2,col=(linear%2)*4;
                scale_buffer.template async_load<4>(shared.scales+wave*64*4,(row_base+min(r,row_count-1))*8+col);
            }
            IQ2RCompressedTriplet pending[2];
            uint64_t pending_bases=0;
            {
                const int nb=n_tile*(Columns/16)+wave*3;
                pending_bases=iq2r_triplet_bases<Classes,2>(aux,nb,lane_row);
#pragma unroll
                for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(nb,kt,2))),lane);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();

#pragma unroll 1
            for(int group=0;group<Groups;++group)
            {
        const int n_block=n_tile*(Columns/16)+wave*3+group*Waves*3;
        const uint64_t bases=pending_bases;
        IQ2RCompressedTriplet compressed[2];
        compressed[0]=pending[0];compressed[1]=pending[1];
        if(group+1<Groups)
        {
            const int nb=n_block+Waves*3;
            pending_bases=iq2r_triplet_bases<Classes,2>(aux,nb,lane_row);
#pragma unroll
            for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,static_cast<int>(triplet_base(physical_tile(nb,kt,2))),lane);
            asm volatile("" ::: "memory");
        }
            // Finish one N16 atom at a time. Retain the exact two K128
            // partial sums, while shortening the lifetime of decoded weights
            // and accumulators enough to support a full M64 tile.
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<0,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<0,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,second,sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+0*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16((first[m][c]+second[c])*iq2r_gain<Classes>(aux,auxiliary_bytes,0)));
                    }
                    out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,(Tiled?(n_tile*M+row_base+out_r)*Columns+out_col%Columns:(row_base+out_r)*6144+out_col)));
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<1,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<1,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,second,sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+1*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16((first[m][c]+second[c])*iq2r_gain<Classes>(aux,auxiliary_bytes,0)));
                    }
                    out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,(Tiled?(n_tile*M+row_base+out_r)*Columns+out_col%Columns:(row_base+out_r)*6144+out_col)));
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<2,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<2,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,second,sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+2*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)
                    {
                        packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16((first[m][c]+second[c])*iq2r_gain<Classes>(aux,auxiliary_bytes,0)));
                    }
                    out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,(Tiled?(n_tile*M+row_base+out_r)*Columns+out_col%Columns:(row_base+out_r)*6144+out_col)));
                }
            }
            }
            __syncthreads();
        }
    }
#endif
}

// Weighted top-9 sum for one N chunk of tiled prefill route outputs: nine FP32
// FMAs in route order, then one BF16 rounding.
template<int Chunks>
__global__ __launch_bounds__(128) void glm53_reduce_chunk_kernel(
    const __hip_bfloat16* __restrict__ route_output,
    const float* __restrict__ route_weights,const int32_t* __restrict__ scatter,
    __hip_bfloat16* __restrict__ output,int tokens,int chunk)
{
#if defined(__gfx950__)
    constexpr int Width=6144/Chunks,Tiles=(Width+1023)/1024;
    const int token=blockIdx.x/Tiles,tile=blockIdx.x%Tiles;
    const int local=tile*1024+threadIdx.x*8;
    if(token>=tokens || local>=Width)return;
    const int col=chunk*Width+local;
    float sums[8]={};
#pragma unroll
    for(int route=0;route<9;++route)
    {
        const int sorted=__builtin_amdgcn_readfirstlane(scatter[token*9+route]);
        if(sorted<0)continue;  // invalid expert id: the route contributes nothing
        const float weight=__builtin_bit_cast(float,__builtin_amdgcn_readfirstlane(__builtin_bit_cast(uint32_t,route_weights[token*9+route])));
        const int64_t address=(static_cast<int64_t>(col/384)*tokens*9+sorted)*384+col%384;
        opus::vector_t<opus::bf16_t,8> values;
        *reinterpret_cast<uint4*>(&values)=*reinterpret_cast<const uint4*>(route_output+address);
#pragma unroll
        for(int i=0;i<8;++i)sums[i]=fmaf(static_cast<float>(values[i]),weight,sums[i]);
    }
    opus::vector_t<opus::bf16_t,8> packed;
#pragma unroll
    for(int i=0;i<8;++i)packed[i]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(sums[i]));
    *reinterpret_cast<uint4*>(output+static_cast<int64_t>(token)*6144+col)=*reinterpret_cast<const uint4*>(&packed);
#endif
}

// Helper wave for glm53_down_route9_kernel (waves 9-11): decodes the last
// HelpTiles K tiles of the last triplet of route 4*(wave-9) once that route's
// wave has staged its codebook copies, and leaves the per-K-tile sums in LDS.
template<int KT,int Classes,int Triplets,int HelpTiles,typename Storage>
__device__ __forceinline__ void glm53_down_route9_helper(
    Storage& shared,const opus::fp8_t* __restrict__ activations,const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ expert_ids,const int32_t* __restrict__ scatter,int data_bytes,int auxiliary_bytes)
{
    constexpr int Atoms=3,K=KT*128;
    const int lane=threadIdx.x,h=__builtin_amdgcn_readfirstlane(threadIdx.y)-9,route=h*4;
    const int lane_row=lane%16,lane_group=lane/16;
    const int token=blockIdx.y,first=(static_cast<int>(blockIdx.x)*Triplets+Triplets-1)*Atoms;
    const int original=token*9+route;
    const int expert=__builtin_amdgcn_readfirstlane(expert_ids[original]);
    const int row=__builtin_amdgcn_readfirstlane(scatter[original]);
    const bool valid=expert>=0 && expert<257 && row>=0 && row<static_cast<int>(gridDim.y)*9;
    __hip_atomic_store(&shared.help_flag[h],0u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_WORKGROUP);
    const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
    const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
    const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(row)*K);
    IQ2RCompressedTriplet compressed[HelpTiles];
    IQ2RActivationFragment a[HelpTiles];
    uint32_t sa[HelpTiles];
    uint64_t bases=0;
    if(valid)
    {
#pragma unroll
        for(int j=0;j<HelpTiles;++j)
        {
            const int ktile=KT-HelpTiles+j;
            const uint32_t tr=blockIdx.x*Triplets+Triplets-1;
            compressed[j]=iq2r_scheduled_load_compact_uniform(data,data_bytes,static_cast<int>(((tr>>1)*KT+ktile)*kGroupBytes+(tr&1)*kTripletBytes),lane);
            const int ak=ktile*128+lane_group*16;
            *reinterpret_cast<uint4*>(a[j].bytes)=*reinterpret_cast<const uint4*>(input+ak);
            *reinterpret_cast<uint4*>(a[j].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
            sa[j]=scales[static_cast<int64_t>(row)*(K/32)+ktile*4+lane_group];
        }
        bases=iq2r_triplet_bases<Classes,KT>(aux,first,lane_row);
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
    if(valid)
    {
        while(__builtin_amdgcn_readfirstlane(__hip_atomic_load(&shared.help_flag[h],__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_WORKGROUP))!=1u)
            __builtin_amdgcn_s_sleep(1);
        asm volatile("" ::: "memory");
        const uint64_t* route_book=shared.codebook[route];
#pragma unroll
        for(int j=0;j<HelpTiles;++j)
        {
            opus::vector_t<float,4> k_sums[Atoms]={};
            opus::i32x8_t b[3];
            uint32_t sb0,sb1,sb2;
            const uint32_t kt_bases=iq2r_kt_bases<Classes>(bases,KT-HelpTiles+j);
            glm53_decode_packed_atom<0,Classes>(compressed[j],route_book,kt_bases,b[0],sb0);
            glm53_decode_packed_atom<1,Classes>(compressed[j],route_book,kt_bases,b[1],sb1);
            glm53_decode_packed_atom<2,Classes>(compressed[j],route_book,kt_bases,b[2],sb2);
            const uint32_t sb=sb0|(sb1<<8)|(sb2<<16);
            uint32_t s=sa[j];
            asm volatile("" : "+v"(s));
            iq2r_triplet_mfma<0>(a[j].words,b,k_sums,s*0x01010101u,sb);
#pragma unroll
            for(int atom=0;atom<Atoms;++atom)
                *(lane_group==0?&shared.help_sums[h][1+j][atom*16+lane_row]:shared.help_spare)=k_sums[atom][0];
        }
    }
    else if(lane_group==0)
    {
#pragma unroll
        for(int j=0;j<HelpTiles;++j)
#pragma unroll
            for(int atom=0;atom<Atoms;++atom)shared.help_sums[h][1+j][atom*16+lane_row]=0.0f;
    }
    __syncthreads();
}

// Down for M<=4: one CTA per (Triplets x 48-column blocks, token) and one
// wave per route. Each wave decodes its expert's triplets with a private
// codebook copy; the routes are then combined with nine ordered FP32 FMAs.
template<int KT,int Classes=0,int Triplets=1>
__global__ __launch_bounds__((GLM53_X_R9IDLE || (Classes>0 && GLM53_X_R9HELP))?768:(Classes>0 && GLM53_X_R9S8)?512:576,1) void glm53_down_route9_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ scatter,
    const float* __restrict__ route_weights,
    __hip_bfloat16* __restrict__ output,
    int data_bytes, int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int Atoms=3,N=6144,K=KT*128,Columns=Triplets*Atoms*16;
    constexpr int BookWords=iq2r_codebook_words<Classes,1>();
    // Helper waves 9-11 decode the last HelpTiles K tiles of the last triplet of
    // routes 0, 4 and 8, whose waves share SIMD0 with each other.
    constexpr bool Help=Classes>0 && GLM53_X_R9HELP;
    constexpr int HelpTiles=Triplets*KT/4>1?Triplets*KT/4:1;
    // IQ2R-v2 on 8 waves: route 8's (triplet, K tile) units go round-robin
    // over the 8 route waves, two per SIMD.
    constexpr bool Split=Classes>0 && GLM53_X_R9S8;
    constexpr int SplitUnits=Triplets*KT,SplitPerWave=(SplitUnits+7)/8;
    struct Storage {
        alignas(16) uint64_t codebook[9][BookWords];
        // Columns results per route, then a spare slot (padded to 16 bytes)
        // that lane groups 1-3 write.
        alignas(16) __hip_bfloat16 routes[9][Columns+8];
        float help_sums[3][HelpTiles+1][48];
        float help_gain[3];
        float help_spare[4];
        uint32_t help_flag[3];
        float split_sums[(Split || (Classes>0 && GLM53_X_R9X))?SplitUnits*Atoms:1][16];
        float split_spare[4];
        // IQ2R-v2: each route's activation row, staged by a global -> LDS DMA.
        alignas(16) uint8_t act[Classes>0 && GLM53_X_R9AL?9:1][K];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x;
    const int route=Classes>0 && GLM53_X_R9SR?__builtin_amdgcn_readfirstlane(threadIdx.y):threadIdx.y;
    if constexpr(Help)
        if(route>=9)
        {
            glm53_down_route9_helper<KT,Classes,Triplets,HelpTiles>(shared,activations,scales,all_data,all_auxiliary,
                                                                    expert_ids,scatter,data_bytes,auxiliary_bytes);
            return;
        }
    // timing probe only: per-wave timestamps past the output rows
    constexpr bool TS=Classes>0 && GLM53_X_TS;
    [[maybe_unused]] uint32_t ts[14]={};
    if constexpr(TS){ts[0]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());ts[8]=static_cast<uint32_t>(__builtin_amdgcn_s_memrealtime());}
    if(GLM53_X_R9IDLE && route>=9){if(GLM53_X_R9EB && Classes>0)__builtin_amdgcn_s_barrier();if(GLM53_X_R9BAR && Classes>0)__syncthreads();__syncthreads();return;}
    const int lane_row=lane%16,lane_group=lane/16;
    const int token=blockIdx.y,first=static_cast<int>(blockIdx.x)*Triplets*Atoms;
    const int original=token*9+route;
    // threadIdx.y is wave-uniform (64-lane x dimension): scalar expert and row
    // keep the codebook DMA free of a waterfall loop.
    const int expert=GLM53_X_R9RFL?__builtin_amdgcn_readfirstlane(expert_ids[original]):expert_ids[original];
    const int row=GLM53_X_R9RFL?__builtin_amdgcn_readfirstlane(scatter[original]):scatter[original];
    // Route 8's ids load with the wave's own, and the checks are branch-free
    // so the grid size load does not wait for them: one scalar wait.
    constexpr bool Load8=Split || (Classes>0 && GLM53_X_R9X);
    [[maybe_unused]] const int expert8=Load8?__builtin_amdgcn_readfirstlane(expert_ids[token*9+8]):0;
    [[maybe_unused]] const int row8=Load8?__builtin_amdgcn_readfirstlane(scatter[token*9+8]):0;
    const int rows=static_cast<int>(gridDim.y)*9;
    // IQ2R-v2: the scalars the epilogue and the loads need are loaded with
    // the prologue's, so no scalar load latency lands after the vector loads.
    constexpr bool Early=Classes>0 && GLM53_X_R9LAT;
    if constexpr(Early) asm volatile("" : "+s"(data_bytes), "+s"(auxiliary_bytes));
    [[maybe_unused]] float early_weights[9];
    if constexpr(Early)
#pragma unroll
        for(int r=0;r<9;++r)early_weights[r]=route_weights[token*9+r];
    const bool valid=Classes>0 && GLM53_X_R9PRO
                     ?((expert>=0)&(expert<257)&(row>=0)&(row<rows))
                     :expert>=0 && expert<257 && row>=0 && row<rows && !(GLM53_X_R9DROP && route==GLM53_X_R9DROP-1);
    const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
    const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
    if constexpr(TS) ts[1]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
    if constexpr(Classes==0)
    {
        if(valid)
            for(int entry=lane;entry<kCodebookBytes/16;entry+=64)
                reinterpret_cast<uint4*>(shared.codebook[route])[entry]=reinterpret_cast<const uint4*>(aux)[entry];
        __syncthreads();
    }
    const uint64_t* route_book=shared.codebook[route];
    // IQ2R-v2: the route's copies sit route * Classes * 4 KB into the
    // codebooks. Its low bits ride in each level byte above the class bits and
    // the rest in byte 2 of the copy offset, so one v_perm gives the whole
    // offset from codebook[0] (see iq2r_copy_offset).
    constexpr int RouteLowBits=4-iq2r_log2_classes<Classes>();
    const bool rb=Classes>0 && GLM53_X_RB;
    const uint64_t* rb_book=rb?shared.codebook[0]:route_book;
    const uint32_t rb_bits=rb?static_cast<uint32_t>(route&((1<<RouteLowBits)-1))*(0x01010101u<<(8-RouteLowBits)):0u;
    const uint32_t rb_high=rb?static_cast<uint32_t>(route>>RouteLowBits):0u;
    static_assert(Classes==0 || BookWords*8==1<<(16-RouteLowBits));
    [[maybe_unused]] bool valid8=false;
    [[maybe_unused]] const uint8_t* aux8=nullptr;
    [[maybe_unused]] float gain8=0.0f;
    [[maybe_unused]] IQ2RCompressedTriplet split_c[SplitPerWave];
    [[maybe_unused]] IQ2RActivationFragment split_a[SplitPerWave];
    [[maybe_unused]] uint32_t split_sa[SplitPerWave];
    [[maybe_unused]] uint64_t split_bases[SplitPerWave];
    if constexpr(Split)
    {
        valid8=GLM53_X_R9S8!=2 && ((expert8>=0)&(expert8<257)&(row8>=0)&(row8<rows));
        aux8=all_auxiliary+static_cast<int64_t>(expert8)*auxiliary_bytes;
        if(valid8)
        {
            const uint8_t* data8=all_data+static_cast<int64_t>(expert8)*data_bytes;
            const auto* input8=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(row8)*K);
            gain8=iq2r_gain<Classes>(aux8,auxiliary_bytes,0);
            if constexpr(GLM53_X_R9S8!=4) iq2r_stage_codebook_async<BookWords,8>(shared.codebook[8],aux8,route,lane);
#pragma unroll
            for(int j=0;j<SplitPerWave;++j)
            {
                const int u=route+8*j;
                if(SplitUnits%8!=0 && u>=SplitUnits)break;
                const int t=u/KT,ktile=u%KT;
                const uint32_t tr=blockIdx.x*Triplets+t;
                split_c[j]=iq2r_scheduled_load_compact_uniform(data8,data_bytes,static_cast<int>(((tr>>1)*KT+ktile)*kGroupBytes+(tr&1)*kTripletBytes),lane);
                const int ak=ktile*128+lane_group*16;
                *reinterpret_cast<uint4*>(split_a[j].bytes)=*reinterpret_cast<const uint4*>(input8+ak);
                *reinterpret_cast<uint4*>(split_a[j].bytes+16)=*reinterpret_cast<const uint4*>(input8+ak+64);
                split_sa[j]=scales[static_cast<int64_t>(row8)*(K/32)+ktile*4+lane_group];
                split_bases[j]=iq2r_triplet_bases<Classes,KT>(aux8,first+t*Atoms,lane_row);
            }
        }
    }
    // IQ2R-v2: route 8's (triplet, K tile) units are decoded by waves 0-3 (one
    // per SIMD) after their own route; wave 8 only stages route 8's copies and
    // activation row and raises a flag. Its data lands last (youngest wave),
    // so this keeps a whole route's decode off the CTA's tail.
    constexpr bool Xs=Classes>0 && GLM53_X_R9X && !Split;
    constexpr int XPer=(SplitUnits+3)/4;
    [[maybe_unused]] IQ2RCompressedTriplet x_c[XPer];
    [[maybe_unused]] uint64_t x_bases[XPer];
    [[maybe_unused]] uint32_t x_words[XPer];
    [[maybe_unused]] const uint8_t* data8=nullptr;
    const auto xload=[&]()
    {
        gain8=iq2r_gain<Classes>(aux8,auxiliary_bytes,0);
#pragma unroll
        for(int j=0;j<XPer;++j)
        {
            const int u=route+4*j;
            if(SplitUnits%4!=0 && u>=SplitUnits)break;
            const int t=u/KT,ktile=u%KT;
            const uint32_t tr=blockIdx.x*Triplets+t;
            x_c[j]=iq2r_scheduled_load_compact_uniform(data8,data_bytes,static_cast<int>(((tr>>1)*KT+ktile)*kGroupBytes+(tr&1)*kTripletBytes),lane);
            x_bases[j]=iq2r_triplet_bases<Classes,KT>(aux8,first+t*Atoms,lane_row);
            x_words[j]=reinterpret_cast<const uint32_t*>(scales+static_cast<int64_t>(row8)*(K/32))[ktile];
        }
    };
    [[maybe_unused]] uint4 gen_base[4];
    if constexpr(Classes>0 && GLM53_X_R9GEN>=2)
        if(valid)
        {
#pragma unroll
            for(int i=0;i<4;++i)gen_base[i]=reinterpret_cast<const uint4*>(aux)[i*64+lane];
            asm volatile("" ::: "memory");
        }
    if constexpr(Xs)
    {
        valid8=(expert8>=0)&(expert8<257)&(row8>=0)&(row8<rows);
        aux8=all_auxiliary+static_cast<int64_t>(expert8)*auxiliary_bytes;
        data8=all_data+static_cast<int64_t>(expert8)*data_bytes;
        if(GLM53_X_R9X==1 && route<4 && valid8)xload();
        if(route==8)__hip_atomic_store(&shared.help_flag[0],0u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_WORKGROUP);
        __builtin_amdgcn_s_barrier();  // the flag is clear before any wave reads it
    }
    // GEN==3: routes 4-8 form their copies before issuing their data loads,
    // which stall behind routes 0-3's in the memory pipe anyway.
    constexpr bool GenEarly=Classes>0 && GLM53_X_R9GEN==3;
    if constexpr(GenEarly)
        if(valid && route>=4)
        {
            if constexpr(TS)
            {
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            }
            iq2r_v2_form_copies<Classes>(shared.codebook[route],gen_base,lane);
            asm volatile("" ::: "memory");
            if constexpr(TS)
            {
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                ts[12]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
            }
            if constexpr(TS && GLM53_X_TS==2) iq2r_v2_form_copies<Classes>(shared.codebook[route],gen_base,lane);
            asm volatile("" ::: "memory");
            if constexpr(TS)
            {
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                ts[13]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
            }
        }
    // IQ2R-v2: the waves issue their loads one SIMD at a time (three
    // barriers, every wave passes all three), so the memory pipe serves
    // SIMD 0's three routes first and their decode overlaps the later loads.
    constexpr bool Grouped=Classes>0 && GLM53_X_R9GRP;
    constexpr int IssueGroups=GLM53_X_R9GRP==3?3:4;
    const int issue_group=Grouped?(GLM53_X_R9GRP==3?route/4:GLM53_X_R9GRP==2?3-route%4:route%4):0;
    if constexpr(Grouped)
        for(int i=0;i<issue_group;++i)__builtin_amdgcn_s_barrier();
    if(valid)
    {
        // IQ2R-v2: each wave stages its own route's copies and only reads
        // those, so its vmcnt wait (after every load of the CTA is issued)
        // replaces the barrier.
        if constexpr(Classes>0 && GLM53_X_R9SLP>0)
            for(int i=0;i<route;++i)__builtin_amdgcn_s_sleep(GLM53_X_R9SLP);
        if constexpr(Classes>0)
        {
            if constexpr(GLM53_X_R9ROT)
            {
                // Rotate the chunk order by CTA so concurrent CTAs staging the
                // same expert do not hit the same L2 channel together.
                constexpr int Chunks=(GLM53_X_R9W?512*GLM53_X_R9W:BookWords)*8/1024;
                opus::gmem<uint8_t> buffer(aux,static_cast<unsigned int>(Chunks*1024));
                const int rot=__builtin_amdgcn_readfirstlane(static_cast<int>(blockIdx.x+blockIdx.y*7+route*5));
#pragma unroll
                for(int i=0;i<Chunks;++i)
                {
                    const int chunk=(i+rot)%Chunks;
                    buffer.template async_load<16>(reinterpret_cast<uint8_t*>(shared.codebook[route])+chunk*1024,lane*16,chunk*1024,
                                                   opus::number<0>{},opus::number<0>{});
                }
                asm volatile("" ::: "memory");
            }
            else if constexpr(GLM53_X_R9GEN)
            {
                if constexpr(GLM53_X_R9GEN==1)
                {
#pragma unroll
                for(int i=0;i<4;++i)gen_base[i]=reinterpret_cast<const uint4*>(aux)[i*64+lane];
                asm volatile("" ::: "memory");
                }
            }
            else if constexpr(!GLM53_X_R9VG && !GLM53_X_R9STG && !GLM53_X_R9LATE) iq2r_stage_codebook_async<GLM53_X_R9W?512*GLM53_X_R9W:BookWords,1>(shared.codebook[route],aux,0,lane);
        }
        constexpr int StageChunks=Classes>0 && GLM53_X_R9VG?(GLM53_X_R9W?512*GLM53_X_R9W:BookWords)*8/1024:1;
        uint4 staged[StageChunks];
        if constexpr(Classes>0 && GLM53_X_R9VG)
        {
#pragma unroll
            for(int i=0;i<StageChunks;++i)staged[i]=reinterpret_cast<const uint4*>(aux)[i*64+lane];
        }
        const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(row)*K);
        // IQ2R-v2: the activation row goes to LDS with one DMA instruction
        // instead of two 16-byte loads per K tile in every lane (16 lanes
        // share each address), and the scale row is a scalar load whose
        // byte the lane group picks with one v_perm, replicated.
        constexpr bool ActLds=Classes>0 && GLM53_X_R9AL;
        constexpr bool ScalarScales=Classes>0 && GLM53_X_R9SS;
        if constexpr(ActLds)
        {
            if(lane<K/16)
            {
                opus::gmem<uint8_t> buffer(input,static_cast<unsigned int>(K));
                buffer.template async_load<16>(shared.act[route],lane*16,0,opus::number<0>{},opus::number<0>{});
            }
            asm volatile("" ::: "memory");
        }
        if constexpr(Xs)
            if(route==8)
            {
                if constexpr(GLM53_X_R9GEN && !GenEarly) iq2r_v2_form_copies<Classes>(shared.codebook[route],gen_base,lane);
                if constexpr(TS) ts[2]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                if constexpr(TS) ts[3]=ts[4]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
                __hip_atomic_store(&shared.help_flag[0],1u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_WORKGROUP);
                goto route8_staged;
            }
        [[maybe_unused]] uint32_t scale_words[KT];
        if constexpr(ScalarScales)
        {
            const auto* scale_row=reinterpret_cast<const uint32_t*>(scales+static_cast<int64_t>(row)*(K/32));
#pragma unroll
            for(int ktile=0;ktile<KT;++ktile)scale_words[ktile]=scale_row[ktile];
        }
        [[maybe_unused]] float early_gain=Early?iq2r_gain<Classes>(aux,auxiliary_bytes,0):1.0f;
        IQ2RCompressedTriplet compressed[Triplets][KT];
        IQ2RActivationFragment a[KT];
        uint32_t sa[KT];
        uint64_t bases[Triplets];
#pragma unroll
        for(int ktile=0;ktile<KT;++ktile)
        {
#pragma unroll
            for(int t=0;t<Triplets;++t)
            {
                const uint32_t tr=blockIdx.x*Triplets+t;  // first+t*Atoms == 3*tr
                const int base=Classes>0 && GLM53_X_R9ADDR?static_cast<int>(((tr>>1)*KT+ktile)*kGroupBytes+(tr&1)*kTripletBytes)
                                                          :static_cast<int>(triplet_base(physical_tile(first+t*Atoms,ktile,KT)));
                compressed[t][ktile]=iq2r_scheduled_load_compact_uniform(data,data_bytes,base,lane);
            }
            const int ak=ktile*128+lane_group*16;
            if(ActLds){}
            else if(Classes>0 && GLM53_X_R9AX==1 && ktile>0)a[ktile]=a[0];  // timing probe only
            else
            {
            *reinterpret_cast<uint4*>(a[ktile].bytes)=*reinterpret_cast<const uint4*>(input+ak);
            *reinterpret_cast<uint4*>(a[ktile].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
            }
            if(ScalarScales){}
            else if(Classes>0 && GLM53_X_R9AX==2 && ktile>0)sa[ktile]=sa[0];  // timing probe only
            else
            sa[ktile]=scales[static_cast<int64_t>(row)*(K/32)+ktile*4+lane_group];
        }
#pragma unroll
        for(int t=0;t<Triplets;++t)bases[t]=iq2r_triplet_bases<Classes,KT>(aux,first+t*Atoms,lane_row);
        if constexpr(Classes>0 && (GLM53_X_R9STG>0 || GLM53_X_R9LATE))
        {
            // Stagger the book DMAs so the books land one after another and
            // early routes decode while later books stream in.
            if constexpr(GLM53_X_R9STG>0)
            {
            const int order=GLM53_X_R9ORD?(route%4==0?route/4:route%4==1?3+route/4:route%4==2?5+route/4:7+route/4):route;
            for(int i=0;i<order;++i)__builtin_amdgcn_s_sleep(GLM53_X_R9STG);
            }
            iq2r_stage_codebook_async<BookWords,1>(shared.codebook[route],aux,0,lane);
        }
        if constexpr(Classes>0 && GLM53_X_R9VG)
        {
#pragma unroll
            for(int i=0;i<StageChunks;++i)reinterpret_cast<uint4*>(shared.codebook[route])[i*64+lane]=staged[i];
        }
        else if constexpr(Classes>0)
        {
            if constexpr(Help || GLM53_X_R9EB) __builtin_amdgcn_s_barrier();  // helpers have cleared the flags
            if constexpr(Grouped)
                for(int i=issue_group;i<IssueGroups-1;++i)__builtin_amdgcn_s_barrier();
            if constexpr(GLM53_X_R9GEN)
                if(!GenEarly || route<4)iq2r_v2_form_copies<Classes>(shared.codebook[route],gen_base,lane);
            if constexpr(TS) ts[2]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            if constexpr(TS) ts[3]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
            // route 8's loads go out once this wave's own data is in
            if constexpr(Xs && GLM53_X_R9X==2)
                if(route<4 && valid8)xload();
        }
        if constexpr(Early)
        {
            // after the vector wait: the scalar loads' wait lands here
            asm volatile("" : "+s"(early_gain));
#pragma unroll
            for(int r=0;r<9;++r)asm volatile("" : "+s"(early_weights[r]));
        }
        if constexpr(ScalarScales)
        {
#pragma unroll
            for(int ktile=0;ktile<KT;++ktile)
            {
                uint32_t word=scale_words[ktile];
                if constexpr(Early) asm volatile("" : "+s"(word));
                sa[ktile]=__builtin_amdgcn_perm(0u,word,static_cast<uint32_t>(lane_group)*0x01010101u);
            }
        }
        if constexpr(ActLds)
        {
#pragma unroll
            for(int ktile=0;ktile<KT;++ktile)
            {
                const int ak=ktile*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[ktile].bytes)=*reinterpret_cast<const uint4*>(shared.act[route]+ak);
                *reinterpret_cast<uint4*>(a[ktile].bytes+16)=*reinterpret_cast<const uint4*>(shared.act[route]+ak+64);
            }
        }
        if constexpr(Classes>0 && GLM53_X_R9BAR) __syncthreads();
        const bool helped=Help && route%4==0;
        if constexpr(Help)
            if(helped)
            {
                __hip_atomic_store(&shared.help_flag[route/4],1u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_WORKGROUP);
                shared.help_gain[route/4]=iq2r_gain<Classes>(aux,auxiliary_bytes,0);
            }
        constexpr bool Pipe=Classes>0 && GLM53_X_R9PIPE;
        [[maybe_unused]] uint64_t raw[2][Atoms][4];
        const auto gather=[&](int unit,uint64_t (&r)[Atoms][4])
        {
            const IQ2RCompressedTriplet& c=compressed[unit/KT][unit%KT];
            const IQ2RLevels levels=iq2r_levels<Classes>(c.third.z,rb_bits);
            glm53_gather_atom<0,Classes>(c,rb_book,levels,rb_high,r[0]);
            glm53_gather_atom<1,Classes>(c,rb_book,levels,rb_high,r[1]);
            glm53_gather_atom<2,Classes>(c,rb_book,levels,rb_high,r[2]);
        };
        if constexpr(Pipe) gather(0,raw[0]);
#pragma unroll
        for(int t=0;t<Triplets;++t)
        {
            opus::vector_t<float,4> k_sums[KT][Atoms]={};
#pragma unroll
            for(int ktile=0;ktile<KT;++ktile)
            {
                if(helped && t==Triplets-1 && ktile>=KT-HelpTiles)break;
                if constexpr(Pipe)
                {
                    const int unit=t*KT+ktile;
                    if(unit+1<Triplets*KT)gather(unit+1,raw[(unit+1)&1]);
                    const IQ2RCompressedTriplet& c=compressed[t][ktile];
                    opus::i32x8_t b[3]={glm53_sign_atom<0>(c,raw[unit&1][0]),glm53_sign_atom<1>(c,raw[unit&1][1]),
                                        glm53_sign_atom<2>(c,raw[unit&1][2])};
                    uint32_t s=sa[ktile];
                    asm volatile("" : "+v"(s));
                    iq2r_triplet_mfma<0>(a[ktile].words,b,k_sums[ktile],s,iq2r_scale_bytes<Classes>(iq2r_kt_bases<Classes>(bases[t],ktile),c.third.z));
                    continue;
                }
                opus::i32x8_t b[3];
                uint32_t sb0,sb1,sb2;
                const uint32_t kt_bases=iq2r_kt_bases<Classes>(bases[t],ktile);
                if constexpr(GLM53_X_R9ND)
                {
                    const auto& c=compressed[t][ktile];
#pragma unroll
                    for(int i=0;i<3;++i)
                        b[i]=opus::i32x8_t{static_cast<int>(c.paired.x),static_cast<int>(c.paired.y),static_cast<int>(c.paired.z),static_cast<int>(c.paired.w),
                                           static_cast<int>(c.third.x),static_cast<int>(c.third.y),static_cast<int>(c.third.z^i),static_cast<int>(kt_bases)};
                    sb0=kt_bases;sb1=kt_bases>>8;sb2=kt_bases>>16;
                }
                else
                {
                glm53_decode_packed_atom<0,Classes>(compressed[t][ktile],rb_book,kt_bases,b[0],sb0,rb_bits,rb_high);
                glm53_decode_packed_atom<1,Classes>(compressed[t][ktile],rb_book,kt_bases,b[1],sb1,rb_bits,rb_high);
                glm53_decode_packed_atom<2,Classes>(compressed[t][ktile],rb_book,kt_bases,b[2],sb2,rb_bits,rb_high);
                }
                const uint32_t sb=Classes>0 && GLM53_X_R9SB?iq2r_scale_bytes<Classes>(kt_bases,compressed[t][ktile].third.z)
                                                           :sb0|(sb1<<8)|(sb2<<16);
                uint32_t s=sa[ktile];
                asm volatile("" : "+v"(s));
                iq2r_triplet_mfma<0>(a[ktile].words,b,k_sums[ktile],ScalarScales?s:s*0x01010101u,sb);
            }
            // Every lane group stores, not only lane group 0, so the MFMAs above are
            // not sunk into a lane_group==0 branch (see glm53_row_offset). Lane
            // groups 1-3 write the spare slot, which is never read.
#pragma unroll
            for(int atom=0;atom<Atoms;++atom)
            {
                // ((k0+k1)+k2)+k3 in K-tile order.
                float value=k_sums[0][atom][0];
                if(helped && t==Triplets-1)
                {
                    // the helper adds the remaining K tiles in the combine
#pragma unroll
                    for(int ktile=1;ktile<KT-HelpTiles;++ktile)value+=k_sums[ktile][atom][0];
                    *(lane_group==0?&shared.help_sums[route/4][0][atom*16+lane_row]:shared.help_spare)=value;
                    continue;
                }
#pragma unroll
                for(int ktile=1;ktile<KT;++ktile)value+=k_sums[ktile][atom][0];
                shared.routes[route][lane_group==0?(t*Atoms+atom)*16+lane_row:Columns]=__float2bfloat16(value*(Early?early_gain:iq2r_gain<Classes>(aux,auxiliary_bytes,0)));
            }
        }
        if constexpr(TS) ts[4]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
    }
    else
    {
        if constexpr(Help || (Classes>0 && GLM53_X_R9EB)) __builtin_amdgcn_s_barrier();
        if constexpr(Grouped)
            for(int i=issue_group;i<IssueGroups-1;++i)__builtin_amdgcn_s_barrier();
        if constexpr(Classes>0 && GLM53_X_R9BAR) __syncthreads();
        if(lane_group==0)
#pragma unroll
        for(int column=lane_row;column<Columns;column+=16)
            shared.routes[route][column]=__float2bfloat16(0.0f);
        if constexpr(Help)
            if(route%4==0)
            {
                if(lane_group==0)
#pragma unroll
                    for(int atom=0;atom<Atoms;++atom)shared.help_sums[route/4][0][atom*16+lane_row]=0.0f;
                shared.help_gain[route/4]=0.0f;
            }
    }
route8_staged:
    if constexpr(Xs)
        if(route<4 && valid8)
        {
            if(GLM53_X_R9X==2 && !valid)xload();
            while(__builtin_amdgcn_readfirstlane(__hip_atomic_load(&shared.help_flag[0],__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_WORKGROUP))==0u)
                __builtin_amdgcn_s_sleep(1);
            if constexpr(TS) ts[7]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
            constexpr uint32_t Bits8=(8u&((1u<<RouteLowBits)-1u))*(0x01010101u<<(8-RouteLowBits));
            constexpr uint32_t High8=8u>>RouteLowBits;
            static_assert(!Xs || GLM53_X_RB);
#pragma unroll
            for(int j=0;j<XPer;++j)
            {
                const int u=route+4*j;
                if(SplitUnits%4!=0 && u>=SplitUnits)break;
                const int t=u/KT,ktile=u%KT;
                const IQ2RCompressedTriplet& c=x_c[j];
                IQ2RActivationFragment a8;
                const int ak=ktile*128+lane_group*16;
                *reinterpret_cast<uint4*>(a8.bytes)=*reinterpret_cast<const uint4*>(shared.act[8]+ak);
                *reinterpret_cast<uint4*>(a8.bytes+16)=*reinterpret_cast<const uint4*>(shared.act[8]+ak+64);
                const IQ2RLevels levels=iq2r_levels<Classes>(c.third.z,Bits8);
                uint64_t r8[Atoms][4];
                glm53_gather_atom<0,Classes>(c,shared.codebook[0],levels,High8,r8[0]);
                glm53_gather_atom<1,Classes>(c,shared.codebook[0],levels,High8,r8[1]);
                glm53_gather_atom<2,Classes>(c,shared.codebook[0],levels,High8,r8[2]);
                opus::i32x8_t w[3]={glm53_sign_atom<0>(c,r8[0]),glm53_sign_atom<1>(c,r8[1]),glm53_sign_atom<2>(c,r8[2])};
                const uint32_t word=x_words[j];
                uint32_t s8=__builtin_amdgcn_perm(0u,word,static_cast<uint32_t>(lane_group)*0x01010101u);
                asm volatile("" : "+v"(s8));
                opus::vector_t<float,4> part[Atoms]={};
                iq2r_triplet_mfma<0>(a8.words,w,part,s8,iq2r_scale_bytes<Classes>(iq2r_kt_bases<Classes>(x_bases[j],ktile),c.third.z));
#pragma unroll
                for(int atom=0;atom<Atoms;++atom)
                    *(lane_group==0?&shared.split_sums[u*Atoms+atom][lane_row]:shared.split_spare)=part[atom][0];
            }
        }
    if constexpr(Split)
    {
        // Every wave staged a share of route 8's copies.
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        if(GLM53_X_R9S8!=3 && valid8)
        {
            constexpr bool RB=GLM53_X_RB;
            constexpr uint32_t Bits8=RB?(8u&((1u<<RouteLowBits)-1u))*(0x01010101u<<(8-RouteLowBits)):0u;
            constexpr uint32_t High8=RB?8u>>RouteLowBits:0u;
            const uint64_t* book8=RB?shared.codebook[0]:shared.codebook[8];
#pragma unroll
            for(int j=0;j<SplitPerWave;++j)
            {
                const int u=route+8*j;
                if(SplitUnits%8!=0 && u>=SplitUnits)break;
                const int ktile=u%KT;
                const uint32_t kt_bases=iq2r_kt_bases<Classes>(split_bases[j],ktile);
                opus::i32x8_t w[3];
                uint32_t sb0,sb1,sb2;
                glm53_decode_packed_atom<0,Classes>(split_c[j],book8,kt_bases,w[0],sb0,Bits8,High8);
                glm53_decode_packed_atom<1,Classes>(split_c[j],book8,kt_bases,w[1],sb1,Bits8,High8);
                glm53_decode_packed_atom<2,Classes>(split_c[j],book8,kt_bases,w[2],sb2,Bits8,High8);
                uint32_t s=split_sa[j];
                asm volatile("" : "+v"(s));
                opus::vector_t<float,4> part[Atoms]={};
                iq2r_triplet_mfma<0>(split_a[j].words,w,part,s*0x01010101u,sb0|(sb1<<8)|(sb2<<16));
#pragma unroll
                for(int atom=0;atom<Atoms;++atom)
                    *(lane_group==0?&shared.split_sums[u*Atoms+atom][lane_row]:shared.split_spare)=part[atom][0];
            }
        }
    }
    __syncthreads();
    if constexpr(TS) ts[5]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
    if(route==0 && lane_group<Triplets)
    {
#pragma unroll
        for(int atom=0;atom<Atoms;++atom)
        {
            const int column=(lane_group*Atoms+atom)*16+lane_row;
            float combined=0.0f;
#pragma unroll
            for(int r=0;r<(Split || Xs?8:9);++r)
            {
                float value=__bfloat162float(shared.routes[r][column]);
                if constexpr(Help)
                    if(r%4==0 && lane_group==Triplets-1)
                    {
                        float sum=shared.help_sums[r/4][0][atom*16+lane_row];
#pragma unroll
                        for(int j=1;j<=HelpTiles;++j)sum+=shared.help_sums[r/4][j][atom*16+lane_row];
                        value=__bfloat162float(__float2bfloat16(sum*shared.help_gain[r/4]));
                    }
                combined=fmaf(value,Early?early_weights[r]:route_weights[token*9+r],combined);
            }
            if constexpr(Split || Xs)
            {
                float value=0.0f;
                if(valid8)
                {
                    // ((k0+k1)+k2)+k3 in K-tile order, as the route waves sum
                    float sum=shared.split_sums[lane_group*KT*Atoms+atom][lane_row];
#pragma unroll
                    for(int ktile=1;ktile<KT;++ktile)sum+=shared.split_sums[(lane_group*KT+ktile)*Atoms+atom][lane_row];
                    value=__bfloat162float(__float2bfloat16(sum*gain8));
                }
                combined=fmaf(value,route_weights[token*9+8],combined);
            }
            output[static_cast<int64_t>(token)*N+first*16+column]=__float2bfloat16(combined);
        }
    }
    if constexpr(TS)
    {
        ts[6]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
        ts[9]=static_cast<uint32_t>(__builtin_amdgcn_s_memrealtime());
        ts[10]=__builtin_amdgcn_s_getreg((31<<11)|4);
        ts[11]=__builtin_amdgcn_s_getreg((31<<11)|20);
        uint32_t* out=reinterpret_cast<uint32_t*>(output+static_cast<int64_t>(gridDim.y)*N)+((blockIdx.y*gridDim.x+blockIdx.x)*9+route)*16;
        if(lane<14)
        {
            uint32_t v=ts[0];
#pragma unroll
            for(int i=1;i<14;++i)v=lane==i?ts[i]:v;
            out[lane]=v;
        }
    }
#endif
}

// IQ2R-v2 down for M<=4 on 12 waves. One wave per route puts three of the
// nine waves on one SIMD, whose decode then sets the time; here the routes'
// (triplet, K tile) units go round-robin over 12 waves (3 per SIMD) and the
// codebook copies are staged by all waves (12 KB each). Per-unit MFMA sums go
// through LDS and are added in K-tile order before the gain and BF16 rounding,
// then the routes are combined with the same nine ordered FP32 FMAs, so the
// output equals glm53_down_route9_kernel's.
template<int KT,int Classes,int Triplets>
__global__ __launch_bounds__(768,1) void glm53_down_route12_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ scatter,
    const float* __restrict__ route_weights,
    __hip_bfloat16* __restrict__ output,
    int data_bytes, int auxiliary_bytes)
{
#if defined(__gfx950__)
    static_assert(Classes>0);
    constexpr int Atoms=3,N=6144,K=KT*128,Columns=Triplets*Atoms*16;
    constexpr int Waves=12,Units=9*Triplets*KT,Rounds=(Units+Waves-1)/Waves;
    constexpr int BookWords=iq2r_codebook_words<Classes,1>(),BookChunks=BookWords*8/1024;
    constexpr int StageChunks=9*BookChunks/Waves;
    static_assert(9*BookChunks%Waves==0 && StageChunks<=BookChunks);
    struct Storage {
        alignas(16) uint64_t codebook[9][BookWords];
        // Unit u = (route * Triplets + triplet) * KT + K tile, then a spare
        // slot that lane groups 1-3 write.
        alignas(16) float partial[Units][Atoms*16];
        float spare[4];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=__builtin_amdgcn_readfirstlane(threadIdx.y);
    const int lane_row=lane%16,lane_group=lane/16;
    const int token=blockIdx.y,first=static_cast<int>(blockIdx.x)*Triplets*Atoms;
    const int rows=static_cast<int>(gridDim.y)*9;
    // Scalar route lookups, all issued before the first use. An invalid
    // route reads expert 0 / row 0 and is skipped after the loads.
    struct Route { int expert,row; bool valid; };
    const auto route_of=[&](int route){
        Route r;
        r.expert=__builtin_amdgcn_readfirstlane(expert_ids[token*9+route]);
        r.row=__builtin_amdgcn_readfirstlane(scatter[token*9+route]);
        r.valid=r.expert>=0 && r.expert<257 && r.row>=0 && r.row<rows;
        if(!r.valid)r.expert=r.row=0;
        return r;
    };
    // This wave's StageChunks consecutive 1 KB chunks of the nine books span
    // at most two routes.
    const int chunk0=wave*StageChunks,route0=chunk0/BookChunks;
    const int split=min(StageChunks,(route0+1)*BookChunks-chunk0);
    const Route stage[2]={route_of(route0),route_of(min(route0+1,8))};
    Route units[Rounds];
#pragma unroll
    for(int round=0;round<Rounds;++round)
        units[round]=route_of(min(wave+round*Waves,Units-1)/(Triplets*KT));
#pragma unroll
    for(int part=0;part<2;++part)
    {
        const int route=route0+part;
        if(!stage[part].valid || (part==1 && split==StageChunks))continue;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(stage[part].expert)*auxiliary_bytes;
        opus::gmem<uint8_t> buffer(aux,static_cast<unsigned int>(BookWords*8));
        const int begin=part==0?chunk0:chunk0+split,end=part==0?chunk0+split:chunk0+StageChunks;
        for(int chunk=begin;chunk<end;++chunk)
        {
            const int local=chunk-route*BookChunks;
            buffer.template async_load<16>(reinterpret_cast<uint8_t*>(shared.codebook[route])+local*1024,lane*16,local*1024,
                                           opus::number<0>{},opus::number<0>{});
        }
    }
    asm volatile("" ::: "memory");
    IQ2RCompressedTriplet compressed[Rounds];
    IQ2RActivationFragment a[Rounds];
    uint32_t sa[Rounds],kt_bases[Rounds];
#pragma unroll
    for(int round=0;round<Rounds;++round)
    {
        const int unit=min(wave+round*Waves,Units-1);
        const int t=unit/KT%Triplets,ktile=unit%KT;
        const uint8_t* data=all_data+static_cast<int64_t>(units[round].expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(units[round].expert)*auxiliary_bytes;
        const int base=static_cast<int>(triplet_base(physical_tile(first+t*Atoms,ktile,KT)));
        compressed[round]=iq2r_scheduled_load_compact_uniform(data,data_bytes,base,lane);
        const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(units[round].row)*K);
        const int ak=ktile*128+lane_group*16;
        *reinterpret_cast<uint4*>(a[round].bytes)=*reinterpret_cast<const uint4*>(input+ak);
        *reinterpret_cast<uint4*>(a[round].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
        sa[round]=scales[static_cast<int64_t>(units[round].row)*(K/32)+ktile*4+lane_group];
        kt_bases[round]=iq2r_v2_bases<Classes,1>(aux,(first+t*Atoms)/3,KT/2,ktile/2,lane_row);
    }
    // The combining waves fetch every route's gain and weight now, so their
    // scalar loads overlap the staging.
    constexpr int CombineWaves=(Columns+63)/64;
    float gains[9],weights[9];
    bool valid[9];
    if(wave<CombineWaves)
    {
#pragma unroll
        for(int route=0;route<9;++route)
        {
            const Route r=route_of(route);
            valid[route]=r.valid;
            gains[route]=iq2r_gain<Classes>(all_auxiliary+static_cast<int64_t>(r.expert)*auxiliary_bytes,auxiliary_bytes,0);
            weights[route]=route_weights[token*9+route];
        }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
#pragma unroll
    for(int round=0;round<Rounds;++round)
    {
        // Branch-free across rounds: an invalid route's unit decodes garbage
        // that the combine ignores, a unit past the end writes the spare slot.
        const int unit=min(wave+round*Waves,Units-1);
        const bool store=Units%Waves==0 || wave+round*Waves<Units;
        const uint64_t* route_book=shared.codebook[unit/(Triplets*KT)];
        opus::i32x8_t b[3];
        uint32_t sb0,sb1,sb2;
        glm53_decode_packed_atom<0,Classes>(compressed[round],route_book,kt_bases[round],b[0],sb0);
        glm53_decode_packed_atom<1,Classes>(compressed[round],route_book,kt_bases[round],b[1],sb1);
        glm53_decode_packed_atom<2,Classes>(compressed[round],route_book,kt_bases[round],b[2],sb2);
        const uint32_t sb=sb0|(sb1<<8)|(sb2<<16);
        uint32_t s=sa[round];
        asm volatile("" : "+v"(s));
        opus::vector_t<float,4> sums[Atoms]={};
        iq2r_triplet_mfma<0>(a[round].words,b,sums,s*0x01010101u,sb);
        // Every lane group stores (see glm53_down_route9_kernel).
#pragma unroll
        for(int atom=0;atom<Atoms;++atom)
            *(lane_group==0 && store?&shared.partial[unit][atom*16+lane_row]:shared.spare)=sums[atom][0];
    }
    __syncthreads();
    const int column=wave*64+lane;
    if(wave<CombineWaves && column<Columns)
    {
        const int t=column/(Atoms*16),offset=column%(Atoms*16);
        float combined=0.0f;
#pragma unroll
        for(int route=0;route<9;++route)
        {
            float value=0.0f;
            if(valid[route])
            {
                // ((k0+k1)+k2)+k3 in K-tile order.
                const int unit=(route*Triplets+t)*KT;
                value=shared.partial[unit][offset];
#pragma unroll
                for(int ktile=1;ktile<KT;++ktile)value+=shared.partial[unit+ktile][offset];
                value=__bfloat162float(__float2bfloat16(value*gains[route]));
            }
            combined=fmaf(value,weights[route],combined);
        }
        output[static_cast<int64_t>(token)*N+first*16+column]=__float2bfloat16(combined);
    }
#endif
}

// Bit j of each byte of ``signs`` is the sign of that byte of word j; shifted
// left by 7 - j it lands on the E4M3 sign bit. The pair {signs << 1, signs}
// shifted left by 6 - 2 * codeword aligns both words of a codeword at once
// (bits carried into the high word land below bit 7 and are masked off).
__device__ __forceinline__ uint64_t glm53_sign_pair(uint32_t signs)
{
    uint32_t lo;
    asm volatile("v_lshlrev_b32 %0, 1, %1" : "=v"(lo) : "v"(signs));
    return (static_cast<uint64_t>(signs)<<32)|lo;
}

template<int Word>
__device__ __forceinline__ uint64_t glm53_sign_word(uint64_t magnitude,uint64_t sign_pair)
{
    uint64_t aligned=sign_pair;
    if constexpr(Word<3)
        asm volatile("v_lshlrev_b64 %0, %1, %2" : "=v"(aligned) : "n"(6-2*Word),"v"(sign_pair));
    uint32_t lo,hi;
    asm volatile(
        "v_and_or_b32 %0, %2, %4, %3\n\t"
        "v_and_or_b32 %1, %5, %4, %6\n\t"
        : "=&v"(lo),"=&v"(hi)
        : "v"(static_cast<uint32_t>(aligned)),"v"(static_cast<uint32_t>(magnitude)),"s"(0x80808080u),
          "v"(static_cast<uint32_t>(aligned>>32)),"v"(static_cast<uint32_t>(magnitude>>32)));
    return (static_cast<uint64_t>(hi)<<32)|lo;
}

template<int MAtoms,int Classes>
__device__ __forceinline__ void glm53_quad_decode_mfma(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m)
{
    const IQ2RLevels levels=iq2r_levels<Classes>(compressed.metadata,iq2r_book_bits<Classes>(__lane_id()&15));
    if constexpr(Classes>0) bases=iq2r_scale_bytes<Classes>(bases,compressed.metadata);
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        const uint32_t lows=atom==0?compressed.first.x:atom==1?compressed.first.z:atom==2?compressed.second.x:compressed.second.z;
        const uint32_t signs=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
        const uint32_t highs=(compressed.metadata>>(atom*8))&15u;
        union { opus::i32x8_t words;uint64_t codewords[4]; } decoded;
        const uint64_t* book0=codebook+iq2r_copy_offset<Classes>(levels.low,atom)/8;
        const uint64_t* book1=codebook+iq2r_copy_offset<Classes>(levels.high,atom)/8;
        uint64_t mag[4];
        mag[0]=book0[lows&511u];
        mag[1]=book0[(lows>>9)&511u];
        mag[2]=book1[(lows>>18)&511u];
        mag[3]=book1[(lows>>27)|(highs<<5)];
        asm volatile("" : "+v"(mag[0]), "+v"(mag[1]), "+v"(mag[2]), "+v"(mag[3]));
        const uint64_t sign_pair=glm53_sign_pair(signs);
        decoded.codewords[0]=glm53_sign_word<0>(mag[0],sign_pair);
        decoded.codewords[1]=glm53_sign_word<1>(mag[1],sign_pair);
        decoded.codewords[2]=glm53_sign_word<2>(mag[2],sign_pair);
        decoded.codewords[3]=glm53_sign_word<3>(mag[3],sign_pair);
        const uint32_t sb=Classes==0?iq2r_scale_b<0>(bases,compressed.metadata,atom):(bases>>(atom*8))&255u;
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            accumulators[m][atom]=mma(activation[m].words,decoded.words,accumulators[m][atom],
                                      scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }
    }
}

// TP8 (K=256) down, ordered variant: same tiling as glm53_down_prefill_tp8_kernel,
// but the K1 MFMA accumulates onto the K0 result (one FP32 chain per output).
template<int Waves,int MAtoms,int Groups,int XCD,int TaskGroup,int Classes=0>
__global__ __launch_bounds__(64*Waves,1) void glm53_down_tp8_ordered_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    __hip_bfloat16* __restrict__ output,
    int M,int expert_count,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int Rows=16*MAtoms,Columns=48*Waves*Groups,Threads=64*Waves;
    struct Storage {
        alignas(16) uint64_t codebook[iq2r_codebook_words<Classes,1>()];
        alignas(16) uint8_t input[Rows*256];
        uint8_t scales[Rows*8];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=Classes>0 && GLM53_X_A32?__builtin_amdgcn_readfirstlane(threadIdx.y):threadIdx.y,linear=wave*64+lane;
    const int lane_row=lane%16,lane_group=lane/16;
    const int nt=task_count[0];
    opus::gmem<uint8_t> a_buffer(activations,static_cast<unsigned int>(M*256));
    // Descriptor bounds are bytes; typed offsets below are BF16 elements.
    opus::gmem<opus::bf16_t> out_buffer(output,static_cast<unsigned int>(M*6144*sizeof(__hip_bfloat16)));
    opus::gmem<uint8_t> scale_buffer(activation_scales,static_cast<unsigned int>(M*8));
    int previous_expert=-1;
    for(int work=blockIdx.x;work<nt*(6144/Columns);work+=gridDim.x)
    {
        int index=work;
        if constexpr(XCD>0)
        {
            const int total=nt*(6144/Columns);
            const int per=(total+XCD-1)/XCD,tall=total%XCD==0?XCD:total%XCD;
            const int die=work%XCD,local=work/XCD;
            index=die<tall?die*per+local:tall*per+(die-tall)*(per-1)+local;
        }
        int task=index%nt,output_tile=index/nt;
        if constexpr(TaskGroup>0)
        {
            const int first=(index/(TaskGroup*(6144/Columns)))*TaskGroup;
            const int valid=min(TaskGroup,nt-first),local=index%(TaskGroup*(6144/Columns));
            task=first+local%valid;output_tile=local/valid;
        }
        const int n_tile=output_tile;
        const int begin=tasks[task*3],count=tasks[task*3+1],expert=tasks[task*3+2];
        if(begin<0 || count<=0 || begin+count>M || expert<0 || expert>=expert_count)continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(previous_expert!=expert)
        {
            if constexpr(Classes==0) { for(int i=linear;i<512;i+=Threads)shared.codebook[i]=reinterpret_cast<const uint64_t*>(aux)[i]; }
            else iq2r_stage_codebook_async<iq2r_codebook_words<Classes,1>(),Waves>(shared.codebook,aux,wave,lane);
            previous_expert=expert;
        }
        for(int row_base=begin;row_base<begin+count;row_base+=Rows)
        {
            const int row_count=min(Rows,begin+count-row_base);
            const int active_m=(row_count+15)/16;
#pragma unroll
            for(int copy=0;copy<(Rows*256+Threads*16-1)/(Threads*16);++copy)
            {
                const int offset=(linear+copy*Threads)*16;
                const int r=offset/256,col=(offset%256)^((r&15)<<4);
                const int src=(row_base+min(r,row_count-1))*256+col;
                const int dest=(wave*64+copy*Threads)*16;
                if(dest<Rows*256)a_buffer.template async_load<16>(shared.input+dest,src);
            }
            if(linear<Rows*2)
            {
                const int r=linear/2,col=(linear%2)*4;
                scale_buffer.template async_load<4>(shared.scales+wave*64*4,(row_base+min(r,row_count-1))*8+col);
            }
            IQ2RCompressedTriplet pending[2];
            uint64_t pending_bases=0;
            {
                const int nb=n_tile*(Columns/16)+wave*3;
                pending_bases=iq2r_triplet_bases<Classes,2>(aux,nb,lane_row);
#pragma unroll
                for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,Classes>0 && GLM53_X_A32?glm53_ws_source_base<2>(nb,kt):static_cast<int>(triplet_base(physical_tile(nb,kt,2))),lane);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();

#pragma unroll 1
            for(int group=0;group<Groups;++group)
            {
        const int n_block=n_tile*(Columns/16)+wave*3+group*Waves*3;
        const uint64_t bases=pending_bases;
        IQ2RCompressedTriplet compressed[2];
        compressed[0]=pending[0];compressed[1]=pending[1];
        if(group+1<Groups)
        {
            const int nb=n_block+Waves*3;
            pending_bases=iq2r_triplet_bases<Classes,2>(aux,nb,lane_row);
#pragma unroll
            for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,Classes>0 && GLM53_X_A32?glm53_ws_source_base<2>(nb,kt):static_cast<int>(triplet_base(physical_tile(nb,kt,2))),lane);
            asm volatile("" ::: "memory");
        }
            [[maybe_unused]] IQ2RLevels lv[2];[[maybe_unused]] uint32_t sc[2];
            if constexpr(Classes>0 && GLM53_X_T8LV && !(GLM53_X_T8KO && MAtoms==1))
#pragma unroll
                for(int kt=0;kt<2;++kt)
                {
                    lv[kt]=iq2r_levels<Classes>(compressed[kt].third.z);
                    sc[kt]=iq2r_scale_bytes<Classes>(iq2r_kt_bases<Classes>(bases,kt),compressed[kt].third.z);
                }
            if constexpr(Classes>0 && GLM53_X_T8KO && MAtoms==1)
            {
                // K tiles outermost: the three atoms share each K tile's
                // activation fragments; K1 still accumulates onto K0's result.
                auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                opus::vector_t<float,4> first[3][MAtoms];
                {
                    IQ2RActivationFragment a[MAtoms];uint32_t sa[MAtoms];
#pragma unroll
                    for(int m=0;m<MAtoms;++m)
                    {
                        if(m>=active_m)continue;
                        const int r=m*16+lane_row,col=lane_group*16;
                        *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                        *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                        sa[m]=shared.scales[r*8+lane_group];
                    }
#define GLM53_TP8_K0(ATOM) \
                    { \
                        opus::i32x8_t w;uint32_t sb; \
                        glm53_decode_packed_atom<ATOM,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w,sb); \
                        _Pragma("unroll") \
                        for(int m=0;m<MAtoms;++m) \
                        { \
                            if(m>=active_m)continue; \
                            first[ATOM][m]=mma(w,a[m].words,opus::vector_t<float,4>{},sb,sa[m]); \
                        } \
                    }
                    GLM53_TP8_K0(0) GLM53_TP8_K0(1) GLM53_TP8_K0(2)
#undef GLM53_TP8_K0
                }
                {
                    IQ2RActivationFragment a[MAtoms];uint32_t sa[MAtoms];
#pragma unroll
                    for(int m=0;m<MAtoms;++m)
                    {
                        if(m>=active_m)continue;
                        const int r=m*16+lane_row,col=128+lane_group*16;
                        *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                        *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                        sa[m]=shared.scales[r*8+4+lane_group];
                    }
#define GLM53_TP8_K1(ATOM) \
                    { \
                        opus::i32x8_t w;uint32_t sb; \
                        glm53_decode_packed_atom<ATOM,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w,sb); \
                        _Pragma("unroll") \
                        for(int m=0;m<MAtoms;++m) \
                        { \
                            if(m>=active_m)continue; \
                            const opus::vector_t<float,4> second=mma(w,a[m].words,first[ATOM][m],sb,sa[m]); \
                            const int out_r=m*16+lane_row; \
                            const int out_col=n_block*16+ATOM*16+lane_group*4; \
                            opus::vector_t<opus::bf16_t,4> packed; \
                            _Pragma("unroll") \
                            for(int c=0;c<4;++c)packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(second[c]*iq2r_gain<Classes>(aux,auxiliary_bytes,0))); \
                            out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,(row_base+out_r)*6144+out_col)); \
                        } \
                    }
                    GLM53_TP8_K1(0) GLM53_TP8_K1(1) GLM53_TP8_K1(2)
#undef GLM53_TP8_K1
                }
            }
            else
            {
            // One N16 atom at a time: K0 MFMA, then K1 MFMA onto its result.
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<0,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0,0u,0u,Classes>0 && GLM53_X_T8LV?&lv[0]:nullptr,Classes>0 && GLM53_X_T8LV?&sc[0]:nullptr);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<0,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1,0u,0u,Classes>0 && GLM53_X_T8LV?&lv[1]:nullptr,Classes>0 && GLM53_X_T8LV?&sc[1]:nullptr);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,first[m],sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+0*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(second[c]*iq2r_gain<Classes>(aux,auxiliary_bytes,0)));
                    out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,(row_base+out_r)*6144+out_col));
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<1,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0,0u,0u,Classes>0 && GLM53_X_T8LV?&lv[0]:nullptr,Classes>0 && GLM53_X_T8LV?&sc[0]:nullptr);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<1,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1,0u,0u,Classes>0 && GLM53_X_T8LV?&lv[1]:nullptr,Classes>0 && GLM53_X_T8LV?&sc[1]:nullptr);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,first[m],sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+1*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(second[c]*iq2r_gain<Classes>(aux,auxiliary_bytes,0)));
                    out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,(row_base+out_r)*6144+out_col));
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<2,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0,0u,0u,Classes>0 && GLM53_X_T8LV?&lv[0]:nullptr,Classes>0 && GLM53_X_T8LV?&sc[0]:nullptr);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+lane_group];
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    first[m]=mma(w0,a.words,first[m],sb0,sa);
                }
                opus::i32x8_t w1;uint32_t sb1;
                glm53_decode_packed_atom<2,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1,0u,0u,Classes>0 && GLM53_X_T8LV?&lv[1]:nullptr,Classes>0 && GLM53_X_T8LV?&sc[1]:nullptr);
#pragma unroll
                for(int m=0;m<MAtoms;++m)
                {
                    if(m>=active_m)continue;
                    const int r=m*16+lane_row,col=128+lane_group*16;
                    IQ2RActivationFragment a;
                    *reinterpret_cast<uint4*>(a.bytes)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col)^((r&15)<<4)));
                    *reinterpret_cast<uint4*>(a.bytes+16)=*reinterpret_cast<const uint4*>(shared.input+((r*256+col+64)^((r&15)<<4)));
                    const uint32_t sa=shared.scales[r*8+4+lane_group];
                    opus::vector_t<float,4> second={};
                    auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
                    second=mma(w1,a.words,first[m],sb1,sa);
                    const int out_r=m*16+lane_row;
                    const int out_col=n_block*16+2*16+lane_group*4;
                    opus::vector_t<opus::bf16_t,4> packed;
#pragma unroll
                    for(int c=0;c<4;++c)packed[c]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(second[c]*iq2r_gain<Classes>(aux,auxiliary_bytes,0)));
                    out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,(row_base+out_r)*6144+out_col));
                }
            }
            }
            }
            __syncthreads();
        }
    }
#endif
}

// Each explicit ds_read_b64 contributes one outstanding LDS operation.
// A group consumes four operations. Callers issue the next group first, so
// wait4 completes the current group while the next four reads stay in flight.
template<int Classes=0>
__device__ __forceinline__ void glm53_issue_codebook(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,int atom,
    uint64_t (&mag)[4],IQ2RLevels levels={0u,0u})
{
    const uint32_t lows=atom==0?compressed.first.x:atom==1?compressed.first.z:atom==2?compressed.second.x:compressed.second.z;
    const uint32_t highs=(compressed.metadata>>(atom*8))&15u;
    const uint32_t base=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(codebook));
    // Copies are 4 KB apart and an entry offset is below 4 KB, so OR joins
    // them: one v_and_or_b32 per codeword, as for IQ2R.
    const uint32_t copy0=GLM53_X_NOCOPY?0u:iq2r_copy_offset<Classes>(levels.low,atom);
    const uint32_t copy1=GLM53_X_NOCOPY?0u:iq2r_copy_offset<Classes>(levels.high,atom);
    const uint32_t a0=base+(copy0|(lows&511u)*8);
    const uint32_t a1=base+(copy0|((lows>>9)&511u)*8);
    const uint32_t a2=base+(copy1|((lows>>18)&511u)*8);
    const uint32_t a3=base+(copy1|((lows>>27)|(highs<<5))*8);
    asm volatile(
        "ds_read_b64 %0, %4\n\t"
        "ds_read_b64 %1, %5\n\t"
        "ds_read_b64 %2, %6\n\t"
        "ds_read_b64 %3, %7\n\t"
        : "=&v"(mag[0]), "=&v"(mag[1]), "=&v"(mag[2]), "=&v"(mag[3])
        : "v"(a0), "v"(a1), "v"(a2), "v"(a3) : "memory");
}

template<int MAtoms,int Classes=0>
__device__ __forceinline__ void glm53_quad_decode_mfma_cached(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m)
{
    const IQ2RLevels levels=iq2r_levels<Classes>(compressed.metadata,iq2r_book_bits<Classes>(__lane_id()&15));
    if constexpr(Classes>0) bases=iq2r_scale_bytes<Classes>(bases,compressed.metadata);
    uint64_t pending[4];
    glm53_issue_codebook<Classes>(compressed,codebook,0,pending,levels);
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        uint64_t mag[4];
#pragma unroll
        for(int i=0;i<4;++i)mag[i]=pending[i];
        if(atom<3)glm53_issue_codebook<Classes>(compressed,codebook,atom+1,pending,levels);
        if(atom<3)asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        else asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        const uint32_t signs=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
        union { opus::i32x8_t words;uint64_t codewords[4]; } decoded;
        const uint64_t sign_pair=glm53_sign_pair(signs);
        decoded.codewords[0]=glm53_sign_word<0>(mag[0],sign_pair);
        decoded.codewords[1]=glm53_sign_word<1>(mag[1],sign_pair);
        decoded.codewords[2]=glm53_sign_word<2>(mag[2],sign_pair);
        decoded.codewords[3]=glm53_sign_word<3>(mag[3],sign_pair);
        const uint32_t sb=Classes==0?iq2r_scale_b<0>(bases,compressed.metadata,atom):(bases>>(atom*8))&255u;
#pragma unroll
        for(int m=0;m<MAtoms;++m) {
            if(m>=active_m)continue;
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            accumulators[m][atom]=mma(activation[m].words,decoded.words,accumulators[m][atom],
                                      scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }
    }
}

// Decode gate: one CTA per (16-row task, N tile); the eight waves each sum
// a K768 slice and the partials are reduced through LDS.
template<bool XCD,int NT,int Classes=0>
__global__ __launch_bounds__(512,1) void glm53_gate_decode_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int K=6144, MAtoms=1;
    constexpr int BookWords=iq2r_codebook_words<Classes,2>();
    struct Storage {
        alignas(16) uint64_t codebook[BookWords];
        union {
            alignas(16) uint8_t cache[8][kQuadBytes];
            alignas(16) float partial[8][4][4][64];
        } reuse;
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*NT;
    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int index=XCD?glm53_remap8(work,total):work;
        const int task=XCD?index/NT:work%num_tasks;
        const int n_tile=XCD?index%NT:work/num_tasks;
        const int row_begin=tasks[task*3];
        const int row_end=min(row_begin+16,row_begin+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting both union and codebook.
        // The codebook DMA, scales, gain and first weights load together.
        iq2r_stage_codebook_async<BookWords,8>(shared.codebook,aux,wave,lane);
        const int nbase=n_tile*4;
        uint32_t bases=0;
        if constexpr(Classes==0)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        }
        const uint64_t* lane_book=shared.codebook;
        uint32_t d_words[3]={};
        if constexpr(Classes>0)
        {
#pragma unroll
            for(int c=0;c<3;++c) d_words[c]=iq2r_v2_bases<Classes,2>(aux,n_tile,24,wave*3+c,lane_row);
        }
        const float gain=iq2r_gain<Classes>(aux,auxiliary_bytes,lane_row);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        opus::gmem<uint8_t> buffer(data,static_cast<unsigned int>(data_bytes));
        int base=(n_tile*48+wave*6)*kQuadBytes;
        iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=wave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if(m>=active_m) continue;
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            iq2r_wait_vmcnt<0>();
            const IQ2RCompressedQuad compressed=iq2r_read_quad(shared.reuse.cache[wave],lane);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if(iteration<5) {
                base+=kQuadBytes;
                iq2r_issue_quad(buffer,shared.reuse.cache[wave],lane,base);
            }
            glm53_quad_decode_mfma_cached<MAtoms,Classes>(compressed,lane_book,Classes==0?bases:d_words[iteration/2],a,sa,accumulators,active_m);
        }
        // No wave may overwrite another wave's cache before its final read.
        __syncthreads();
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
#pragma unroll
            for(int atom=0;atom<4;++atom)
                for(int c=0;c<4;++c) shared.reuse.partial[wave][atom][c][lane]=accumulators[m][atom][c];
            __syncthreads();
            if(wave<4)
            {
                const int local_row=m*16+lane_group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=shared.reuse.partial[source][atom][wave][lane];
                    values[atom]=__bfloat162float(__float2bfloat16(value*gain));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(lane_row%2==0)
                {
                    float activated[4];float abs_max=aiter::kAmaxFloor;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*(NT*32)+n_tile*32+atom*8+lane_row/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(lane_row==0) output_scales[static_cast<int64_t>(output_row)*NT+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
    }
#endif
}

template<int MAtoms,int Classes=0>
__device__ __forceinline__ void glm53_cross_k_decode_mfma(
    const IQ2RCompressedQuad& compressed,const IQ2RCompressedQuad& next,
    const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m,
    uint64_t (&pending)[4],bool first_k,bool next_k)
{
    const uint32_t book_bits=iq2r_book_bits<Classes>(__lane_id()&15);
    const IQ2RLevels levels=iq2r_levels<Classes>(compressed.metadata,book_bits);
    const IQ2RLevels next_levels=iq2r_levels<Classes>(next.metadata,book_bits);
    if constexpr(Classes>0) bases=iq2r_scale_bytes<Classes>(bases,compressed.metadata);
    if(first_k)glm53_issue_codebook<Classes>(compressed,codebook,0,pending,levels);
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        uint64_t mag[4];
#pragma unroll
        for(int i=0;i<4;++i)mag[i]=pending[i];
        if(atom<3)glm53_issue_codebook<Classes>(compressed,codebook,atom+1,pending,levels);
        else if(next_k)glm53_issue_codebook<Classes>(next,codebook,0,pending,next_levels);
        if(atom<3 || next_k)asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        else asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        const uint32_t signs=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
        union { opus::i32x8_t words;uint64_t codewords[4]; } decoded;
        const uint64_t sign_pair=glm53_sign_pair(signs);
        decoded.codewords[0]=glm53_sign_word<0>(mag[0],sign_pair);
        decoded.codewords[1]=glm53_sign_word<1>(mag[1],sign_pair);
        decoded.codewords[2]=glm53_sign_word<2>(mag[2],sign_pair);
        decoded.codewords[3]=glm53_sign_word<3>(mag[3],sign_pair);
        const uint32_t sb=Classes==0?iq2r_scale_b<0>(bases,compressed.metadata,atom):(bases>>(atom*8))&255u;
#pragma unroll
        for(int m=0;m<MAtoms;++m) {
            if(m>=active_m)continue;
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            accumulators[m][atom]=mma(activation[m].words,decoded.words,accumulators[m][atom],
                                      scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }
    }
}

template<int MAtoms,int Classes=0>
__device__ __forceinline__ void glm53_cross_k_decode_mfma_carried(
    const IQ2RCompressedQuad& compressed,const IQ2RCompressedQuad& next,
    const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m,
    uint64_t (&pending)[4],bool first_k,bool next_k,IQ2RLevels& levels,uint32_t book_bits)
{
    const IQ2RLevels next_levels=iq2r_levels<Classes>(next.metadata,book_bits);
    if constexpr(Classes>0) bases=iq2r_scale_bytes<Classes>(bases,compressed.metadata);
    if(first_k)glm53_issue_codebook<Classes>(compressed,codebook,0,pending,levels);
#pragma unroll
    for(int atom=0;atom<4;++atom)
    {
        uint64_t mag[4];
#pragma unroll
        for(int i=0;i<4;++i)mag[i]=pending[i];
        if(atom<3)glm53_issue_codebook<Classes>(compressed,codebook,atom+1,pending,levels);
        else if(next_k)glm53_issue_codebook<Classes>(next,codebook,0,pending,next_levels);
        if(atom<3 || next_k)asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        else asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        const uint32_t signs=atom==0?compressed.first.y:atom==1?compressed.first.w:atom==2?compressed.second.y:compressed.second.w;
        union { opus::i32x8_t words;uint64_t codewords[4]; } decoded;
        const uint64_t sign_pair=glm53_sign_pair(signs);
        decoded.codewords[0]=glm53_sign_word<0>(mag[0],sign_pair);
        decoded.codewords[1]=glm53_sign_word<1>(mag[1],sign_pair);
        decoded.codewords[2]=glm53_sign_word<2>(mag[2],sign_pair);
        decoded.codewords[3]=glm53_sign_word<3>(mag[3],sign_pair);
        const uint32_t sb=Classes==0?iq2r_scale_b<0>(bases,compressed.metadata,atom):(bases>>(atom*8))&255u;
#pragma unroll
        for(int m=0;m<MAtoms;++m) {
            if(m>=active_m)continue;
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            auto& acc=accumulators[m][atom];
            if constexpr(Classes>0 && GLM53_X_OPSEL) {
                // The MFMA's scale op_sel reads the atom's byte of the scale word.
                if(atom==0) acc=mma(activation[m].words,decoded.words,acc,scale_a[m],bases,opus::number<0>{},opus::number<0>{});
                else if(atom==1) acc=mma(activation[m].words,decoded.words,acc,scale_a[m],bases,opus::number<0>{},opus::number<1>{});
                else if(atom==2) acc=mma(activation[m].words,decoded.words,acc,scale_a[m],bases,opus::number<0>{},opus::number<2>{});
                else acc=mma(activation[m].words,decoded.words,acc,scale_a[m],bases,opus::number<0>{},opus::number<3>{});
            }
            else acc=mma(activation[m].words,decoded.words,acc,scale_a[m],sb,opus::number<0>{},opus::number<0>{});
        }
    }
    levels=next_levels;
}

// Decode gate variant that keeps weights in registers (no LDS staging), so
// work items only synchronize to load a new expert's codebook.
template<int NT,int Classes=0>
__global__ __launch_bounds__(512,(Classes>0 && (GLM53_X_GG || GLM53_X_XQ))?4:1) void glm53_gate_decode_nobarrier_kernel(
    const opus::fp8_t* __restrict__ activations,
    const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,
    const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,
    int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int K=6144, MAtoms=1;
    constexpr int GateBooks=Classes>0?GLM53_GATE_BOOKS:2;
    constexpr int BookWords=iq2r_codebook_words<Classes,GateBooks>();
    // GLM53_X_PALIAS: the partial sums reuse the item's own codebook buffer
    // (dead after the K loop), so two buffers fit in the LDS of one.
    constexpr bool Alias=Classes>0 && GLM53_X_PALIAS && BookWords*8>=8*4*4*64*4;
    constexpr bool Double=GLM53_GATE_DOUBLE || Alias;
    struct Storage {
        alignas(16) uint64_t codebook[Double?2:1][BookWords];
        alignas(16) float partial[Alias?1:8][4][4][64];
        alignas(16) uint4 stage[(Classes>0 && GLM53_X_LDSB)?512:1];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*NT;
    // Experts whose codebooks buffer ``current`` and the other buffer hold.
    int current=0,current_expert=-1,other_expert=-1;
    const int xcd=blockIdx.x%8,rank=blockIdx.x/8,ranks=(gridDim.x-xcd+7)/8;
    const int per=(total+7)/8,tall=total%8==0?8:total%8;
    const int first=xcd<tall?xcd*per:tall*per+(xcd-tall)*(per-1);
    const int length=xcd<tall?per:per-1;
    // A CTA takes GLM53_X_CHUNK adjacent items (N tiles of one task) per
    // stride, so they share one codebook staging.
    constexpr int Chunk=GLM53_X_CHUNK;
    constexpr bool Blocked=Classes>0 && GLM53_X_BLK;
    const int run=length/ranks,extra=length%ranks;
    const int run_begin=rank*run+min(rank,extra),run_length=run+(rank<extra);
    // XCD-local index of the CTA's step-th item (>= length past its last).
    const auto local_of=[&](int s) {
        if constexpr(Blocked) return s<run_length?run_begin+s:length;
        else return ((s/Chunk)*ranks+rank)*Chunk+s%Chunk;
    };
    // The book of current_expert was DMA'd by the previous item's epilogue
    // and still needs the wait + barrier.
    bool staged=false;
    // GLM53_X_XQ: the previous item loaded this item's first quad and rows
    // and staged its book (waves 4-7, published by the item's last barrier).
    constexpr bool XQ=Classes>0 && GLM53_X_XQ && !Double && Chunk==1;
    [[maybe_unused]] bool prefetched=false;
    [[maybe_unused]] IQ2RCompressedQuad carried{};
    [[maybe_unused]] int carried_row=0;
    [[maybe_unused]] uint64_t ts_acc[4]={};[[maybe_unused]] uint32_t ts_items=0,ts_new=0;
    [[maybe_unused]] uint64_t ts0=0,ts1=0,ts2=0;
    for(int step=0;;++step)
    {
        const int local=local_of(step);
        if(local>=length) break;
        const int index=first+local;
        const int task=index/NT,n_tile=index%NT;
        const int row_begin=tasks[task*3];
        const int row_end=min(row_begin+16,row_begin+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        if constexpr(GLM53_X_GTS) { ts0=__builtin_amdgcn_s_memtime(); ++ts_items; }
        [[maybe_unused]] int next_expert=-1,next_tile=0,next_begin=0,next_end=0;
        if constexpr(XQ) {
            const int next_local=local+ranks;
            if(next_local<length) {
                const int next_task=(first+next_local)/NT;
                next_tile=(first+next_local)%NT;
                next_begin=tasks[next_task*3];
                next_end=min(next_begin+16,next_begin+tasks[next_task*3+1]);
                next_expert=tasks[next_task*3+2];
                if(next_begin>=next_end || next_begin<0 || next_end>routes || next_expert<0 || next_expert>=257) next_expert=-1;
            }
        }
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // A codebook prefetched by the previous item is complete: its DMA
        // preceded that item's last loads, and the item ended in barriers.
        if(Double && expert!=current_expert && expert==other_expert) {
            current^=1; other_expert=current_expert; current_expert=expert;
        }
        const bool new_expert=expert!=current_expert;
        [[maybe_unused]] uint4 gate_base[1];
        if(new_expert) {
            if constexpr(Classes>0 && GateBooks==2 && (GLM53_X_GG==3 || GLM53_X_NOFORM==1)) {}
            else if constexpr(Classes>0 && GateBooks==2 && GLM53_X_NOFORM==2) {
                iq2r_v2_gate_base_load<8>(gate_base,aux,wave,lane);
                asm volatile("" :: "v"(gate_base[0].x),"v"(gate_base[0].y),"v"(gate_base[0].z),"v"(gate_base[0].w));
            }
            else if constexpr(Classes>0 && GateBooks==2 && GLM53_X_NOFORM==3) {
                gate_base[0]=make_uint4(lane*0x01010101u&0x37373737u,expert,wave,lane);
                iq2r_v2_gate_form<8>(shared.codebook[current],gate_base,wave,lane);
            }
            else if constexpr(Classes>0 && GateBooks==2 && GLM53_X_GG) {
                int l=lane;
                if constexpr(GLM53_X_LDSB) asm volatile("" : "+v"(l));
                iq2r_v2_gate_base_load<8>(gate_base,aux,wave,l);
                // GG 2: form at once, so the copy-0 chunk is not live across the other loads.
                if constexpr(GLM53_X_GG==2) iq2r_v2_gate_form<8>(shared.codebook[current],gate_base,wave,l);
            }
            else iq2r_stage_codebook_async<GLM53_X_STG4?512:BookWords,8>(shared.codebook[current],GLM53_X_STG0?all_auxiliary:aux,wave,lane);
            current_expert=expert;
        }
        const int nbase=n_tile*4;
        uint32_t bases=0;
        if constexpr(Classes==0)
        {
#pragma unroll
            for(int atom=0;atom<4;++atom)
                bases|=static_cast<uint32_t>(aux[kCodebookBytes+nbase+atom])<<(atom*8);
        }
        const uint64_t* lane_book=shared.codebook[current];
        const float2 gains=iq2r_gains<Classes>(aux,auxiliary_bytes);
        // IQ2R-v2 d words of the wave's three K256 columns, 16 words apart.
        const uint32_t* d_lane=Classes>0?iq2r_v2_bases_at<Classes,GateBooks>(aux,n_tile,24,wave*3,lane_row):nullptr;
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=(XQ && prefetched)?carried_row:gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        constexpr bool ABuf=Classes>0 && GLM53_X_ABUF && !XQ;
        const int swave=ABuf?__builtin_amdgcn_readfirstlane(wave):wave;
        int base=(n_tile*48+swave*6)*kQuadBytes;
        [[maybe_unused]] opus::i32x4_t act_rsrc{},scale_rsrc{},d_rsrc{};
        [[maybe_unused]] int act_off[MAtoms]={},scale_off[MAtoms]={},d_off=0;
        if constexpr(ABuf) {
            act_rsrc=iq2r_buffer_rsrc(activations,routes/9*K);
            scale_rsrc=iq2r_buffer_rsrc(scales,routes/9*192);
            d_rsrc=iq2r_buffer_rsrc(aux,auxiliary_bytes);
#pragma unroll
            for(int m=0;m<MAtoms;++m) { act_off[m]=rows[m]*K+lane_group*16;scale_off[m]=rows[m]*192+lane_group; }
            d_off=BookWords*8+(n_tile*24*16+lane_row)*4;
        }
        [[maybe_unused]] uint32_t d_cols[3]={};
        if constexpr(ABuf && GLM53_X_DBC) {
            const uint32_t x=GLM53_X_NOD?0x70707070u:iq2r_scheduled_buffer_load1(d_rsrc,BookWords*8+min(lane,47)*4,(n_tile*24+swave*3)*64,0);
            // (x0,x0,x2,x2), (x1,x1,x3,x3); then rows 0-1 or 2-3 to all rows
            const auto pairs=__builtin_amdgcn_permlane16_swap(x,x,false,true);
            const auto even=__builtin_amdgcn_permlane32_swap(pairs[0],pairs[0],false,true);
            const auto odd=__builtin_amdgcn_permlane32_swap(pairs[1],pairs[1],false,true);
            d_cols[0]=even[0];d_cols[1]=odd[0];d_cols[2]=even[1];
        }
        uint64_t pending_book[4]={};
        IQ2RCompressedQuad pending;
        if(XQ && prefetched) pending=carried;
        else pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
        constexpr bool PF2=ABuf && GLM53_X_PF2;
        [[maybe_unused]] IQ2RCompressedQuad ahead{};
        if constexpr(PF2) ahead=iq2r_scheduled_load_quad(data,data_bytes,base+kQuadBytes,lane);
        if constexpr(Classes>0 && GateBooks==2 && GLM53_X_GG==3) {
            // GG 3: the copy-0 load goes out after the gather and first quad,
            // so the three loads share one latency before the formation.
            if(new_expert) {
                iq2r_v2_gate_base_load<8>(gate_base,aux,wave,lane);
                iq2r_v2_gate_form<8>(shared.codebook[current],gate_base,wave,lane);
            }
        }
        uint32_t d_words[3]={};
        if(Classes>0 && GLM53_X_DPRE) {
#pragma unroll
            for(int c=0;c<3;++c) d_words[c]=d_lane[c*16];
        }
        if(new_expert || staged) {
            if constexpr(Classes>0 && GateBooks==2 && GLM53_X_GG==1) iq2r_v2_gate_form<8>(shared.codebook[current],gate_base,wave,lane);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
            staged=false;
            if constexpr(GLM53_X_GTS) ++ts_new;
        }
        if constexpr(GLM53_X_GTS) { ts1=__builtin_amdgcn_s_memtime(); ts_acc[0]+=ts1-ts0; }
        [[maybe_unused]] int stage_expert=-1;
        if constexpr(Classes>0 && GateBooks==2 && GLM53_X_LDSB && !Double && !XQ) {
            const int next_local=local_of(step+1);
            const int next_expert=next_local<length?tasks[(first+next_local)/NT*3+2]:-1;
            if(next_expert!=expert && next_expert>=0 && next_expert<257) {
                stage_expert=next_expert;
                if(wave>=4) {
                    int l=lane;asm volatile("" : "+v"(l));
                    iq2r_v2_gate_base_dma<4>(shared.stage,all_auxiliary+static_cast<int64_t>(next_expert)*auxiliary_bytes,wave-4,l);
                }
            }
        }
        if constexpr(Double) {
            // Stage the next item's codebook in the other buffer, which no
            // wave reads before the next item.
            const int next_local=local_of(step+1);
            const int next_expert=next_local<length?tasks[(first+next_local)/NT*3+2]:-1;
            if(next_expert>=0 && next_expert<257 && (Alias || next_expert!=expert) && next_expert!=other_expert) {
                iq2r_stage_codebook_async<BookWords,8>(shared.codebook[current^1],
                    all_auxiliary+static_cast<int64_t>(next_expert)*auxiliary_bytes,wave,lane);
                other_expert=next_expert;
            }
        }
        [[maybe_unused]] uint32_t touch=0u;
        if constexpr(Classes>0 && GLM53_X_PF) {
            // Pull the next item's codebook into L2 while this item streams:
            // one dword per 128-byte line, the wave's 1/8 of the book.
            const int next_local=local_of(step+1);
            const int next_expert=next_local<length?tasks[(first+next_local)/NT*3+2]:-1;
            if(next_expert!=expert && next_expert>=0 && next_expert<257 && lane*128<BookWords*8/8)
                touch=*reinterpret_cast<const uint32_t*>(all_auxiliary+static_cast<int64_t>(next_expert)*auxiliary_bytes+wave*(BookWords*8/8)+lane*128);
        }
        const uint32_t book_bits=GateBooks==1?0u:iq2r_book_bits<Classes>(lane_row);
        IQ2RLevels levels=iq2r_levels<Classes>(pending.metadata,book_bits);
#if GLM53_X_UNR==6
#pragma unroll 6
#elif GLM53_X_UNR==3
#pragma unroll 3
#elif GLM53_X_UNR==2
#pragma unroll 2
#endif
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=swave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if(m>=active_m) continue;
                if constexpr(ABuf) {
                    *reinterpret_cast<opus::i32x4_t*>(a[m].bytes)=iq2r_scheduled_load_dwordx4(act_rsrc,act_off[m],kt*128,0);
                    *reinterpret_cast<opus::i32x4_t*>(a[m].bytes+16)=iq2r_scheduled_load_dwordx4(act_rsrc,act_off[m],kt*128+64,0);
                    sa[m]=iq2r_buffer_load_u8(scale_rsrc,scale_off[m],kt*4,0);
                    continue;
                }
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*((GLM53_X_ORDER || (Classes>0 && GLM53_X_OPSEL))?1u:0x01010101u);
            }
            // IQ2R-v2 d word of this K256, reloaded with the activations
            // rather than held for the whole K768 slice.
            if constexpr(ABuf && GLM53_X_DBC) bases=iteration<2?d_cols[0]:iteration<4?d_cols[1]:d_cols[2];
            else if constexpr(ABuf) bases=GLM53_X_NOD?0x70707070u:iq2r_scheduled_buffer_load1(d_rsrc,d_off,(kt/2)*64,0);
            else if constexpr(Classes>0) bases=GLM53_X_NOD?0x70707070u:GLM53_X_DPRE?(iteration<2?d_words[0]:iteration<4?d_words[1]:d_words[2]):d_lane[iteration/2*16];
            if(GLM53_X_ORDER) __builtin_amdgcn_sched_barrier(0);
            const IQ2RCompressedQuad compressed=pending;
            if constexpr(PF2) {
                // Unconditional, so the loop stays straight-line and its
                // waits exact: past the last quad the descriptor has no
                // records, and the load returns zeros without a request.
                base+=kQuadBytes;
                pending=ahead;
                ahead=iq2r_scheduled_load_quad(data,iteration<4?data_bytes:0,base+kQuadBytes,lane);
            }
            else if(iteration<5) {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            glm53_cross_k_decode_mfma_carried<MAtoms,Classes>(compressed,pending,lane_book,bases,a,sa,accumulators,active_m,pending_book,iteration==0,iteration<5,levels,book_bits);
        }
        if constexpr(GLM53_X_GTS) { ts2=__builtin_amdgcn_s_memtime(); ts_acc[1]+=ts2-ts1; }
        // GEF 2/3: load the next expert's copies 0 now, under the partial
        // stores and the first barrier; they are formed after that barrier.
        [[maybe_unused]] constexpr int FormWaves=GLM53_X_GEF==3?4:8;
        [[maybe_unused]] uint4 form_base[8/FormWaves];
        [[maybe_unused]] int form_expert=-1;
        if constexpr(Classes>0 && GateBooks==2 && GLM53_X_GEF>=2 && !Double && !XQ) {
            const int next_local=local_of(step+1);
            const int next_expert=next_local<length?tasks[(first+next_local)/NT*3+2]:-1;
            if(next_expert!=expert && next_expert>=0 && next_expert<257) {
                form_expert=next_expert;
                if(FormWaves==8 || wave>=4)
                    iq2r_v2_gate_base_load<FormWaves>(form_base,all_auxiliary+static_cast<int64_t>(next_expert)*auxiliary_bytes,wave%FormWaves,lane);
            }
        }
        if constexpr(XQ) {
            prefetched=next_expert>=0;
            if(prefetched) {
                carried=iq2r_scheduled_load_quad(all_data+static_cast<int64_t>(next_expert)*data_bytes,data_bytes,
                                                 (next_tile*48+wave*6)*kQuadBytes,lane);
                carried_row=gather[min(next_begin+lane_row,next_end-1)]/9;
            }
        }
        // A prefetch DMA must land before the barrier below publishes it.
        if constexpr(Double) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        if constexpr(Classes>0 && GLM53_X_PF) asm volatile("" :: "v"(touch));
        // Weights stay in registers, so no barrier is needed before the
        // partial stores; the store and reader barriers protect the partials.
        auto partial=reinterpret_cast<float(*)[4][4][64]>(Alias?reinterpret_cast<float*>(shared.codebook[current]):&shared.partial[0][0][0][0]);
        if constexpr(Alias) { __syncthreads(); current_expert=-1; }
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
#pragma unroll
            for(int atom=0;atom<4;++atom)
                for(int c=0;c<4;++c) partial[wave][atom][c][lane]=accumulators[m][atom][c];
            __syncthreads();
            if constexpr(XQ) {
                // Every wave is past its last book read; waves 4-7 have no
                // reduction work, so they stage the next book and wait for it.
                if(m==0 && next_expert>=0 && next_expert!=expert) {
                    if(wave>=4) {
                        iq2r_stage_codebook_async<BookWords,4>(shared.codebook[current],
                            all_auxiliary+static_cast<int64_t>(next_expert)*auxiliary_bytes,wave-4,lane);
                        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    }
                    current_expert=next_expert;
                }
            }
            if constexpr(Classes>0 && GateBooks==2 && GLM53_X_LDSB && !Double && !XQ) {
                // Every wave is past its last book read: waves 4-7, idle in
                // the reduction, form the next book from their staged chunks.
                if(m==0 && stage_expert>=0) {
                    if(wave>=4) {
                        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                        // recomputed here: hoisted addresses would stay live across items
                        int l=lane;asm volatile("" : "+v"(l));
#pragma unroll
                        for(int j=0;j<2;++j) {
                            const int c=wave-4+j*4;
                            iq2r_v2_form_chunk(reinterpret_cast<uint4*>(shared.codebook[current])+(c>>2)*1024,(c&3)*64+l,shared.stage[c*64+l]);
                        }
                    }
                    current_expert=stage_expert;
                }
            }
            if constexpr(Classes>0 && GateBooks==2 && GLM53_X_GEF>=2 && !Double && !XQ) {
                // Every wave is past its last book read.
                if(m==0 && form_expert>=0) {
                    if(FormWaves==8 || wave>=4) iq2r_v2_gate_form<FormWaves>(shared.codebook[current],form_base,wave%FormWaves,lane);
                    current_expert=form_expert;
                }
            }
            if constexpr(Classes>0 && GateBooks==2 && GLM53_X_GEF==1 && !Double && !XQ) {
                // Every wave is past its last book read. Waves 4-7 have no
                // reduction work: they form the next item's book from its
                // copies 0, published by the barrier that ends the item.
                const int next_local=local_of(step+1);
                const int next_expert=next_local<length?tasks[(first+next_local)/NT*3+2]:-1;
                if(m==0 && next_expert!=expert && next_expert>=0 && next_expert<257) {
                    if(wave>=4) {
                        uint4 next_base[2];
                        iq2r_v2_gate_base_load<4>(next_base,all_auxiliary+static_cast<int64_t>(next_expert)*auxiliary_bytes,wave-4,lane);
                        iq2r_v2_gate_form<4>(shared.codebook[current],next_base,wave-4,lane);
                    }
                    current_expert=next_expert;
                }
            }
            if constexpr(Classes>0 && GLM53_X_GEB && !Double && !XQ) {
                // Every wave is past its last book read: stage the next
                // item's book now, under the reduction and the next loads.
                const int next_local=local_of(step+1);
                const int next_expert=next_local<length?tasks[(first+next_local)/NT*3+2]:-1;
                if(m==0 && next_expert!=expert && next_expert>=0 && next_expert<257) {
                    iq2r_stage_codebook_async<BookWords,8>(shared.codebook[current],
                        all_auxiliary+static_cast<int64_t>(next_expert)*auxiliary_bytes,wave,lane);
                    current_expert=next_expert;
                    staged=true;
                }
            }
            if(wave<4)
            {
                int group=lane_group,row16=lane_row;
                if constexpr(Classes>0 && GLM53_X_LDSB) { int l=lane;asm volatile("" : "+v"(l));group=l/16;row16=l%16; }
                const int local_row=m*16+group*4+wave;
                const int output_row=row_begin+local_row;
                float values[4];
#pragma unroll
                for(int atom=0;atom<4;++atom)
                {
                    float value=0.0f;
#pragma unroll
                    for(int source=0;source<8;++source)
                        value+=partial[source][atom][wave][lane];
                    values[atom]=__bfloat162float(__float2bfloat16(value*iq2r_row_gain(gains,row16)));
                }
                // Adjacent lanes hold the interleaved gate and up columns.
                // Read the odd lane before restricting execution to even lanes.
                float up[4];
#pragma unroll
                for(int atom=0;atom<4;++atom) up[atom]=__shfl_xor(values[atom],1);
                if(row16%2==0)
                {
                    float activated[4];float abs_max=aiter::kAmaxFloor;
#pragma unroll
                    for(int atom=0;atom<4;++atom)
                    {
                        const float swish=values[atom]/(1.0f+__expf(-values[atom]));
                        activated[atom]=__bfloat162float(__float2bfloat16(swish*(up[atom]+0.0f)));
                        abs_max=fmaxf(abs_max,fabsf(activated[atom]));
                    }
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,2));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,4));
                    abs_max=fmaxf(abs_max,__shfl_xor(abs_max,8));
                    if(output_row<row_end)
                    {
                        const auto bs=fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode,MxDtype::FP8_E4M3>(abs_max);
                        const float inverse=1.0f/bs.dq_scale;
#pragma unroll
                        for(int atom=0;atom<4;++atom)
                            output[static_cast<int64_t>(output_row)*(NT*32)+n_tile*32+atom*8+row16/2]=opus::fp32_to_fp8(activated[atom]*inverse);
                        if(row16==0) output_scales[static_cast<int64_t>(output_row)*NT+n_tile]=bs.byte;
                    }
                }
            }
            __syncthreads();
        }
        if constexpr(GLM53_X_GTS) ts_acc[2]+=__builtin_amdgcn_s_memtime()-ts2;
    }
    if constexpr(GLM53_X_GTS) {
        // [block][wave]: setup, K loop, epilogue cycles, items, new-expert items
        if(lane==0) {
            uint32_t* r=reinterpret_cast<uint32_t*>(output+static_cast<int64_t>(routes)*(NT*32))+(blockIdx.x*8+wave)*8;
            r[0]=static_cast<uint32_t>(ts_acc[0]);r[1]=static_cast<uint32_t>(ts_acc[1]);r[2]=static_cast<uint32_t>(ts_acc[2]);
            r[3]=ts_items;r[4]=ts_new;r[5]=static_cast<uint32_t>(__builtin_amdgcn_s_memtime());
        }
    }
#endif
}

} // namespace

void iq2r_check_gfx950()
{
    // The IQ2R kernel bodies are compiled for gfx950 only; elsewhere they are empty.
    static const std::string arch = get_gpu_arch();
    AITER_CHECK(arch == "gfx950", "IQ2R kernels require gfx950, not ", arch);
}

void iq2r_encode_out(const aiter_tensor_t& weight,
                     const aiter_tensor_t& importance,
                     const aiter_tensor_t& codebook,
                     aiter_tensor_t& indices,
                     aiter_tensor_t& scales,
                     aiter_tensor_t& data,
                     aiter_tensor_t& auxiliary,
                     aiter_tensor_t& scale_delta_overflow,
                     int64_t valid_k,
                     int64_t exponent_radius,
                     double codebook_max)
{
    iq2r_check_gfx950();
    AITER_CHECK(weight.is_gpu() && importance.is_gpu() && codebook.is_gpu() &&
                    indices.is_gpu() && scales.is_gpu() && data.is_gpu() &&
                    auxiliary.is_gpu() && scale_delta_overflow.is_gpu(),
                "IQ2R encoder requires GPU tensors");
    const int device = weight.device_id;
    AITER_CHECK(importance.device_id == device && codebook.device_id == device &&
                    indices.device_id == device && scales.device_id == device &&
                    data.device_id == device && auxiliary.device_id == device &&
                    scale_delta_overflow.device_id == device,
                "IQ2R encoder tensors must be on the same device");
    AITER_CHECK(weight.dtype() == AITER_DTYPE_fp32 &&
                    importance.dtype() == AITER_DTYPE_fp32 &&
                    codebook.dtype() == AITER_DTYPE_fp32,
                "IQ2R encoder inputs must be float32");
    AITER_CHECK(indices.dtype() == AITER_DTYPE_i16 && scales.dtype() == AITER_DTYPE_u8 &&
                    data.dtype() == AITER_DTYPE_u8 && auxiliary.dtype() == AITER_DTYPE_u8 &&
                    scale_delta_overflow.dtype() == AITER_DTYPE_i32,
                "IQ2R encoder outputs have invalid dtypes");
    AITER_CHECK(weight.dim() == 2 && importance.dim() == 1 &&
                    importance.size(0) == weight.size(1) && codebook.dim() == 2 &&
                    codebook.size(0) == 512 && codebook.size(1) == 8,
                "IQ2R encoder expects weight [N,K], importance [K], codebook [512,8]");
    AITER_CHECK(weight.is_contiguous() && importance.is_contiguous() &&
                    codebook.is_contiguous() && indices.is_contiguous() &&
                    scales.is_contiguous() && data.is_contiguous() &&
                    auxiliary.is_contiguous() && scale_delta_overflow.is_contiguous(),
                "IQ2R encoder tensors must be contiguous");
    const int64_t N = weight.size(0);
    const int64_t storage_k = weight.size(1);
    AITER_CHECK(N > 0 && N % 16 == 0 && storage_k > 0 && storage_k % 128 == 0 &&
                    valid_k > 0 && valid_k <= storage_k && valid_k % 32 == 0,
                "IQ2R encoder requires N%16==0, storage_K%128==0, valid_K%32==0");
    AITER_CHECK(exponent_radius >= 0 && exponent_radius <= 16,
                "IQ2R exponent_radius must be in [0,16]");
    const int64_t physical_n_blocks = ((N / 16 + 5) / 6) * 6;
    const int64_t k_tiles = storage_k / 128;
    const int64_t tiles = physical_n_blocks * k_tiles;
    AITER_CHECK(indices.numel() == tiles * 64 * 4 && scales.numel() == tiles * 64,
                "IQ2R encoder scratch sizes are invalid");
    AITER_CHECK(data.numel() == (physical_n_blocks / 6) * k_tiles * kGroupBytes &&
                    auxiliary.numel() == kCodebookBytes + physical_n_blocks + 3,
                "IQ2R encoder output sizes are invalid");
    AITER_CHECK(scale_delta_overflow.numel() == 1,
                "IQ2R scale_delta_overflow must have one element");

    HipDeviceGuard device_guard(device);
    const auto stream = getCurrentHIPStream();
    const int64_t logical_blocks = N * (valid_k / 32);
    hipLaunchKernelGGL(iq2r_encode_assign_kernel,
                       dim3(static_cast<uint32_t>(logical_blocks)),
                       dim3(64),
                       0,
                       stream,
                       static_cast<const float*>(weight.data_ptr()),
                       static_cast<const float*>(importance.data_ptr()),
                       static_cast<const float*>(codebook.data_ptr()),
                       static_cast<float>(codebook_max),
                       static_cast<int>(N),
                       static_cast<int>(storage_k),
                       static_cast<int>(valid_k),
                       static_cast<int>(exponent_radius),
                       reinterpret_cast<uint16_t*>(indices.data_ptr()),
                       static_cast<uint8_t*>(data.data_ptr()),
                       static_cast<uint8_t*>(scales.data_ptr()));
    hipLaunchKernelGGL(iq2r_encode_base_kernel,
                       dim3(static_cast<uint32_t>(physical_n_blocks)),
                       dim3(256),
                       0,
                       stream,
                       static_cast<const uint8_t*>(scales.data_ptr()),
                       static_cast<int>(N),
                       static_cast<int>(storage_k),
                       static_cast<int>(valid_k),
                       static_cast<uint8_t*>(auxiliary.data_ptr()) + kCodebookBytes);
    const int64_t triplets = tiles / kAtomsPerTriplet;
    hipLaunchKernelGGL(iq2r_encode_pack_kernel,
                       dim3(static_cast<uint32_t>((triplets * 64 + 255) / 256)),
                       dim3(256),
                       0,
                       stream,
                       reinterpret_cast<const uint16_t*>(indices.data_ptr()),
                       static_cast<const uint8_t*>(scales.data_ptr()),
                       static_cast<const uint8_t*>(auxiliary.data_ptr()) + kCodebookBytes,
                       triplets,
                       static_cast<int>(N),
                       static_cast<int>(storage_k),
                       static_cast<int>(valid_k),
                       static_cast<uint8_t*>(data.data_ptr()),
                       static_cast<int32_t*>(scale_delta_overflow.data_ptr()));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_materialize_out(const aiter_tensor_t& data,
                          const aiter_tensor_t& auxiliary,
                          aiter_tensor_t& output,
                          int64_t logical_n,
                          int64_t logical_k)
{
    iq2r_check_gfx950();
    validate_weights(data, auxiliary, logical_n, logical_k);
    AITER_CHECK(output.is_gpu() && output.device_id == data.device_id,
                "IQ2R materialized output must be on the same GPU");
    AITER_CHECK(output.dtype() == AITER_DTYPE_fp32 && output.dim() == 2 &&
                    output.size(0) == logical_n && output.size(1) == logical_k,
                "IQ2R materialized output must be float32 [N,K]");
    AITER_CHECK(output.is_contiguous(), "IQ2R materialized output must be contiguous");
    const int64_t elements = logical_n * logical_k;
    constexpr int threads = 256;
    HipDeviceGuard device_guard(data.device_id);
    hipLaunchKernelGGL(iq2r_materialize_kernel,
                       dim3(static_cast<uint32_t>((elements + threads - 1) / threads)),
                       dim3(threads),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const uint8_t*>(data.data_ptr()),
                       static_cast<const uint8_t*>(auxiliary.data_ptr()),
                       static_cast<float*>(output.data_ptr()),
                       static_cast<int>(logical_n),
                       static_cast<int>(logical_k));
    HIP_CALL_LAUNCH(hipGetLastError());
}

// GLM-5.3 IQ2R packed MoE launchers. Tensor-parallel width follows from the
// packed gate stack: 16 N tiles (intermediate 512) at TP4, 8 tiles at TP8.
namespace {

constexpr int kGlm53Hidden = 6144;
constexpr int kGlm53Experts = 257;
constexpr int kGlm53TopK = 9;
// IQ2R-v2 codebook copies (aiter/ops/iq2r_v2.py level set c4e4q).
constexpr int kIQ2RV2Classes = 4;

// Packed IQ2R-v2 auxiliary bytes (iq2r_glm53_v2_aux_bytes in aiter/iq2r_glm53.py).
int64_t glm53_v2_auxiliary_bytes(int64_t n, int64_t k, int books, int classes)
{
    const int64_t atoms = books == 2 ? 4 : 3;
    return books * classes * kCodebookBytes + n / (16 * atoms) * (k / 256) * 64 + 8;
}

// 0 for IQ2R auxiliary bytes, kIQ2RV2Classes for IQ2R-v2, -1 otherwise.
int glm53_classes(const aiter_tensor_t& auxiliary, int64_t n, int64_t k, int books)
{
    if(auxiliary.dim() != 2)
        return -1;
    if(auxiliary.size(1) == expected_auxiliary_bytes(n))
        return 0;
    if(auxiliary.size(1) == glm53_v2_auxiliary_bytes(n, k, books, kIQ2RV2Classes))
        return kIQ2RV2Classes;
    return -1;
}

int glm53_tp4_from_gate(const aiter_tensor_t& data, const aiter_tensor_t& auxiliary, int* classes = nullptr)
{
    const bool tp4 = data.dim() == 2 && data.size(1) == 16 * 48 * kQuadBytes;
    AITER_CHECK(data.dtype() == AITER_DTYPE_u8 && data.dim() == 2 && data.size(0) == kGlm53Experts &&
                    (tp4 || data.size(1) == 8 * 48 * kQuadBytes),
                "GLM-5.3 IQ2R gate requires the packed quad stack [257, N/32*48*2304]");
    const int format = glm53_classes(auxiliary, tp4 ? 1024 : 512, kGlm53Hidden, 2);
    AITER_CHECK(auxiliary.dtype() == AITER_DTYPE_u8 && auxiliary.dim() == 2 &&
                    auxiliary.size(0) == kGlm53Experts && format >= 0,
                "GLM-5.3 IQ2R gate requires canonical IQ2R or packed IQ2R-v2 auxiliary bytes");
    if(classes)
        *classes = format;
    return tp4;
}

// Calls launch(std::integral_constant<int, C>{}) for the auxiliary's format.
template<typename Launch>
void glm53_with_classes(int classes, Launch&& launch)
{
    if(classes == 0)
        launch(std::integral_constant<int, 0>{});
    else
        launch(std::integral_constant<int, kIQ2RV2Classes>{});
}

void glm53_check_device(std::initializer_list<const aiter_tensor_t*> tensors, const aiter_tensor_t& data)
{
    iq2r_check_gfx950();
    AITER_CHECK(data.is_gpu() && data.is_contiguous(),
                "GLM-5.3 IQ2R packed weights must be contiguous on the GPU");
    for(const auto* tensor : tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id == data.device_id && tensor->is_contiguous(),
                    "GLM-5.3 IQ2R tensors must be contiguous on the weight GPU");
}

void glm53_check_gate_io(const aiter_tensor_t& activations,
                         const aiter_tensor_t& scales,
                         const aiter_tensor_t& output,
                         const aiter_tensor_t& output_scales,
                         int rows,
                         int routes,
                         int intermediate)
{
    AITER_CHECK(activations.dtype() == AITER_DTYPE_fp8 && activations.dim() == 2 &&
                    activations.size(0) == rows && activations.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R gate input must be FP8 [rows, 6144]");
    AITER_CHECK(scales.dtype() == AITER_DTYPE_u8 && scales.dim() == 2 && scales.size(0) == rows &&
                    scales.size(1) == kGlm53Hidden / kScaleBlock,
                "GLM-5.3 IQ2R gate input scales must be uint8 [rows, 192]");
    AITER_CHECK(output.dtype() == AITER_DTYPE_fp8 && output.dim() == 2 && output.size(0) == routes &&
                    output.size(1) == intermediate && output_scales.dtype() == AITER_DTYPE_u8 &&
                    output_scales.dim() == 2 && output_scales.size(0) == routes &&
                    output_scales.size(1) == intermediate / kScaleBlock,
                "GLM-5.3 IQ2R gate output must be FP8 [routes, I] with uint8 scales [routes, I/32]");
}

void glm53_check_tasks(const aiter_tensor_t& tasks, const aiter_tensor_t& task_count, int64_t capacity)
{
    AITER_CHECK(tasks.dtype() == AITER_DTYPE_i32 && tasks.dim() == 2 && tasks.size(1) == 3 &&
                    tasks.size(0) >= capacity,
                "GLM-5.3 IQ2R tasks must be int32 [capacity, 3]");
    AITER_CHECK(task_count.dtype() == AITER_DTYPE_i32 && task_count.dim() == 1 && task_count.size(0) == 1,
                "GLM-5.3 IQ2R task count must be int32 [1]");
}

int glm53_down_intermediate(const aiter_tensor_t& activations,
                            const aiter_tensor_t& scales,
                            const aiter_tensor_t& data,
                            const aiter_tensor_t& auxiliary,
                            int* classes)
{
    const int intermediate = static_cast<int>(activations.size(1));
    AITER_CHECK(intermediate == 256 || intermediate == 512, "GLM-5.3 IQ2R down K must be 256 or 512");
    *classes = glm53_classes(auxiliary, kGlm53Hidden, intermediate, 1);
    AITER_CHECK(*classes >= 0, "GLM-5.3 IQ2R down requires canonical IQ2R or packed IQ2R-v2 auxiliary bytes");
    if(*classes == 0)
        validate_weights(data, auxiliary, kGlm53Hidden, intermediate);
    else
        AITER_CHECK(data.dtype() == AITER_DTYPE_u8 && data.dim() == 2 &&
                        data.size(1) == expected_data_bytes(kGlm53Hidden, intermediate) &&
                        auxiliary.dtype() == AITER_DTYPE_u8 && auxiliary.size(0) == data.size(0),
                    "GLM-5.3 IQ2R-v2 down data byte count does not match K");
    AITER_CHECK(data.size(0) == kGlm53Experts && activations.dtype() == AITER_DTYPE_fp8 &&
                    activations.dim() == 2 && activations.size(0) % kGlm53TopK == 0,
                "GLM-5.3 IQ2R down input must be FP8 [tokens*9, K]");
    AITER_CHECK(scales.dtype() == AITER_DTYPE_u8 && scales.dim() == 2 &&
                    scales.size(0) == activations.size(0) && scales.size(1) == intermediate / kScaleBlock,
                "GLM-5.3 IQ2R down scales must be uint8 [routes, K/32]");
    return intermediate;
}

} // namespace

void iq2r_glm53_gate_m1_out(const aiter_tensor_t& activations,
                            const aiter_tensor_t& scales,
                            const aiter_tensor_t& data,
                            const aiter_tensor_t& auxiliary,
                            const aiter_tensor_t& tasks,
                            const aiter_tensor_t& task_count,
                            aiter_tensor_t& output,
                            aiter_tensor_t& output_scales)
{
    int classes = 0;
    const bool tp4 = glm53_tp4_from_gate(data, auxiliary, &classes);
    glm53_check_device({&activations, &scales, &auxiliary, &tasks, &task_count, &output, &output_scales}, data);
    const int routes = kGlm53TopK;
    glm53_check_gate_io(activations, scales, output, output_scales, routes, routes, tp4 ? 512 : 256);
    glm53_check_tasks(tasks, task_count, routes);
    HipDeviceGuard device_guard(data.device_id);
    glm53_with_classes(classes, [&](auto tag) {
    constexpr int C = decltype(tag)::value;
    hipLaunchKernelGGL((tp4 ? glm53_gate_m1_kernel<16, C> : glm53_gate_m1_kernel<8, C>),
                       dim3(2 * static_cast<int>(get_num_cu_func())),
                       dim3(64, 8),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const opus::fp8_t*>(activations.data_ptr()),
                       static_cast<const uint8_t*>(scales.data_ptr()),
                       static_cast<const uint8_t*>(data.data_ptr()),
                       static_cast<const uint8_t*>(auxiliary.data_ptr()),
                       static_cast<const int32_t*>(tasks.data_ptr()),
                       static_cast<const int32_t*>(task_count.data_ptr()),
                       static_cast<opus::fp8_t*>(output.data_ptr()),
                       static_cast<uint8_t*>(output_scales.data_ptr()),
                       routes,
                       static_cast<int>(data.size(1)),
                       static_cast<int>(auxiliary.size(1)));
    });
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_gate_out(const aiter_tensor_t& activations,
                         const aiter_tensor_t& scales,
                         const aiter_tensor_t& data,
                         const aiter_tensor_t& auxiliary,
                         const aiter_tensor_t& tasks,
                         const aiter_tensor_t& task_count,
                         const aiter_tensor_t& gather,
                         aiter_tensor_t& output,
                         aiter_tensor_t& output_scales,
                         int64_t kernel,
                         int64_t grid_multiplier)
{
    int classes = 0;
    const bool tp4 = glm53_tp4_from_gate(data, auxiliary, &classes);
    glm53_check_device(
        {&activations, &scales, &auxiliary, &tasks, &task_count, &gather, &output, &output_scales}, data);
    const int tokens = static_cast<int>(activations.size(0));
    const int routes = tokens * kGlm53TopK;
    glm53_check_gate_io(activations, scales, output, output_scales, tokens, routes, tp4 ? 512 : 256);
    AITER_CHECK(gather.dtype() == AITER_DTYPE_i32 && gather.dim() == 1 && gather.size(0) == routes,
                "GLM-5.3 IQ2R gate gather must be int32 [routes]");
    AITER_CHECK(grid_multiplier >= 1 && grid_multiplier <= 8, "GLM-5.3 IQ2R gate grid multiplier");
    const bool prefill = kernel == kGlm53GatePrefill;
    AITER_CHECK(kernel == kGlm53GateDecode || prefill || kernel == kGlm53GateDecodeNoBarrier,
                "GLM-5.3 IQ2R gate kernel id");
    AITER_CHECK(tokens >= 2 && tokens <= (prefill ? 4096 : 1024),
                "GLM-5.3 IQ2R gate token range: decode 2..1024, prefill 2..4096");
    const int rows = prefill ? 64 : 16;
    glm53_check_tasks(tasks, task_count, (routes + rows - 1) / rows + std::min(routes, 258));
    HipDeviceGuard device_guard(data.device_id);
    using gate_kernel = void (*)(const opus::fp8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
                                 const int32_t*, const int32_t*, const int32_t*, opus::fp8_t*, uint8_t*,
                                 int, int, int);
    gate_kernel launch;
    int waves = 8;
    glm53_with_classes(classes, [&](auto tag) {
        constexpr int C = decltype(tag)::value;
        if(prefill)
        {
            launch = tp4 ? glm53_gate_prefill_kernel<4, 8, 4, 2, 16, C> : glm53_gate_prefill_kernel<4, 8, 4, 2, 8, C>;
            waves  = 4;
        }
        else if(kernel == kGlm53GateDecodeNoBarrier)
            launch = tp4 ? glm53_gate_decode_nobarrier_kernel<16, C> : glm53_gate_decode_nobarrier_kernel<8, C>;
        else
            launch = tp4 ? glm53_gate_decode_kernel<true, 16, C> : glm53_gate_decode_kernel<false, 8, C>;
    });
    hipLaunchKernelGGL(launch,
                       dim3(static_cast<int>(grid_multiplier) * static_cast<int>(get_num_cu_func())),
                       dim3(64, waves),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const opus::fp8_t*>(activations.data_ptr()),
                       static_cast<const uint8_t*>(scales.data_ptr()),
                       static_cast<const uint8_t*>(data.data_ptr()),
                       static_cast<const uint8_t*>(auxiliary.data_ptr()),
                       static_cast<const int32_t*>(tasks.data_ptr()),
                       static_cast<const int32_t*>(task_count.data_ptr()),
                       static_cast<const int32_t*>(gather.data_ptr()),
                       static_cast<opus::fp8_t*>(output.data_ptr()),
                       static_cast<uint8_t*>(output_scales.data_ptr()),
                       routes,
                       static_cast<int>(data.size(1)),
                       static_cast<int>(auxiliary.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_down_out(const aiter_tensor_t& activations,
                         const aiter_tensor_t& scales,
                         const aiter_tensor_t& data,
                         const aiter_tensor_t& auxiliary,
                         const aiter_tensor_t& tasks,
                         const aiter_tensor_t& task_count,
                         aiter_tensor_t& output,
                         int64_t kernel,
                         int64_t grid_multiplier)
{
    int classes = 0;
    const int K = glm53_down_intermediate(activations, scales, data, auxiliary, &classes);
    const bool tp4 = K == 512;
    glm53_check_device({&activations, &scales, &auxiliary, &tasks, &task_count, &output}, data);
    const int routes = static_cast<int>(activations.size(0));
    AITER_CHECK(routes >= 2 * kGlm53TopK && routes <= 1024 * kGlm53TopK, "GLM-5.3 IQ2R decode down covers M2..1024");
    glm53_check_tasks(tasks, task_count, (routes + 31) / 32 + std::min(routes, 258));
    AITER_CHECK(output.dtype() == AITER_DTYPE_bf16 && output.dim() == 2 && output.size(0) == routes &&
                    output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R down output must be BF16 [routes, 6144]");
    AITER_CHECK(kernel == kGlm53DownPacked || kernel == kGlm53DownSingle || (kernel == kGlm53DownOrdered && !tp4), "GLM-5.3 IQ2R down kernel id");
    AITER_CHECK(grid_multiplier >= 1 && grid_multiplier <= 16, "GLM-5.3 IQ2R down grid multiplier");
    HipDeviceGuard device_guard(data.device_id);
    const dim3 grid(static_cast<int>(grid_multiplier) * static_cast<int>(get_num_cu_func()));
#define GLM53_DOWN_ARGS                                                                     \
    static_cast<const opus::fp8_t*>(activations.data_ptr()),                                \
        static_cast<const uint8_t*>(scales.data_ptr()),                                     \
        static_cast<const uint8_t*>(data.data_ptr()),                                       \
        static_cast<const uint8_t*>(auxiliary.data_ptr()),                                  \
        static_cast<const int32_t*>(tasks.data_ptr()),                                      \
        static_cast<const int32_t*>(task_count.data_ptr()),                                 \
        static_cast<__hip_bfloat16*>(output.data_ptr()), routes, kGlm53Experts,             \
        static_cast<int>(data.size(1)), static_cast<int>(auxiliary.size(1))
    // Single: one 16-row atom per pass for decode-sized tasks, three workgroups per CU.
    glm53_with_classes(classes, [&](auto tag) {
    constexpr int C = decltype(tag)::value;
    if(tp4 && kernel == kGlm53DownSingle)
        hipLaunchKernelGGL((glm53_down_tp4_kernel<4, 1, 2, 8, 4, 4, (C > 0 ? GLM53_X_S8OCC : 3), C>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS);
    else if(kernel == kGlm53DownSingle)
        hipLaunchKernelGGL((glm53_down_tp8_ordered_kernel<4, 1, 2, 8, 4, C>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS);
    else if(tp4)
        hipLaunchKernelGGL((glm53_down_tp4_kernel<4, 2, 2, 8, 4, 4, 1, C>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS);
    else if(kernel == kGlm53DownOrdered)
        hipLaunchKernelGGL((glm53_down_tp8_ordered_kernel<4, 2, 2, 8, 4, C>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS);
    else
        hipLaunchKernelGGL((glm53_down_prefill_tp8_kernel<4, 2, 2, 8, 4, 1, false, C>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS, 0);
    });
#undef GLM53_DOWN_ARGS
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_down_route9_out(const aiter_tensor_t& activations,
                                const aiter_tensor_t& scales,
                                const aiter_tensor_t& data,
                                const aiter_tensor_t& auxiliary,
                                const aiter_tensor_t& expert_ids,
                                const aiter_tensor_t& scatter,
                                const aiter_tensor_t& route_weights,
                                aiter_tensor_t& output)
{
    int classes = 0;
    const int K = glm53_down_intermediate(activations, scales, data, auxiliary, &classes);
    glm53_check_device({&activations, &scales, &auxiliary, &expert_ids, &scatter, &route_weights, &output}, data);
    const int routes = static_cast<int>(activations.size(0));
    const int tokens = routes / kGlm53TopK;
    AITER_CHECK(tokens == 1 || tokens == 2 || tokens == 4, "GLM-5.3 IQ2R route9 down covers M1, M2 and M4");
    AITER_CHECK(expert_ids.dtype() == AITER_DTYPE_i32 && expert_ids.numel() == routes &&
                    scatter.dtype() == AITER_DTYPE_i32 && scatter.numel() == routes,
                "GLM-5.3 IQ2R route9 expert ids and scatter must be int32 [tokens, 9]");
    AITER_CHECK(route_weights.dtype() == AITER_DTYPE_fp32 && route_weights.numel() == routes,
                "GLM-5.3 IQ2R route9 weights must be float32 [tokens, 9]");
    AITER_CHECK(output.dtype() == AITER_DTYPE_bf16 && output.dim() == 2 && output.size(0) == tokens &&
                    output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R route9 output must be BF16 [tokens, 6144]");
    HipDeviceGuard device_guard(data.device_id);
    glm53_with_classes(classes, [&](auto tag) {
    constexpr int C = decltype(tag)::value;
    const int triplets = tokens >= GLM53_X_R9T ? 2 : 1;
    if constexpr(C > 0 && GLM53_X_R12)
    {
        hipLaunchKernelGGL((K == 512 ? (triplets == 2 ? glm53_down_route12_kernel<4, C, 2> : glm53_down_route12_kernel<4, C, 1>)
                                     : (triplets == 2 ? glm53_down_route12_kernel<2, C, 2> : glm53_down_route12_kernel<2, C, 1>)),
                           dim3(128 / triplets, tokens),
                           dim3(64, 12),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const opus::fp8_t*>(activations.data_ptr()),
                           static_cast<const uint8_t*>(scales.data_ptr()),
                           static_cast<const uint8_t*>(data.data_ptr()),
                           static_cast<const uint8_t*>(auxiliary.data_ptr()),
                           static_cast<const int32_t*>(expert_ids.data_ptr()),
                           static_cast<const int32_t*>(scatter.data_ptr()),
                           static_cast<const float*>(route_weights.data_ptr()),
                           static_cast<__hip_bfloat16*>(output.data_ptr()),
                           static_cast<int>(data.size(1)),
                           static_cast<int>(auxiliary.size(1)));
        return;
    }
    hipLaunchKernelGGL((K == 512 ? (triplets == 2 ? glm53_down_route9_kernel<4, C, 2> : glm53_down_route9_kernel<4, C>)
                                 : (triplets == 2 ? glm53_down_route9_kernel<2, C, 2> : glm53_down_route9_kernel<2, C>)),
                       dim3(128 / triplets, tokens),
                       dim3(64, (GLM53_X_R9IDLE || (C > 0 && GLM53_X_R9HELP)) ? 12 : (C > 0 && GLM53_X_R9S8) ? 8 : 9),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const opus::fp8_t*>(activations.data_ptr()),
                       static_cast<const uint8_t*>(scales.data_ptr()),
                       static_cast<const uint8_t*>(data.data_ptr()),
                       static_cast<const uint8_t*>(auxiliary.data_ptr()),
                       static_cast<const int32_t*>(expert_ids.data_ptr()),
                       static_cast<const int32_t*>(scatter.data_ptr()),
                       static_cast<const float*>(route_weights.data_ptr()),
                       static_cast<__hip_bfloat16*>(output.data_ptr()),
                       static_cast<int>(data.size(1)),
                       static_cast<int>(auxiliary.size(1)));
    });
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_down_reduce_out(const aiter_tensor_t& activations,
                                const aiter_tensor_t& scales,
                                const aiter_tensor_t& data,
                                const aiter_tensor_t& auxiliary,
                                const aiter_tensor_t& tasks,
                                const aiter_tensor_t& task_count,
                                aiter_tensor_t& route_output,
                                const aiter_tensor_t& scatter,
                                const aiter_tensor_t& route_weights,
                                aiter_tensor_t& output,
                                int64_t chunks)
{
    int classes = 0;
    const int K = glm53_down_intermediate(activations, scales, data, auxiliary, &classes);
    glm53_check_device(
        {&activations, &scales, &auxiliary, &tasks, &task_count, &route_output, &scatter, &route_weights, &output},
        data);
    const int routes = static_cast<int>(activations.size(0));
    const int tokens = routes / kGlm53TopK;
    AITER_CHECK(tokens >= 2 && tokens <= 4096, "GLM-5.3 IQ2R prefill down covers M2..4096");
    glm53_check_tasks(tasks, task_count, (routes + 63) / 64 + std::min(routes, 258));
    AITER_CHECK(route_output.dtype() == AITER_DTYPE_bf16 && route_output.dim() == 2 &&
                    route_output.size(0) == routes && route_output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R prefill route output must be BF16 [routes, 6144]");
    AITER_CHECK(scatter.dtype() == AITER_DTYPE_i32 && scatter.numel() >= routes &&
                    route_weights.dtype() == AITER_DTYPE_fp32 && route_weights.numel() == routes,
                "GLM-5.3 IQ2R prefill scatter and weights");
    AITER_CHECK(output.dtype() == AITER_DTYPE_bf16 && output.dim() == 2 && output.size(0) == tokens &&
                    output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R prefill output must be BF16 [tokens, 6144]");
    AITER_CHECK(chunks == 1 || chunks == 2, "GLM-5.3 IQ2R prefill down splits N into 1 or 2 chunks");
    HipDeviceGuard device_guard(data.device_id);
    const int cus = static_cast<int>(get_num_cu_func());
    for(int chunk = 0; chunk < chunks; ++chunk)
    {
#define GLM53_PREFILL_DOWN(KERNEL, GRID, ...)                                                   \
    hipLaunchKernelGGL((KERNEL), dim3((GRID) * cus), dim3(64, 4), 0, getCurrentHIPStream(),      \
                       static_cast<const opus::fp8_t*>(activations.data_ptr()),                \
                       static_cast<const uint8_t*>(scales.data_ptr()),                         \
                       static_cast<const uint8_t*>(data.data_ptr()),                           \
                       static_cast<const uint8_t*>(auxiliary.data_ptr()),                      \
                       static_cast<const int32_t*>(tasks.data_ptr()),                          \
                       static_cast<const int32_t*>(task_count.data_ptr()), __VA_ARGS__,        \
                       static_cast<int>(data.size(1)), static_cast<int>(auxiliary.size(1)), chunk)
        // TP4: 16 persistent CTAs per CU with weight-stationary K loops; TP8:
        // N-wave tiles sharing one cached activation tile.
        glm53_with_classes(classes, [&](auto tag) {
        constexpr int C = decltype(tag)::value;
        if(K == 512 && chunks == 1)
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp4_kernel<4, 2, 8, 4, 4, 1, 3, C>), 16,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, kGlm53Experts);
        else if(K == 512)
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp4_kernel<4, 2, 8, 4, 4, 2, 3, C>), 16,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, kGlm53Experts);
        else if(chunks == 1)
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp8_kernel<4, 2, 2, 8, 4, 1, true, C>), 8,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, kGlm53Experts);
        else
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp8_kernel<4, 2, 2, 8, 4, 2, true, C>), 8,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, kGlm53Experts);
        });
#undef GLM53_PREFILL_DOWN
        const int width = kGlm53Hidden / static_cast<int>(chunks);
        hipLaunchKernelGGL((chunks == 1 ? glm53_reduce_chunk_kernel<1> : glm53_reduce_chunk_kernel<2>),
                           dim3(tokens * ((width + 1023) / 1024)),
                           dim3(128),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const __hip_bfloat16*>(route_output.data_ptr()),
                           static_cast<const float*>(route_weights.data_ptr()),
                           static_cast<const int32_t*>(scatter.data_ptr()),
                           static_cast<__hip_bfloat16*>(output.data_ptr()),
                           tokens,
                           chunk);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace aiter
