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
constexpr int kAtomsPerTriplet     = 3;
constexpr int kNBlocksPerGroup     = 6;
constexpr int kAtomTwoRecordBytes  = 12;
constexpr int kAtomTwoOffset       = 1024;
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

__device__ __forceinline__ uint32_t load_u32(const uint8_t* pointer)
{
    return *reinterpret_cast<const uint32_t*>(pointer);
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

__device__ __forceinline__ IQ2RCompressedTriplet
iq2r_load_compact_triplet(const uint8_t* data, int source_base, int lane)
{
    IQ2RCompressedTriplet result;
    const uint8_t* third = data + source_base + kAtomTwoOffset +
                           lane * kAtomTwoRecordBytes;
    result.paired = *reinterpret_cast<const uint4*>(
        data + source_base + lane * 16);
    result.third.x = load_u32(third);
    result.third.y = load_u32(third + 4);
    result.third.z = load_u32(third + 8);
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

// GLM-5.3 IQ2R-v2 packed MoE kernels.

constexpr int kQuadBytes=2304;

struct IQ2RCompressedQuad { uint4 first; uint4 second; uint32_t metadata; };

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

// ``Policy`` is the loads' cache-policy operand: 2 (nt) for the decode gates,
// whose records are read once; 0 elsewhere (prefill reuses them).
template<int Policy=0>
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
    out.first=__builtin_bit_cast(uint4,iq2r_scheduled_load_dwordx4(rsrc,lane*16,sb,Policy));
    out.second=__builtin_bit_cast(uint4,iq2r_scheduled_load_dwordx4(rsrc,lane*16+1024,sb,Policy));
    out.metadata=iq2r_scheduled_buffer_load1(rsrc,lane*4+2048,sb,Policy);
    asm volatile("" ::: "memory");
    return out;
}

// IQ2R-v2 (aiter/ops/iq2r_v2.py, iq2r_glm53_pack_v2) with Classes codebook
// copies: ``bases`` holds the lane's own E8M0 d bytes for the
// current K256 (one per atom), and the high nibble of lane group g is
// exponent(block g) * Classes + class(block s(g)), s = (0, 2, 1, 3). Gate and
// up rows share one codebook. The MFMA
// scale is d + exponent; codewords come from codebook copy ``class`` (copies
// 4 KB apart in LDS). A data lane's codewords 0-1 belong to block g/2 and 2-3
// to block g/2 + 2: with s, those classes sit in lane groups 2*(g/2) and
// 2*(g/2) + 1, which one permlane16 swap of the metadata word delivers.
struct IQ2RLevels { uint32_t low, high; };

template<int Classes>
constexpr int iq2r_log2_classes()
{
    static_assert(Classes==4, "IQ2R-v2 c4e4q");
    return 2;
}

// Bytes of class << 4 per atom, for the blocks of a data lane's codewords 0-1
// (low) and 2-3 (high), OR-ed with ``or_bits`` (route9 carries its route's
// copy offset there). Shifted left by 8 such a byte is the byte offset of the
// codebook copy, see iq2r_copy_offset.
template<int Classes>
__device__ __forceinline__ IQ2RLevels iq2r_levels(uint32_t metadata,uint32_t or_bits=0u)
{
    constexpr uint32_t mask=static_cast<uint32_t>(Classes-1)*0x10101010u;
    // permlane16_swap(x, x) -> rows (x0,x0,x2,x2), (x1,x1,x3,x3). Masking
    // first is one mask for both rows; or_bits may depend on lane % 16
    // only, which the swap keeps.
    const uint32_t bits=(metadata&mask)|or_bits;
    const auto rows=__builtin_amdgcn_permlane16_swap(bits,bits,false,true);
    return {static_cast<uint32_t>(rows[0]),static_cast<uint32_t>(rows[1])};
}

// Byte offset of the codebook copy for ``atom``: one v_perm moves the atom's
// byte to byte 1 (x 256) and zeroes the rest.
template<int Classes>
__device__ __forceinline__ uint32_t iq2r_copy_offset(uint32_t levels,int atom,uint32_t high=0u)
{
    return __builtin_amdgcn_perm(high,levels,0x0c04000cu|(static_cast<uint32_t>(atom)<<8));
}

// E8M0 weight scales of the lane's own 32-blocks, one byte per atom: d +
// exponent, added byte-wise for all atoms at once (no byte overflows:
// d <= 254 - exponent).
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
    return (iq2r_scale_bytes<Classes>(bases,metadata)>>(atom*8))&255u;
}

// Codebook words a packed auxiliary holds: Classes copies of the 512-entry book.
template<int Classes>
constexpr int iq2r_codebook_words() { return Classes*512; }

// Stage an expert's Classes codebook copies in LDS with
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

// Codebook copies 1-3 from copy 0, the codebook itself. For a normal
// E4M3 byte b in 0x08..0x78, copy c's byte E4M3(b * 2^(c/4)) (the packer's
// FP32 product rounded to nearest even) is b + delta_c[b & 7], and no byte
// carries into the next: copy 1 adds 2, copies 2 and 3 add a v_perm table
// entry (k2, k3: deltas for mantissas 0-3, 4-7 are the high word's). The
// encoder keeps every codebook byte in that range (aiter/ops/iq2r_v2.py).
__device__ __forceinline__ void iq2r_v2_copy_words(uint32_t w,uint32_t k2,uint32_t k3,
                                                   uint32_t& r1,uint32_t& r2,uint32_t& r3)
{
    const uint32_t m=w&0x07070707u;
    r1=w+0x02020202u;                                        // x 2^(1/4)
    r2=w+__builtin_amdgcn_perm(0x04040404u,k2,m);            // x 2^(2/4)
    r3=w+__builtin_amdgcn_perm(0x06060606u,k3,m);            // x 2^(3/4)
}

// The 4 copies of a 4 KB book in LDS from copy 0; base[i] holds bytes
// (i*64+lane)*16 of copy 0. Route9 forms one book per wave.
template<int Classes>
__device__ __forceinline__ void iq2r_v2_form_copies(uint64_t* lds,const uint4 (&base)[4],int lane)
{
    static_assert(Classes==4,"c4e4q multipliers 2^(c/4)");
    uint4* out=reinterpret_cast<uint4*>(lds);
#pragma unroll
    for(int i=0;i<4;++i)
    {
        // keeps the formation after the caller's later loads are issued
        uint32_t w[4]={base[i].x,base[i].y,base[i].z,base[i].w};
        asm volatile("" : "+v"(w[0]),"+v"(w[1]),"+v"(w[2]),"+v"(w[3]));
        out[i*64+lane]=make_uint4(w[0],w[1],w[2],w[3]);
        uint32_t r1[4],r2[4],r3[4];
#pragma unroll
        for(int d=0;d<4;++d) iq2r_v2_copy_words(w[d],0x05040403u,0x06060605u,r1[d],r2[d],r3[d]);
        out[256+i*64+lane]=make_uint4(r1[0],r1[1],r1[2],r1[3]);
        out[512+i*64+lane]=make_uint4(r2[0],r2[1],r2[2],r2[3]);
        out[768+i*64+lane]=make_uint4(r3[0],r3[1],r3[2],r3[3]);
        __builtin_amdgcn_sched_barrier(0);
    }
}

// Book copies in LDS from copy 0 by a workgroup of Threads (256 or 512)
// threads, instead of a 16 KB copy stage: each thread loads 4096 / Threads
// bytes of copy 0 (iq2r_v2_book_base_load) and forms their four copies
// (iq2r_v2_book_form), which a caller may delay until its other loads are
// issued.
template<int Threads>
using IQ2RV2BookBase=std::conditional_t<Threads==256,uint4,uint2>;

template<int Threads>
__device__ __forceinline__ IQ2RV2BookBase<Threads> iq2r_v2_book_base_load(const uint8_t* aux,int thread)
{
    static_assert(Threads==256 || Threads==512);
    const IQ2RV2BookBase<Threads> base=reinterpret_cast<const IQ2RV2BookBase<Threads>*>(aux)[thread];
    asm volatile("" ::: "memory");
    return base;
}

template<int Threads>
__device__ __forceinline__ void iq2r_v2_book_form(uint64_t* lds,IQ2RV2BookBase<Threads> base,int thread)
{
    using Vector=IQ2RV2BookBase<Threads>;
    constexpr int Words=sizeof(Vector)/4;
    // materialized here rather than hoisted into VGPRs held by the caller's loops
    uint32_t k2=0x05040403u,k3=0x06060605u;
    asm volatile("" : "+v"(k2),"+v"(k3));
    uint32_t w[Words],copies[3][Words];
    __builtin_memcpy(w,&base,sizeof(base));
#pragma unroll
    for(int d=0;d<Words;++d) iq2r_v2_copy_words(w[d],k2,k3,copies[0][d],copies[1][d],copies[2][d]);
    Vector* book=reinterpret_cast<Vector*>(lds);
    book[thread]=base;
#pragma unroll
    for(int c=1;c<4;++c) __builtin_memcpy(book+c*Threads+thread,copies[c-1],sizeof(Vector));
}

// Output gains (even rows, odd rows): the auxiliary's last 8 bytes.
template<int Classes>
__device__ __forceinline__ float iq2r_gain(const uint8_t* aux,int auxiliary_bytes,int row)
{
    return reinterpret_cast<const float*>(aux+auxiliary_bytes-8)[row&1];
}

// Both gains of a workgroup-uniform expert: a scalar load, so they stay in
// SGPRs until the epilogue selects the lane's row (iq2r_row_gain).
template<int Classes>
__device__ __forceinline__ float2 iq2r_gains(const uint8_t* aux,int auxiliary_bytes)
{
    return *reinterpret_cast<const float2*>(aux+auxiliary_bytes-8);
}

__device__ __forceinline__ float iq2r_row_gain(float2 gains,int row) { return (row&1)?gains.y:gains.x; }

// d word of one lane: the table follows the codebook copies,
// [blocks][K/256][16][4] bytes.
template<int Classes>
__device__ __forceinline__ const uint32_t* iq2r_v2_bases_at(const uint8_t* aux,int block,int columns,int column,int lane_row)
{
    return reinterpret_cast<const uint32_t*>(aux+iq2r_codebook_words<Classes>()*8)
        +(block*columns+column)*16+lane_row;
}

template<int Classes>
__device__ __forceinline__ uint32_t iq2r_v2_bases(const uint8_t* aux,int block,int columns,int column,int lane_row)
{
    return *iq2r_v2_bases_at<Classes>(aux,block,columns,column,lane_row);
}

// Packed gate records: 9-bit packed indices, sign bit planes and sign-free
// codebook copies (see iq2r_glm53_pack_v2 in aiter/iq2r_glm53.py).
template<int MAtoms,int Classes>
__device__ __forceinline__ void glm53_quad_decode_mfma(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m);

template<int NT,int Classes>
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
    constexpr int BookWords=iq2r_codebook_words<Classes>();
    struct Storage {
        alignas(16) uint64_t codebook[BookWords];
        alignas(16) opus::vector_t<float,4> partial[8][4][64];
    };
    __shared__ Storage shared;
    constexpr int Policy=2;  // non-temporal: see iq2r_scheduled_load_quad
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
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=256) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // Previous work ends with a barrier protecting the codebook and partials.
        // Copy 0 of the book loads with the scales, gain and first weights, and
        // the other copies are formed below.
        const uint2 gate_base=iq2r_v2_book_base_load<512>(aux,wave*64+lane);
        // d words, one per K256.
        uint32_t d_words[3]={};
#pragma unroll
        for(int c=0;c<3;++c) d_words[c]=iq2r_v2_bases<Classes>(aux,n_tile,24,wave*3+c,lane_row);
        const float gain=iq2r_gain<Classes>(aux,auxiliary_bytes,lane_row);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=min(row_begin+m*16+lane_row,row_end-1);
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        int base=(n_tile*48+wave*6)*kQuadBytes;
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad<Policy>(data,data_bytes,base,lane);
        iq2r_v2_book_form<512>(shared.codebook,gate_base,wave*64+lane);
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
                pending=iq2r_scheduled_load_quad<Policy>(data,data_bytes,base,lane);
            }
            glm53_quad_decode_mfma<MAtoms,Classes>(compressed,shared.codebook,d_words[iteration/2],a,sa,accumulators,MAtoms);
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

// Prefill gate: waves cover N, while a shared K128 activation tile is reused
// across the CTA. K is summed as eight K768 partials, like the decode gate.
// Two waves per SIMD: left free, the compiler would use one.
template<int Waves,int XCD,int TaskGroup,int Unroll,int NT,int Classes>
__global__ __launch_bounds__(64*Waves,2) void glm53_gate_prefill_kernel(
    const opus::fp8_t* __restrict__ activations,const uint8_t* __restrict__ scales,
    const uint8_t* __restrict__ all_data,const uint8_t* __restrict__ all_auxiliary,
    const int32_t* __restrict__ tasks,const int32_t* __restrict__ task_count,
    const int32_t* __restrict__ gather,opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ output_scales,int routes,int data_bytes,int auxiliary_bytes)
{
#if defined(__gfx950__)
    constexpr int Rows=64,MAtoms=4,Threads=Waves*64;
    constexpr int BookWords=iq2r_codebook_words<Classes>();
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
        if(begin<0 || count<=0 || count>64 || begin+count>routes || expert<0 || expert>=256)continue;
        const int active_m=(count+15)/16;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        if(expert!=previous_expert)
        {
            // The copies are formed from copy 0; the barrier below publishes them.
            iq2r_v2_book_form<Threads>(shared.codebook,iq2r_v2_book_base_load<Threads>(aux,linear),linear);
            previous_expert=expert;
        }
        if(linear<Rows)shared.row_ids[linear]=gather[begin+min(linear,count-1)]/9;
        __syncthreads();
        uint32_t bases=0;
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
        // d words load one K256 ahead, with the records.
        uint32_t pending_d=iq2r_v2_bases<Classes>(aux,n_tile,24,0,lane_row);
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
                bases=pending_d;
                if(kt<47)
                {
                    issue_a(kt+1,1-slot);
                    pending=iq2r_scheduled_load_quad(data,data_bytes,(n_tile*48+kt+1)*kQuadBytes,lane);
                    if((kt&1)==1) pending_d=iq2r_v2_bases<Classes>(aux,n_tile,24,(kt+1)/2,lane_row);
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

// Weight bases of a down triplet: the lane's d word of every K256 column (low
// dword column 0, high dword column 1), see iq2r_kt_bases.
template<int Classes,int KT>
__device__ __forceinline__ uint64_t iq2r_triplet_bases(const uint8_t* aux,int n_block,int lane_row)
{
    constexpr int columns=KT/2;
    uint64_t bases=iq2r_v2_bases<Classes>(aux,n_block/3,columns,0,lane_row);
    if constexpr(columns>1)
        bases|=static_cast<uint64_t>(iq2r_v2_bases<Classes>(aux,n_block/3,columns,1,lane_row))<<32;
    return bases;
}

template<int Classes>
__device__ __forceinline__ uint32_t iq2r_kt_bases(uint64_t bases,int kt)
{
    return static_cast<uint32_t>(bases>>((kt>>1)*32));
}

// Decode in two halves, so a caller can read the next K tile's
// codewords from LDS before it applies this tile's signs.
template<int Atom,int Classes>
__device__ __forceinline__ void glm53_gather_atom(const IQ2RCompressedTriplet& compressed,const uint64_t* codebook,
                                                  const IQ2RLevels& levels,uint32_t book_high,uint64_t (&raw)[4])
{
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

template<int Atom,int Classes>
__device__ __forceinline__ void glm53_decode_packed_atom(
    const IQ2RCompressedTriplet& compressed,
    const uint64_t* codebook,
    uint32_t packed_bases,
    opus::i32x8_t& weight_fragment,
    uint32_t& scale_b,
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
    const IQ2RLevels levels = given_levels ? *given_levels : iq2r_levels<Classes>(metadata);
    union
    {
        opus::i32x8_t words;
        uint64_t codewords[4];
    } decoded;
#pragma unroll
    for(int codeword = 0; codeword < 4; ++codeword)
    {
        const int codebook_index =
            codeword<3?((index_lows>>(codeword*9))&511u):((index_lows>>27)|(index_highs<<5));
        // copy offset (bits 12+) | index * 8 (bits 3-11): disjoint, one v_lshl_or / v_and_or
        const uint32_t book_byte=iq2r_copy_offset<Classes>(codeword < 2 ? levels.low : levels.high, Atom)
                                 |(static_cast<uint32_t>(codebook_index)<<3);
        decoded.codewords[codeword] = glm53_apply_signs(
            *reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(codebook)+book_byte),
            (signs>>(codeword*8))&255u);
    }
    weight_fragment = decoded.words;
    scale_b = given_scales ? (*given_scales>>(Atom*8))&255u : iq2r_scale_b<Classes>(packed_bases, metadata, Atom);
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

// TP4 down, K tiles outermost: the three atoms of a triplet share
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

template<int Waves,int MAtoms,int Groups,int XCD,int TaskGroup,int KT,int Occupancy,int Classes>
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
        alignas(16) uint64_t codebook[iq2r_codebook_words<Classes>()];
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
            iq2r_stage_codebook_async<iq2r_codebook_words<Classes>(),Waves>(shared.codebook,aux,wave,lane);
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
                for(int kt=0;kt<KT;++kt)pending[kt]=iq2r_load_compact_triplet(data,glm53_ws_source_base<KT>(nb,kt),lane);
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
                    for(int kt=0;kt<KT;++kt)pending[kt]=iq2r_load_compact_triplet(data,glm53_ws_source_base<KT>(nb,kt),lane);
                    asm volatile("" ::: "memory");
                }
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
            __syncthreads();
        }
    }
#endif
}

// TP4 (K=512) weight-stationary down: one 64-row task tile stays in LDS while
// each wave decodes a K128 weight triplet once and applies it to every row
// atom. K tiles are outermost, so each output still accumulates
// ((k0+k1)+k2)+k3 from zero-initialized per-tile MFMAs, as in glm53_down_tp4_kernel.
template<int Waves,int Groups,int XCD,int TaskGroup,int KT,int Chunks,int Occupancy,int Classes>
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
        alignas(16) uint64_t codebook[iq2r_codebook_words<Classes>()];
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
            iq2r_v2_book_form<Threads>(shared.codebook,iq2r_v2_book_base_load<Threads>(aux,linear),linear);
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
                    // The atoms share the K tile's levels and scale bytes. Declared
                    // zeroed and then assigned: initialized directly, ROCm 7.2 adds
                    // eight instructions to this loop and the kernel runs ~15% slower.
                    IQ2RLevels lv{};uint32_t sc=0;
                    lv=iq2r_levels<Classes>(current.third.z);
                    sc=iq2r_scale_bytes<Classes>(iq2r_kt_bases<Classes>(bases,kt),current.third.z);
                    glm53_decode_packed_atom<0,Classes>(current,shared.codebook,iq2r_kt_bases<Classes>(bases,kt),w[0],sb[0],&lv,&sc);
                    glm53_decode_packed_atom<1,Classes>(current,shared.codebook,iq2r_kt_bases<Classes>(bases,kt),w[1],sb[1],&lv,&sc);
                    glm53_decode_packed_atom<2,Classes>(current,shared.codebook,iq2r_kt_bases<Classes>(bases,kt),w[2],sb[2],&lv,&sc);
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
// Chunks splits N across launches; outputs are stored [N/Columns][M][Columns]
// for glm53_reduce_chunk_kernel.
template<int Waves,int MAtoms,int Groups,int XCD,int TaskGroup,int Chunks,int Classes>
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
        alignas(16) uint64_t codebook[iq2r_codebook_words<Classes>()];
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
            iq2r_v2_book_form<Threads>(shared.codebook,iq2r_v2_book_base_load<Threads>(aux,linear),linear);
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
            // The atoms share each K tile's levels and scale bytes.
            IQ2RLevels lv[2];uint32_t sc[2];
#pragma unroll
                for(int kt=0;kt<2;++kt)
                {
                    lv[kt]=iq2r_levels<Classes>(compressed[kt].third.z);
                    sc[kt]=iq2r_scale_bytes<Classes>(iq2r_kt_bases<Classes>(bases,kt),compressed[kt].third.z);
                }
            // Finish one N16 atom at a time. Retain the exact two K128
            // partial sums, while shortening the lifetime of decoded weights
            // and accumulators enough to support a full M64 tile.
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<0,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0,&lv[0],&sc[0]);
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
                glm53_decode_packed_atom<0,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1,&lv[1],&sc[1]);
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
                    out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,((n_tile*M+row_base+out_r)*Columns+out_col%Columns)));
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<1,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0,&lv[0],&sc[0]);
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
                glm53_decode_packed_atom<1,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1,&lv[1],&sc[1]);
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
                    out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,((n_tile*M+row_base+out_r)*Columns+out_col%Columns)));
                }
            }
            {
                opus::vector_t<float,4> first[MAtoms]={};
                opus::i32x8_t w0;uint32_t sb0;
                glm53_decode_packed_atom<2,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0,&lv[0],&sc[0]);
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
                glm53_decode_packed_atom<2,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1,&lv[1],&sc[1]);
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
                    out_buffer.template store<4>(packed,glm53_row_offset(out_r<row_count,((n_tile*M+row_base+out_r)*Columns+out_col%Columns)));
                }
            }
            }
            __syncthreads();
        }
    }
#endif
}

// Weighted top-k sum for one N chunk of tiled prefill route outputs: one FP32
// FMA per route in route order, then one BF16 rounding.
template<int Chunks,int TopK>
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
    for(int route=0;route<TopK;++route)
    {
        const int sorted=__builtin_amdgcn_readfirstlane(scatter[token*9+route]);
        if(sorted<0)continue;  // invalid expert id: the route contributes nothing
        const float weight=__builtin_bit_cast(float,__builtin_amdgcn_readfirstlane(__builtin_bit_cast(uint32_t,route_weights[token*TopK+route])));
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

// Down for M<=4: one CTA per (Triplets x 48-column blocks, token) and one
// wave per route (eight waves, two per SIMD). Each route needs its own 16 KB of codebook
// copies, so a wave loads only its expert's 4 KB book and forms the copies in
// its LDS slot (iq2r_v2_form_copies): routes 4.. before issuing their weight
// loads, routes 0-3 after.
template<int KT,int Classes,int Triplets>
__global__ __launch_bounds__(512,1) void glm53_down_route9_v2_kernel(
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
    constexpr int Routes=8,Atoms=3,N=6144,K=KT*128,Columns=Triplets*Atoms*16;
    constexpr int BookWords=iq2r_codebook_words<Classes>();
    struct Storage {
        alignas(16) uint64_t codebook[Routes][BookWords];
        // Columns results per route, then a spare slot (padded to 16 bytes)
        // that lane groups 1-3 write.
        alignas(16) __hip_bfloat16 routes[Routes][Columns+8];
        // Each route's activation row, staged by a global -> LDS DMA.
        alignas(16) uint8_t act[Routes][K];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x;
    // threadIdx.y is wave-uniform (64-lane x dimension): a scalar route, expert
    // and row keep the codebook and activation DMAs free of waterfall loops.
    const int route=__builtin_amdgcn_readfirstlane(threadIdx.y);
    const int lane_row=lane%16,lane_group=lane/16;
    const int token=blockIdx.y,first=static_cast<int>(blockIdx.x)*Triplets*Atoms;
    const int expert=__builtin_amdgcn_readfirstlane(expert_ids[token*Routes+route]);
    const int row=__builtin_amdgcn_readfirstlane(scatter[token*9+route]);
    const int rows=static_cast<int>(gridDim.y)*9;
    // The scalars the epilogue and the loads need are loaded with the
    // prologue's, so no scalar load latency lands after the vector loads.
    asm volatile("" : "+s"(data_bytes), "+s"(auxiliary_bytes));
    float weights[Routes];
#pragma unroll
    for(int r=0;r<Routes;++r)weights[r]=route_weights[token*Routes+r];
    const bool valid=(expert>=0)&(expert<256)&(row>=0)&(row<rows);
    const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
    const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
    // The route's copies sit route * 16 KB into the codebooks. Its low bits
    // ride in each level byte above the class bits and the rest in byte 2 of
    // the copy offset, so one v_perm gives the whole offset from codebook[0]
    // (see iq2r_copy_offset).
    constexpr int RouteLowBits=4-iq2r_log2_classes<Classes>();
    static_assert(BookWords*8==1<<(16-RouteLowBits));
    const uint32_t route_bits=static_cast<uint32_t>(route&((1<<RouteLowBits)-1))*(0x01010101u<<(8-RouteLowBits));
    const uint32_t route_high=static_cast<uint32_t>(route>>RouteLowBits);
    // Copy 0 of the route's book, loaded before any weight load.
    uint4 book_base[4];
    if(valid)
    {
#pragma unroll
        for(int i=0;i<4;++i)book_base[i]=reinterpret_cast<const uint4*>(aux)[i*64+lane];
        asm volatile("" ::: "memory");
    }
    if(valid && route>=4)
    {
        iq2r_v2_form_copies<Classes>(shared.codebook[route],book_base,lane);
        asm volatile("" ::: "memory");
    }
    if(valid)
    {
        // The activation row goes to LDS with one DMA instruction instead of
        // two 16-byte loads per K tile in every lane (16 lanes share each
        // address), and the scale row is a scalar load whose byte the lane
        // group picks with one v_perm, replicated.
        const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(row)*K);
        if(lane<K/16)
        {
            opus::gmem<uint8_t> buffer(input,static_cast<unsigned int>(K));
            buffer.template async_load<16>(shared.act[route],lane*16,0,opus::number<0>{},opus::number<0>{});
        }
        asm volatile("" ::: "memory");
        {
            uint32_t scale_words[KT];
            const auto* scale_row=reinterpret_cast<const uint32_t*>(scales+static_cast<int64_t>(row)*(K/32));
#pragma unroll
            for(int ktile=0;ktile<KT;++ktile)scale_words[ktile]=scale_row[ktile];
            float gain=iq2r_gain<Classes>(aux,auxiliary_bytes,0);
            IQ2RCompressedTriplet compressed[Triplets][KT];
            IQ2RActivationFragment a[KT];
            uint32_t sa[KT];
            uint64_t bases[Triplets];
#pragma unroll
            for(int ktile=0;ktile<KT;++ktile)
#pragma unroll
                for(int t=0;t<Triplets;++t)
                {
                    const uint32_t tr=blockIdx.x*Triplets+t;  // first+t*Atoms == 3*tr
                    compressed[t][ktile]=iq2r_scheduled_load_compact_uniform(
                        data,data_bytes,static_cast<int>(((tr>>1)*KT+ktile)*kGroupBytes+(tr&1)*kTripletBytes),lane);
                }
#pragma unroll
            for(int t=0;t<Triplets;++t)bases[t]=iq2r_triplet_bases<Classes,KT>(aux,first+t*Atoms,lane_row);
            if(route<4)iq2r_v2_form_copies<Classes>(shared.codebook[route],book_base,lane);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            // after the vector wait: the scalar loads' wait lands here
            asm volatile("" : "+s"(gain));
#pragma unroll
            for(int r=0;r<Routes;++r)asm volatile("" : "+s"(weights[r]));
#pragma unroll
            for(int ktile=0;ktile<KT;++ktile)
            {
                uint32_t word=scale_words[ktile];
                asm volatile("" : "+s"(word));
                sa[ktile]=__builtin_amdgcn_perm(0u,word,static_cast<uint32_t>(lane_group)*0x01010101u);
            }
#pragma unroll
            for(int ktile=0;ktile<KT;++ktile)
            {
                const int ak=ktile*128+lane_group*16;
                *reinterpret_cast<uint4*>(a[ktile].bytes)=*reinterpret_cast<const uint4*>(shared.act[route]+ak);
                *reinterpret_cast<uint4*>(a[ktile].bytes+16)=*reinterpret_cast<const uint4*>(shared.act[route]+ak+64);
            }
            // The codebook reads (gather) of unit + 1 issue before the signs
            // and MFMAs of unit.
            uint64_t raw[2][Atoms][4];
            const auto gather=[&](int unit,uint64_t (&r)[Atoms][4])
            {
                const IQ2RCompressedTriplet& c=compressed[unit/KT][unit%KT];
                const IQ2RLevels levels=iq2r_levels<Classes>(c.third.z,route_bits);
                glm53_gather_atom<0,Classes>(c,shared.codebook[0],levels,route_high,r[0]);
                glm53_gather_atom<1,Classes>(c,shared.codebook[0],levels,route_high,r[1]);
                glm53_gather_atom<2,Classes>(c,shared.codebook[0],levels,route_high,r[2]);
            };
            gather(0,raw[0]);
#pragma unroll
            for(int t=0;t<Triplets;++t)
            {
                opus::vector_t<float,4> k_sums[KT][Atoms]={};
#pragma unroll
                for(int ktile=0;ktile<KT;++ktile)
                {
                    const int unit=t*KT+ktile;
                    if(unit+1<Triplets*KT)gather(unit+1,raw[(unit+1)&1]);
                    const IQ2RCompressedTriplet& c=compressed[t][ktile];
                    opus::i32x8_t b[3]={glm53_sign_atom<0>(c,raw[unit&1][0]),glm53_sign_atom<1>(c,raw[unit&1][1]),
                                        glm53_sign_atom<2>(c,raw[unit&1][2])};
                    uint32_t s=sa[ktile];
                    asm volatile("" : "+v"(s));
                    iq2r_triplet_mfma<0>(a[ktile].words,b,k_sums[ktile],s,
                                         iq2r_scale_bytes<Classes>(iq2r_kt_bases<Classes>(bases[t],ktile),c.third.z));
                }
                // Every lane group stores, not only lane group 0, so the MFMAs
                // above are not sunk into a lane_group==0 branch (see
                // glm53_row_offset). Lane groups 1-3 write the spare slot.
#pragma unroll
                for(int atom=0;atom<Atoms;++atom)
                {
                    // ((k0+k1)+k2)+k3 in K-tile order.
                    float value=k_sums[0][atom][0];
#pragma unroll
                    for(int ktile=1;ktile<KT;++ktile)value+=k_sums[ktile][atom][0];
                    shared.routes[route][lane_group==0?(t*Atoms+atom)*16+lane_row:Columns]=__float2bfloat16(value*gain);
                }
            }
        }
    }
    else if(lane_group==0)
    {
#pragma unroll
        for(int column=lane_row;column<Columns;column+=16)
            shared.routes[route][column]=__float2bfloat16(0.0f);
    }
    __syncthreads();
    if(route==0 && lane_group<Triplets)
    {
#pragma unroll
        for(int atom=0;atom<Atoms;++atom)
        {
            const int column=(lane_group*Atoms+atom)*16+lane_row;
            float combined=0.0f;
#pragma unroll
            for(int r=0;r<8;++r)
                combined=fmaf(__bfloat162float(shared.routes[r][column]),weights[r],combined);
            output[static_cast<int64_t>(token)*N+first*16+column]=__float2bfloat16(combined);
        }
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
    const IQ2RLevels levels=iq2r_levels<Classes>(compressed.metadata);
    bases=iq2r_scale_bytes<Classes>(bases,compressed.metadata);
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
        const uint32_t sb=(bases>>(atom*8))&255u;
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
template<int Waves,int MAtoms,int Groups,int XCD,int TaskGroup,int Classes>
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
        alignas(16) uint64_t codebook[iq2r_codebook_words<Classes>()];
        alignas(16) uint8_t input[Rows*256];
        uint8_t scales[Rows*8];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=__builtin_amdgcn_readfirstlane(threadIdx.y),linear=wave*64+lane;
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
            iq2r_stage_codebook_async<iq2r_codebook_words<Classes>(),Waves>(shared.codebook,aux,wave,lane);
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
                for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,glm53_ws_source_base<2>(nb,kt),lane);
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
            for(int kt=0;kt<2;++kt)pending[kt]=iq2r_load_compact_triplet(data,glm53_ws_source_base<2>(nb,kt),lane);
            asm volatile("" ::: "memory");
        }
            [[maybe_unused]] IQ2RLevels lv[2];[[maybe_unused]] uint32_t sc[2];
            if constexpr(MAtoms>1)
#pragma unroll
                for(int kt=0;kt<2;++kt)
                {
                    lv[kt]=iq2r_levels<Classes>(compressed[kt].third.z);
                    sc[kt]=iq2r_scale_bytes<Classes>(iq2r_kt_bases<Classes>(bases,kt),compressed[kt].third.z);
                }
            if constexpr(MAtoms==1)
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
                glm53_decode_packed_atom<0,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0,&lv[0],&sc[0]);
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
                glm53_decode_packed_atom<0,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1,&lv[1],&sc[1]);
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
                glm53_decode_packed_atom<1,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0,&lv[0],&sc[0]);
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
                glm53_decode_packed_atom<1,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1,&lv[1],&sc[1]);
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
                glm53_decode_packed_atom<2,Classes>(compressed[0],shared.codebook,iq2r_kt_bases<Classes>(bases,0),w0,sb0,&lv[0],&sc[0]);
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
                glm53_decode_packed_atom<2,Classes>(compressed[1],shared.codebook,iq2r_kt_bases<Classes>(bases,1),w1,sb1,&lv[1],&sc[1]);
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
template<int Classes>
__device__ __forceinline__ void glm53_issue_codebook(
    const IQ2RCompressedQuad& compressed,const uint64_t* codebook,int atom,
    uint64_t (&mag)[4],IQ2RLevels levels={0u,0u})
{
    const uint32_t lows=atom==0?compressed.first.x:atom==1?compressed.first.z:atom==2?compressed.second.x:compressed.second.z;
    const uint32_t highs=(compressed.metadata>>(atom*8))&15u;
    const uint32_t base=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(codebook));
    // Copies are 4 KB apart and an entry offset is below 4 KB, so OR joins
    // them: one v_and_or_b32 per codeword.
    const uint32_t copy0=iq2r_copy_offset<Classes>(levels.low,atom);
    const uint32_t copy1=iq2r_copy_offset<Classes>(levels.high,atom);
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

// Decode gate K step: the codebook reads of the next atom, or of the next
// quad's atom 0, issue before this atom's signs and MFMAs. ``carried`` holds
// the copy offsets of ``compressed`` from the previous step and returns those
// of ``next``.
template<int MAtoms,int Classes>
__device__ __forceinline__ void glm53_cross_k_decode_mfma(
    const IQ2RCompressedQuad& compressed,const IQ2RCompressedQuad& next,
    const uint64_t* codebook,uint32_t bases,
    const IQ2RActivationFragment* activation,const uint32_t* scale_a,
    opus::vector_t<float,4> (&accumulators)[MAtoms][4],int active_m,
    uint64_t (&pending)[4],bool first_k,bool next_k,IQ2RLevels& carried)
{
    const IQ2RLevels levels=carried;
    const IQ2RLevels next_levels=iq2r_levels<Classes>(next.metadata);
    bases=iq2r_scale_bytes<Classes>(bases,compressed.metadata);
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
#pragma unroll
        for(int m=0;m<MAtoms;++m) {
            if(m>=active_m)continue;
            auto mma=opus::mfma<opus::fp8_t,opus::fp8_t,opus::fp32_t,16,16,128>{};
            auto& acc=accumulators[m][atom];
            // The MFMA's scale op_sel reads the atom's byte of the scale
            // word, so the four atoms' scale bytes stay in one register.
            if(atom==0) acc=mma(activation[m].words,decoded.words,acc,scale_a[m],bases,opus::number<0>{},opus::number<0>{});
            else if(atom==1) acc=mma(activation[m].words,decoded.words,acc,scale_a[m],bases,opus::number<0>{},opus::number<1>{});
            else if(atom==2) acc=mma(activation[m].words,decoded.words,acc,scale_a[m],bases,opus::number<0>{},opus::number<2>{});
            else acc=mma(activation[m].words,decoded.words,acc,scale_a[m],bases,opus::number<0>{},opus::number<3>{});
        }
    }
    carried=next_levels;
}

// Decode gate (M2..1024): the eight waves each sum a K768 slice, and the
// weights stay in registers, so work items synchronize only to form a new
// expert's codebook copies. Items are split into 8 contiguous per-XCD ranges; a new expert's copies are
// formed in LDS from its 4 KB copy 0 (iq2r_v2_book_form); the K loop uses
// buffer descriptors with scalar offsets; weight loads are non-temporal. Two
// workgroups share a CU.
template<int NT,int Classes>
__global__ __launch_bounds__(512,4) void glm53_gate_decode_nobarrier_v2_kernel(
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
    constexpr int BookWords=iq2r_codebook_words<Classes>();
    struct Storage {
        alignas(16) uint64_t codebook[BookWords];
        alignas(16) float partial[8][4][4][64];
    };
    __shared__ Storage shared;
    const int lane=threadIdx.x,wave=threadIdx.y;
    const int lane_row=lane%16,lane_group=lane/16;
    const int num_tasks=task_count[0];
    const int total=num_tasks*NT;
    int current_expert=-1;
    const int xcd=blockIdx.x%8,rank=blockIdx.x/8,ranks=(gridDim.x-xcd+7)/8;
    const int per=(total+7)/8,tall=total%8==0?8:total%8;
    const int first=xcd<tall?xcd*per:tall*per+(xcd-tall)*(per-1);
    const int length=xcd<tall?per:per-1;
    for(int local=rank;local<length;local+=ranks)
    {
        const int index=first+local;
        const int task=index/NT,n_tile=index%NT;
        const int row_begin=tasks[task*3];
        const int row_end=min(row_begin+16,row_begin+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=256) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // The previous item ended in a barrier, so the book may be overwritten.
        const bool new_expert=expert!=current_expert;
        // A new expert's copy 0 loads first; its copies are formed once the
        // item's other loads are issued.
        uint2 book_base{};
        if(new_expert) {
            book_base=iq2r_v2_book_base_load<512>(aux,wave*64+lane);
            current_expert=expert;
        }
        const float2 gains=iq2r_gains<Classes>(aux,auxiliary_bytes);
        int rows[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        const int swave=__builtin_amdgcn_readfirstlane(wave);
        int base=(n_tile*48+swave*6)*kQuadBytes;
        const opus::i32x4_t act_rsrc=iq2r_buffer_rsrc(activations,routes/9*K);
        const opus::i32x4_t scale_rsrc=iq2r_buffer_rsrc(scales,routes/9*192);
        const opus::i32x4_t d_rsrc=iq2r_buffer_rsrc(aux,auxiliary_bytes);
        int act_off[MAtoms],scale_off[MAtoms];
#pragma unroll
        for(int m=0;m<MAtoms;++m) { act_off[m]=rows[m]*K+lane_group*16;scale_off[m]=rows[m]*192+lane_group; }
        // d table: [N tile][24 K256 columns][16 rows] words after the copies.
        const int d_off=BookWords*8+(n_tile*24*16+lane_row)*4;
        uint64_t pending_book[4]={};
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad<2>(data,data_bytes,base,lane);
        if(new_expert) {
            // Plain LDS stores: the barrier publishes them, and the item's
            // first quad and activation loads stay in flight across it.
            iq2r_v2_book_form<512>(shared.codebook,book_base,wave*64+lane);
            __syncthreads();
        }
        IQ2RLevels levels=iq2r_levels<Classes>(pending.metadata);
        for(int iteration=0;iteration<6;++iteration)
        {
            const int kt=swave*6+iteration;
            IQ2RActivationFragment a[MAtoms]={};uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if(m>=active_m) continue;
                *reinterpret_cast<opus::i32x4_t*>(a[m].bytes)=iq2r_scheduled_load_dwordx4(act_rsrc,act_off[m],kt*128,0);
                *reinterpret_cast<opus::i32x4_t*>(a[m].bytes+16)=iq2r_scheduled_load_dwordx4(act_rsrc,act_off[m],kt*128+64,0);
                sa[m]=iq2r_buffer_load_u8(scale_rsrc,scale_off[m],kt*4,0);
            }
            // d word of this K256, loaded with the activations.
            const uint32_t bases=iq2r_scheduled_buffer_load1(d_rsrc,d_off,(kt/2)*64,0);
            const IQ2RCompressedQuad compressed=pending;
            if(iteration<5) {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad<2>(data,data_bytes,base,lane);
            }
            glm53_cross_k_decode_mfma<MAtoms,Classes>(compressed,pending,shared.codebook,bases,a,sa,accumulators,active_m,pending_book,iteration==0,iteration<5,levels);
        }
        // Weights stay in registers, so no barrier is needed before the
        // partial stores; the store and reader barriers protect the partials.
#pragma unroll
        for(int m=0;m<MAtoms;++m)
        {
            if(m>=active_m)continue;
#pragma unroll
            for(int atom=0;atom<4;++atom)
                for(int c=0;c<4;++c) shared.partial[wave][atom][c][lane]=accumulators[m][atom][c];
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
                        value+=shared.partial[source][atom][wave][lane];
                    values[atom]=__bfloat162float(__float2bfloat16(value*iq2r_row_gain(gains,lane_row)));
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

} // namespace

void iq2r_check_gfx950()
{
    // The IQ2R kernel bodies are compiled for gfx950 only; elsewhere they are empty.
    static const std::string arch = get_gpu_arch();
    AITER_CHECK(arch == "gfx950", "IQ2R kernels require gfx950, not ", arch);
}

// GLM-5.3 IQ2R packed MoE launchers. Tensor-parallel width follows from the
// packed gate stack: 16 N tiles (intermediate 512) at TP4, 8 tiles at TP8.
namespace {

constexpr int kGlm53Hidden = 6144;
// 256 routed experts, top-8 (the shared expert stays FP8 outside the MoE).
// Routes keep nine slots per token; slot 8 is empty.
constexpr int kGlm53Experts = 256;
constexpr int kGlm53TopK = 8;
constexpr int kGlm53RouteSlots = 9;
// Codebook copies of the c4e4q level set (aiter/ops/iq2r_v2.py).
constexpr int C = 4;

// Packed auxiliary bytes (iq2r_glm53_v2_aux_bytes in aiter/iq2r_glm53.py).
// atoms: 4 for the gate/up quads, 3 for the down triplets.
int64_t glm53_auxiliary_bytes(int64_t n, int64_t k, int atoms)
{
    return C * kCodebookBytes + n / (16 * atoms) * (k / 256) * 64 + 8;
}

void glm53_check_stack(const aiter_tensor_t& data, const aiter_tensor_t& auxiliary, int64_t data_bytes,
                       int64_t auxiliary_bytes)
{
    AITER_CHECK(data.dtype() == AITER_DTYPE_u8 && data.dim() == 2 && data.size(0) == kGlm53Experts &&
                    data.size(1) == data_bytes,
                "GLM-5.3 IQ2R weights must be a packed uint8 stack of 256 experts");
    AITER_CHECK(auxiliary.dtype() == AITER_DTYPE_u8 && auxiliary.dim() == 2 && auxiliary.size(0) == kGlm53Experts &&
                    auxiliary.size(1) == auxiliary_bytes,
                "GLM-5.3 IQ2R auxiliary must be the packed IQ2R-v2 bytes of 256 experts");
}

bool glm53_tp4_from_gate(const aiter_tensor_t& data, const aiter_tensor_t& auxiliary)
{
    const bool tp4 = data.dim() == 2 && data.size(1) == 16 * 48 * kQuadBytes;
    const int64_t n = tp4 ? 1024 : 512;
    glm53_check_stack(data, auxiliary, n / 64 * 48 * kQuadBytes, glm53_auxiliary_bytes(n, kGlm53Hidden, 4));
    return tp4;
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
                            const aiter_tensor_t& auxiliary)
{
    const int intermediate = static_cast<int>(activations.size(1));
    AITER_CHECK(intermediate == 256 || intermediate == 512, "GLM-5.3 IQ2R down K must be 256 or 512");
    glm53_check_stack(data, auxiliary, expected_data_bytes(kGlm53Hidden, intermediate),
                      glm53_auxiliary_bytes(kGlm53Hidden, intermediate, 3));
    AITER_CHECK(activations.dtype() == AITER_DTYPE_fp8 && activations.dim() == 2 &&
                    activations.size(0) % kGlm53RouteSlots == 0,
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
    const bool tp4 = glm53_tp4_from_gate(data, auxiliary);
    glm53_check_device({&activations, &scales, &auxiliary, &tasks, &task_count, &output, &output_scales}, data);
    const int routes = kGlm53RouteSlots;
    glm53_check_gate_io(activations, scales, output, output_scales, routes, routes, tp4 ? 512 : 256);
    glm53_check_tasks(tasks, task_count, routes);
    HipDeviceGuard device_guard(data.device_id);
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
    const bool tp4 = glm53_tp4_from_gate(data, auxiliary);
    glm53_check_device(
        {&activations, &scales, &auxiliary, &tasks, &task_count, &gather, &output, &output_scales}, data);
    const int tokens = static_cast<int>(activations.size(0));
    const int routes = tokens * kGlm53RouteSlots;
    glm53_check_gate_io(activations, scales, output, output_scales, tokens, routes, tp4 ? 512 : 256);
    AITER_CHECK(gather.dtype() == AITER_DTYPE_i32 && gather.dim() == 1 && gather.size(0) == routes,
                "GLM-5.3 IQ2R gate gather must be int32 [routes]");
    AITER_CHECK(grid_multiplier >= 1 && grid_multiplier <= 8, "GLM-5.3 IQ2R gate grid multiplier");
    const bool prefill = kernel == kGlm53GatePrefill;
    AITER_CHECK(prefill || kernel == kGlm53GateDecodeNoBarrier, "GLM-5.3 IQ2R gate kernel id");
    AITER_CHECK(tokens >= 2 && tokens <= (prefill ? 4096 : 1024),
                "GLM-5.3 IQ2R gate token range: decode 2..1024, prefill 2..4096");
    const int rows = prefill ? 64 : 16;
    glm53_check_tasks(tasks, task_count, (routes + rows - 1) / rows + std::min(routes, kGlm53Experts));
    HipDeviceGuard device_guard(data.device_id);
    const auto launch = prefill ? (tp4 ? glm53_gate_prefill_kernel<4, 8, 4, 2, 16, C>
                                       : glm53_gate_prefill_kernel<4, 8, 4, 2, 8, C>)
                                : (tp4 ? glm53_gate_decode_nobarrier_v2_kernel<16, C>
                                       : glm53_gate_decode_nobarrier_v2_kernel<8, C>);
    hipLaunchKernelGGL(launch,
                       dim3(static_cast<int>(grid_multiplier) * static_cast<int>(get_num_cu_func())),
                       dim3(64, prefill ? 4 : 8),
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
    const int K = glm53_down_intermediate(activations, scales, data, auxiliary);
    const bool tp4 = K == 512;
    glm53_check_device({&activations, &scales, &auxiliary, &tasks, &task_count, &output}, data);
    const int routes = static_cast<int>(activations.size(0));
    AITER_CHECK(routes >= 2 * kGlm53RouteSlots && routes <= 1024 * kGlm53RouteSlots,
                "GLM-5.3 IQ2R decode down covers M2..1024");
    glm53_check_tasks(tasks, task_count, (routes + 31) / 32 + std::min(routes, kGlm53Experts));
    AITER_CHECK(output.dtype() == AITER_DTYPE_bf16 && output.dim() == 2 && output.size(0) == routes &&
                    output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R down output must be BF16 [routes, 6144]");
    AITER_CHECK(kernel == kGlm53DownSingle || (kernel == kGlm53DownOrdered && !tp4), "GLM-5.3 IQ2R down kernel id");
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
        static_cast<__hip_bfloat16*>(output.data_ptr()), routes, static_cast<int>(data.size(0)),             \
        static_cast<int>(data.size(1)), static_cast<int>(auxiliary.size(1))
    // Single: one 16-row atom per pass for decode-sized tasks, three workgroups per CU.
    if(tp4)
        hipLaunchKernelGGL((glm53_down_tp4_kernel<4, 1, 2, 8, 4, 4, 3, C>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS);
    else if(kernel == kGlm53DownSingle)
        hipLaunchKernelGGL((glm53_down_tp8_ordered_kernel<4, 1, 2, 8, 4, C>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS);
    else
        hipLaunchKernelGGL((glm53_down_tp8_ordered_kernel<4, 2, 2, 8, 4, C>),
                           grid, dim3(64, 4), 0, getCurrentHIPStream(), GLM53_DOWN_ARGS);
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
    const int K = glm53_down_intermediate(activations, scales, data, auxiliary);
    glm53_check_device({&activations, &scales, &auxiliary, &expert_ids, &scatter, &route_weights, &output}, data);
    const int routes = static_cast<int>(activations.size(0));
    const int tokens = routes / kGlm53RouteSlots;
    AITER_CHECK(tokens == 1 || tokens == 2 || tokens == 4, "GLM-5.3 IQ2R route9 down covers M1, M2 and M4");
    AITER_CHECK(expert_ids.dtype() == AITER_DTYPE_i32 && expert_ids.numel() == tokens * kGlm53TopK &&
                    scatter.dtype() == AITER_DTYPE_i32 && scatter.numel() == routes,
                "GLM-5.3 IQ2R route9 expert ids must be int32 [tokens, 8], scatter int32 [tokens * 9]");
    AITER_CHECK(route_weights.dtype() == AITER_DTYPE_fp32 && route_weights.numel() == tokens * kGlm53TopK,
                "GLM-5.3 IQ2R route9 weights must be float32 [tokens, 8]");
    AITER_CHECK(output.dtype() == AITER_DTYPE_bf16 && output.dim() == 2 && output.size(0) == tokens &&
                    output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R route9 output must be BF16 [tokens, 6144]");
    HipDeviceGuard device_guard(data.device_id);
    // Two triplets per CTA at M4: half the CTAs, and half the per-route codebook formation.
    const int triplets = tokens >= 4 ? 2 : 1;
    const auto kernel = K == 512
                            ? (triplets == 2 ? glm53_down_route9_v2_kernel<4, C, 2> : glm53_down_route9_v2_kernel<4, C, 1>)
                            : (triplets == 2 ? glm53_down_route9_v2_kernel<2, C, 2> : glm53_down_route9_v2_kernel<2, C, 1>);
    hipLaunchKernelGGL(kernel,
                       dim3(128 / triplets, tokens),
                       dim3(64, kGlm53TopK),
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
    const int K = glm53_down_intermediate(activations, scales, data, auxiliary);
    glm53_check_device(
        {&activations, &scales, &auxiliary, &tasks, &task_count, &route_output, &scatter, &route_weights, &output},
        data);
    const int routes = static_cast<int>(activations.size(0));
    const int tokens = routes / kGlm53RouteSlots;
    AITER_CHECK(tokens >= 2 && tokens <= 4096, "GLM-5.3 IQ2R prefill down covers M2..4096");
    glm53_check_tasks(tasks, task_count, (routes + 63) / 64 + std::min(routes, kGlm53Experts));
    AITER_CHECK(route_output.dtype() == AITER_DTYPE_bf16 && route_output.dim() == 2 &&
                    route_output.size(0) == routes && route_output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R prefill route output must be BF16 [routes, 6144]");
    AITER_CHECK(scatter.dtype() == AITER_DTYPE_i32 && scatter.numel() == routes &&
                    route_weights.dtype() == AITER_DTYPE_fp32 && route_weights.numel() == tokens * kGlm53TopK,
                "GLM-5.3 IQ2R prefill scatter and weights [tokens, 8]");
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
        // TP4: 16 persistent CTAs per CU with weight-stationary K loops, items of
        // 768 columns; TP8: N-wave tiles sharing one cached activation tile.
        if(K == 512 && chunks == 1)
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp4_kernel<4, 4, 8, 4, 4, 1, 3, C>), 16,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, static_cast<int>(data.size(0)));
        else if(K == 512)
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp4_kernel<4, 4, 8, 4, 4, 2, 3, C>), 16,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, static_cast<int>(data.size(0)));
        else if(chunks == 1)
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp8_kernel<4, 2, 2, 8, 4, 1, C>), 8,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, static_cast<int>(data.size(0)));
        else
            GLM53_PREFILL_DOWN((glm53_down_prefill_tp8_kernel<4, 2, 2, 8, 4, 2, C>), 8,
                               static_cast<__hip_bfloat16*>(route_output.data_ptr()), routes, static_cast<int>(data.size(0)));
#undef GLM53_PREFILL_DOWN
        const int width = kGlm53Hidden / static_cast<int>(chunks);
        hipLaunchKernelGGL((chunks == 1 ? glm53_reduce_chunk_kernel<1, kGlm53TopK> : glm53_reduce_chunk_kernel<2, kGlm53TopK>),
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
