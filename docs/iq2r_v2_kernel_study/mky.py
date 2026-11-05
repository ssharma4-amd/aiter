"""Write the y-series gate kernel experiment (from a6) into the exp tree.
GLM53_GATE_BOOKS 1 = one gate/up book staged (B, 16 KB; numerically wrong for 2-book data, timing only)
GLM53_GATE_DOUBLE 1 = two codebook buffers; the next item's book is DMA'd during this item."""
src = open("a6_iq2r_gemm_gfx950.cu").read()
def sub(a, b):
    global src
    assert src.count(a) == 1, a[:80]
    src = src.replace(a, b)
sub("""#ifndef GLM53_GATE_PREFETCH
#define GLM53_GATE_PREFETCH 1
#endif
#ifndef GLM53_CARRY_LEVELS
#define GLM53_CARRY_LEVELS 1
#endif
#ifndef GLM53_GATE_GROUP
#define GLM53_GATE_GROUP 1
#endif
""", """#ifndef GLM53_GATE_BOOKS
#define GLM53_GATE_BOOKS 2
#endif
#ifndef GLM53_GATE_DOUBLE
#define GLM53_GATE_DOUBLE 0
#endif
""")
start = src.index("    constexpr int K=6144, MAtoms=1;\n    constexpr int BookWords=iq2r_codebook_words<Classes,2>();\n    struct Storage {\n        alignas(16) uint64_t codebook[BookWords];\n        alignas(16) float partial[8][4][4][64];")
end = src.index("            if(wave<4)", start)
body = src[start:end]
new = r"""    constexpr int K=6144, MAtoms=1;
    constexpr int GateBooks=Classes>0?GLM53_GATE_BOOKS:2;
    constexpr int BookWords=iq2r_codebook_words<Classes,GateBooks>();
    constexpr bool Double=GLM53_GATE_DOUBLE;
    struct Storage {
        alignas(16) uint64_t codebook[Double?2:1][BookWords];
        alignas(16) float partial[8][4][4][64];
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
    for(int local=rank;local<length;local+=ranks)
    {
        const int index=first+local;
        const int task=index/NT,n_tile=index%NT;
        const int row_begin=tasks[task*3];
        const int row_end=min(row_begin+16,row_begin+tasks[task*3+1]);
        const int expert=tasks[task*3+2];
        const int active_m=(row_end-row_begin+15)/16;
        if(row_begin>=row_end || row_begin<0 || row_end>routes || expert<0 || expert>=257) continue;
        const uint8_t* data=all_data+static_cast<int64_t>(expert)*data_bytes;
        const uint8_t* aux=all_auxiliary+static_cast<int64_t>(expert)*auxiliary_bytes;
        // A codebook prefetched by the previous item is complete: its DMA
        // preceded that item's last loads, and the item ended in barriers.
        if(Double && expert!=current_expert && expert==other_expert) {
            current^=1; other_expert=current_expert; current_expert=expert;
        }
        const bool new_expert=expert!=current_expert;
        if(new_expert) {
            iq2r_stage_codebook_async<BookWords,8>(shared.codebook[current],aux,wave,lane);
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
        for(int m=0;m<MAtoms;++m) rows[m]=gather[min(row_begin+m*16+lane_row,row_end-1)]/9;
        opus::vector_t<float,4> accumulators[MAtoms][4]={};
        int base=(n_tile*48+wave*6)*kQuadBytes;
        uint64_t pending_book[4]={};
        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
        if(new_expert) {
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
        }
        if constexpr(Double) {
            // Stage the next item's codebook in the other buffer, which no
            // wave reads before the next item.
            const int next_local=local+ranks;
            const int next_expert=next_local<length?tasks[(first+next_local)/NT*3+2]:-1;
            if(next_expert>=0 && next_expert<257 && next_expert!=expert && next_expert!=other_expert) {
                iq2r_stage_codebook_async<BookWords,8>(shared.codebook[current^1],
                    all_auxiliary+static_cast<int64_t>(next_expert)*auxiliary_bytes,wave,lane);
                other_expert=next_expert;
            }
        }
        const uint32_t book_bits=GateBooks==1?0u:iq2r_book_bits<Classes>(lane_row);
        IQ2RLevels levels=iq2r_levels<Classes>(pending.metadata,book_bits);
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
            // IQ2R-v2 d word of this K256, reloaded with the activations
            // rather than held for the whole K768 slice.
            if constexpr(Classes>0) bases=d_lane[iteration/2*16];
            const IQ2RCompressedQuad compressed=pending;
            if(iteration<5) {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            glm53_cross_k_decode_mfma_carried<MAtoms,Classes>(compressed,pending,lane_book,bases,a,sa,accumulators,active_m,pending_book,iteration==0,iteration<5,levels,book_bits);
        }
        // A prefetch DMA must land before the barrier below publishes it.
        if constexpr(Double) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
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
"""
src = src[:start] + new + src[end:]
# Remaining references to the old 2-book layout in the launcher (aux size) stay unchanged.
open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu", "w").write(src)
print("ok")
# Timing-only knobs: X_NOCOPY = every codeword from copy 0; X_NOD = constant d (no d loads).
src = open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu").read()
sub("""#ifndef GLM53_GATE_DOUBLE
#define GLM53_GATE_DOUBLE 0
#endif
""", """#ifndef GLM53_GATE_DOUBLE
#define GLM53_GATE_DOUBLE 0
#endif
#ifndef GLM53_X_NOCOPY
#define GLM53_X_NOCOPY 0
#endif
#ifndef GLM53_X_NOD
#define GLM53_X_NOD 0
#endif
""")
sub("""    const uint32_t copy0=iq2r_copy_offset<Classes>(levels.low,atom);
    const uint32_t copy1=iq2r_copy_offset<Classes>(levels.high,atom);
    const uint32_t a0=base+(copy0|(lows&511u)*8);""", """    const uint32_t copy0=GLM53_X_NOCOPY?0u:iq2r_copy_offset<Classes>(levels.low,atom);
    const uint32_t copy1=GLM53_X_NOCOPY?0u:iq2r_copy_offset<Classes>(levels.high,atom);
    const uint32_t a0=base+(copy0|(lows&511u)*8);""")
sub("""            if constexpr(Classes>0) bases=d_lane[iteration/2*16];
            const IQ2RCompressedQuad compressed=pending;
            if(iteration<5) {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            glm53_cross_k_decode_mfma_carried""", """            if constexpr(Classes>0) bases=GLM53_X_NOD?0x70707070u:d_lane[iteration/2*16];
            const IQ2RCompressedQuad compressed=pending;
            if(iteration<5) {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
            }
            glm53_cross_k_decode_mfma_carried""")
open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu", "w").write(src)
print("ok x")
# X_ORDER: issue this K128's activations, scale and d before the next quad's loads.
src = open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu").read()
sub("""#ifndef GLM53_X_NOD
#define GLM53_X_NOD 0
#endif
""", """#ifndef GLM53_X_NOD
#define GLM53_X_NOD 0
#endif
#ifndef GLM53_X_ORDER
#define GLM53_X_ORDER 0
#endif
""")
sub("""            if constexpr(Classes>0) bases=GLM53_X_NOD?0x70707070u:d_lane[iteration/2*16];
            const IQ2RCompressedQuad compressed=pending;
            if(iteration<5) {""", """            if constexpr(Classes>0) bases=GLM53_X_NOD?0x70707070u:d_lane[iteration/2*16];
            if(GLM53_X_ORDER) __builtin_amdgcn_sched_barrier(0);
            const IQ2RCompressedQuad compressed=pending;
            if(iteration<5) {""")
open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu", "w").write(src)
print("ok o")
# X_ORDER also passes the raw activation scale byte (the MFMA reads byte 0), so nothing consumes the loads before the barrier.
src = open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu").read()
nb = src.index("void glm53_gate_decode_nobarrier_kernel(")
head, tail = src[:nb], src[nb:]
a = "                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;"
assert tail.count(a) == 1
tail = tail.replace(a, "                sa[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*(GLM53_X_ORDER?1u:0x01010101u);")
src = head + tail
open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu", "w").write(src)
print("ok o2")
# X_DPRE: load the wave's three d words at item start instead of one per K128.
src = open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu").read()
sub("""#ifndef GLM53_X_ORDER
#define GLM53_X_ORDER 0
#endif
""", """#ifndef GLM53_X_ORDER
#define GLM53_X_ORDER 0
#endif
#ifndef GLM53_X_DPRE
#define GLM53_X_DPRE 0
#endif
""")
sub("""        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
        if(new_expert) {
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
        }
        if constexpr(Double) {""", """        IQ2RCompressedQuad pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
        uint32_t d_words[3]={};
        if(Classes>0 && GLM53_X_DPRE) {
#pragma unroll
            for(int c=0;c<3;++c) d_words[c]=d_lane[c*16];
        }
        if(new_expert) {
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
        }
        if constexpr(Double) {""")
sub("""            if constexpr(Classes>0) bases=GLM53_X_NOD?0x70707070u:d_lane[iteration/2*16];""",
    """            if constexpr(Classes>0) bases=GLM53_X_NOD?0x70707070u:GLM53_X_DPRE?(iteration<2?d_words[0]:iteration<4?d_words[1]:d_words[2]):d_lane[iteration/2*16];""")
open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu", "w").write(src)
print("ok dpre")
