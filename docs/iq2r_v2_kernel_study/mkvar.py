"""Write a throwaway timing variant of the s8 kernel source into the exp tree. usage: mkvar.py x0|x1|x2|x3"""
import sys
src = open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/experiments/iq2r-v2-20261103/s8_iq2r_gemm_gfx950.cu").read()
v = sys.argv[1]
if "p" in v:  # perm selectors as SGPR operands
    v = v.replace("p", "")
    src = src.replace("""    else return __builtin_amdgcn_perm(0u,levels,0x0c0c000cu|(static_cast<uint32_t>(atom)<<8));""",
"""    else {
        uint32_t offset;
        asm("v_perm_b32 %0, 0, %1, %2" : "=v"(offset) : "v"(levels),"s"(0x0c0c000cu|(static_cast<uint32_t>(atom)<<8)));
        return offset;
    }""")
if v.endswith("w"):  # force 4 waves/SIMD (128 VGPRs) on the nobarrier kernel
    v = v[:-1]
    src = src.replace("__global__ __launch_bounds__(512,1) void glm53_gate_decode_nobarrier_kernel(",
                      "__global__ __attribute__((amdgpu_flat_work_group_size(512,512),amdgpu_waves_per_eu(4,4))) void glm53_gate_decode_nobarrier_kernel(")
nb = src.rindex("// Decode gate variant that keeps weights in registers", 0, src.index("void glm53_gate_decode_nobarrier_kernel("))
head, tail = src[:nb], src[nb:]
def sub(text, a, b):
    assert text.count(a) >= 1, a
    return text.replace(a, b, 1)
if v in ("x4", "x5"):  # contiguous chunk of the XCD's item range per CTA
    tail = sub(tail, """    for(int work=blockIdx.x;work<total;work+=gridDim.x)
    {
        const int index=glm53_remap8(work,total);""", """    const int xcd=blockIdx.x%8,ranks=gridDim.x/8,rank=blockIdx.x/8;
    const int per=(total+7)/8,tall=total%8==0?8:total%8;
    const int start=xcd<tall?xcd*per:tall*per+(xcd-tall)*(per-1);
    const int length=xcd<tall?per:per-1;
    const int lo=start+length*rank/ranks,hi=start+length*(rank+1)/ranks;
    for(int index=lo;index<hi;++index)
    {""")
if v == "x6":  # gate codebook copies built in-kernel from the two base books (copy 0 of each)
    helper = """
// x6: E4M3(book * 2^(j/4)) for j = 1..3. Thread t owns base-book dwords t + 512 i (i < 4).
__device__ __forceinline__ void x6_write_copies(uint64_t* lds,const uint32_t (&words)[4],int thread)
{
    uint32_t* out=reinterpret_cast<uint32_t*>(lds);
#pragma unroll
    for(int i=0;i<4;++i) {
        const int d=thread+i*512,at=(d/1024)*4096+d%1024;
        const auto lo=__builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(words[i]),false);
        const auto hi=__builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(words[i]),true);
        out[at]=words[i];
        constexpr float m[3]={0x1.306fe0p+0f,0x1.6a09e6p+0f,0x1.ae89fap+0f};
#pragma unroll
        for(int j=0;j<3;++j) {
            int bits=__builtin_amdgcn_cvt_pk_fp8_f32(lo[0]*m[j],lo[1]*m[j],0,false);
            bits=__builtin_amdgcn_cvt_pk_fp8_f32(hi[0]*m[j],hi[1]*m[j],bits,true);
            out[at+(j+1)*1024]=static_cast<uint32_t>(bits);
        }
    }
}

"""
    tail = helper + tail
    tail = sub(tail, "if(new_expert) iq2r_stage_codebook_async<BookWords,8>(shared.codebook,aux,wave,lane);",
"""const int thread=wave*64+lane;
        uint32_t book_words[4]={};
        if constexpr(Classes>0) { if(new_expert) {
#pragma unroll
            for(int i=0;i<4;++i) { const int d=thread+i*512; book_words[i]=reinterpret_cast<const uint32_t*>(aux)[(d/1024)*4096+d%1024]; } } }
        else if(new_expert) iq2r_stage_codebook_async<BookWords,8>(shared.codebook,aux,wave,lane);""")
    tail = sub(tail, """        if(new_expert) {
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");""", """        if(new_expert) {
            if constexpr(Classes>0) x6_write_copies(shared.codebook,book_words,thread);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");""")
if v.startswith("x8"):  # activations, scales and d one K128 ahead
    old = """        for(int iteration=0;iteration<6;++iteration)
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
            }"""
    new = """        IQ2RActivationFragment a_next[MAtoms]={};uint32_t sa_next[MAtoms]={};uint32_t d_next=0;
        auto load_activations=[&](int iteration)
        {
            const int kt=wave*6+iteration;
#pragma unroll
            for(int m=0;m<MAtoms;++m)
            {
                if(m>=active_m) continue;
                const auto* input=reinterpret_cast<const uint8_t*>(activations+static_cast<int64_t>(rows[m])*K);
                const int ak=kt*128+lane_group*16;
                *reinterpret_cast<uint4*>(a_next[m].bytes)=*reinterpret_cast<const uint4*>(input+ak);
                *reinterpret_cast<uint4*>(a_next[m].bytes+16)=*reinterpret_cast<const uint4*>(input+ak+64);
                sa_next[m]=scales[static_cast<int64_t>(rows[m])*192+kt*4+lane_group]*0x01010101u;
            }
            if constexpr(Classes>0) d_next=d_lane[iteration/2*16];
        };
        load_activations(0);
        for(int iteration=0;iteration<6;++iteration)
        {
            IQ2RActivationFragment a[MAtoms];uint32_t sa[MAtoms];
#pragma unroll
            for(int m=0;m<MAtoms;++m) { a[m]=a_next[m]; sa[m]=sa_next[m]; }
            if constexpr(Classes>0) bases=d_next;
            const IQ2RCompressedQuad compressed=pending;
            if(iteration<5) {
                base+=kQuadBytes;
                pending=iq2r_scheduled_load_quad(data,data_bytes,base,lane);
                load_activations(iteration+1);
            }"""
    tail = sub(tail, old, new)
if v == "x5":
    v = "x2"
if v == "x1":  # stage 8 KB of the 32 KB codebook
    tail = sub(tail, "iq2r_stage_codebook_async<BookWords,8>", "iq2r_stage_codebook_async<(Classes>0?1024:BookWords),8>")
if v == "x2":  # every codeword from copy 0 of book 0
    head = sub(head, "const uint32_t copy0=iq2r_copy_offset<Classes>(levels.low,atom);",
               "const uint32_t copy0=0u*iq2r_copy_offset<Classes>(levels.low,atom);")
    head = sub(head, "const uint32_t copy1=iq2r_copy_offset<Classes>(levels.high,atom);",
               "const uint32_t copy1=0u*iq2r_copy_offset<Classes>(levels.high,atom);")
if v == "x3":  # no d loads
    tail = sub(tail, "if constexpr(Classes>0) bases=d_lane[iteration/2*16];", "if constexpr(Classes>0) bases=0x70707070u;")
open("/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu", "w").write(head + tail)
print("wrote", v)
