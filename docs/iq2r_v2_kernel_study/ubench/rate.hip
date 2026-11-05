// Issue rate of single VALU ops on gfx950: 16 independent ops per loop trip.
#include <hip/hip_runtime.h>
#include <cstdio>
#define OP16(...) __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ \
                  __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ __VA_ARGS__ __VA_ARGS__
template<int K> __global__ __launch_bounds__(512) void k(unsigned* out,int n)
{
    unsigned a=threadIdx.x, b=a*3+1, c=a^7, d=a+5, e=a*7, f=a+11, g=a^0x55, h=a*13;
    for(int i=0;i<n;++i) {
        if constexpr(K==0) asm volatile(OP16("v_lshlrev_b32 %0, 3, %0\n v_lshlrev_b32 %1, 3, %1\n") : "+v"(a),"+v"(b));
        if constexpr(K==1) { unsigned long long x=((unsigned long long)b<<32)|a, y=((unsigned long long)d<<32)|c;
            asm volatile(OP16("v_lshlrev_b64 %0, 3, %0\n v_lshlrev_b64 %1, 3, %1\n") : "+v"(x),"+v"(y)); a=x; b=x>>32; c=y; d=y>>32; }
        if constexpr(K==2) asm volatile(OP16("v_and_or_b32 %0, %0, %2, %1\n v_and_or_b32 %1, %1, %2, %0\n") : "+v"(a),"+v"(b) : "s"(0x80808080u));
        if constexpr(K==3) asm volatile(OP16("v_perm_b32 %0, 0, %0, %2\n v_perm_b32 %1, 0, %1, %2\n") : "+v"(a),"+v"(b) : "v"(c));
        if constexpr(K==4) asm volatile(OP16("v_permlane16_swap_b32 %0, %1\n v_permlane16_swap_b32 %2, %3\n") : "+v"(a),"+v"(b),"+v"(c),"+v"(d));
        if constexpr(K==5) asm volatile(OP16("v_bfe_u32 %0, %0, 3, 9\n v_bfe_u32 %1, %1, 3, 9\n") : "+v"(a),"+v"(b));
        if constexpr(K==6) asm volatile(OP16("v_lshl_or_b32 %0, %0, 3, %1\n v_lshl_or_b32 %1, %1, 3, %0\n") : "+v"(a),"+v"(b));
        if constexpr(K==7) asm volatile(OP16("v_mov_b32 %0, %1\n v_mov_b32 %1, %0\n") : "+v"(a),"+v"(b));
        if constexpr(K==8) { unsigned long long x=((unsigned long long)b<<32)|a, y=((unsigned long long)d<<32)|c;
            asm volatile(OP16("v_lshl_add_u64 %0, %0, 2, %1\n v_lshl_add_u64 %1, %1, 2, %0\n") : "+v"(x),"+v"(y)); a=x; b=x>>32; c=y; d=y>>32; }
        if constexpr(K==9) { unsigned long long x=((unsigned long long)b<<32)|a, y=((unsigned long long)d<<32)|c;
            asm volatile(OP16("v_lshrrev_b64 %0, 3, %0\n v_lshrrev_b64 %1, 3, %1\n") : "+v"(x),"+v"(y)); a=x; b=x>>32; c=y; d=y>>32; }
    }
    out[blockIdx.x*512+threadIdx.x]=a+b+c+d+e+f+g+h;
}
template<int K> float run(unsigned* o,int n,const char* name)
{
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    k<K><<<256*8,512>>>(o,n); hipDeviceSynchronize();
    hipEventRecord(s); k<K><<<256*8,512>>>(o,n); hipEventRecord(e); hipEventSynchronize(e);
    float ms; hipEventElapsedTime(&ms,s,e);
    // wave-instructions issued per SIMD: 2048 CTAs * 8 waves * n * 32 ops / (256 CUs * 4 SIMDs)
    double per_simd=2048.0*8*n*32/(256*4);
    printf("%-22s %8.3f ms  %6.2f cycles/op (at 2.4 GHz)\n",name,ms,ms*1e-3*2.4e9/per_simd);
    return ms;
}
int main(){ unsigned* o; hipMalloc(&o,256*8*512*4); int n=4000;
 run<7>(o,n,"v_mov_b32"); run<0>(o,n,"v_lshlrev_b32"); run<1>(o,n,"v_lshlrev_b64"); run<9>(o,n,"v_lshrrev_b64");
 run<8>(o,n,"v_lshl_add_u64"); run<2>(o,n,"v_and_or_b32"); run<3>(o,n,"v_perm_b32");
 run<4>(o,n,"v_permlane16_swap"); run<5>(o,n,"v_bfe_u32"); run<6>(o,n,"v_lshl_or_b32"); }
