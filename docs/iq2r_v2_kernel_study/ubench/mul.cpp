// VALU rate probe: 8 independent chains of one instruction kind per lane.
#include <hip/hip_runtime.h>
#include <cstdio>
template<int Kind>
__global__ void probe(unsigned* out, unsigned seed, int iters)
{
    unsigned a[8];
    for(int i=0;i<8;++i)a[i]=seed+threadIdx.x*8+i;
    for(int it=0;it<iters;++it)
    {
#pragma unroll
        for(int i=0;i<8;++i)
        {
            if constexpr(Kind==0) asm volatile("v_mul_lo_u32 %0, %0, %1" : "+v"(a[i]) : "s"(0x10204081u));
            else if constexpr(Kind==1) asm volatile("v_mul_u32_u24 %0, %1, %0" : "+v"(a[i]) : "s"(0x204081u));
            else if constexpr(Kind==2) asm volatile("v_and_or_b32 %0, %0, %1, %0" : "+v"(a[i]) : "s"(0x80808080u));
            else if constexpr(Kind==3) asm volatile("v_bfe_u32 %0, %0, 4, 4" : "+v"(a[i]));
            else if constexpr(Kind==4) asm volatile("v_perm_b32 %0, %0, %0, %1" : "+v"(a[i]) : "s"(0x0c040c00u));
        }
    }
    unsigned s=0;for(int i=0;i<8;++i)s^=a[i];
    if(s==0x12345678)out[0]=s;
}
template<int Kind> float run(unsigned* out,int iters)
{
    hipEvent_t a,b;hipEventCreate(&a);hipEventCreate(&b);
    probe<Kind><<<256*4,256>>>(out,1,iters);
    hipEventRecord(a);
    probe<Kind><<<256*4,256>>>(out,1,iters);
    hipEventRecord(b);hipEventSynchronize(b);
    float ms;hipEventElapsedTime(&ms,a,b);return ms;
}
int main()
{
    unsigned* out;hipMalloc(&out,4);
    const int iters=4096;
    const char* names[]={"v_mul_lo_u32","v_mul_u32_u24","v_and_or_b32","v_bfe_u32","v_perm_b32"};
    float t[5]={run<0>(out,iters),run<1>(out,iters),run<2>(out,iters),run<3>(out,iters),run<4>(out,iters)};
    // waves: 1024 blocks * 4 waves; per wave iters*8 instr
    for(int k=0;k<5;++k)
    {
        const double instr_per_simd=1024.0*4*iters*8/(256*4);
        printf("%-16s %.3f ms  %.2f ns/instr/SIMD\n",names[k],t[k],t[k]*1e6/instr_per_simd);
    }
}
