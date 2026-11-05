#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
// per-wave VALU throughput: 8 independent ops x 64 unrolled x iters, cycles via s_memtime
#define REP8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
template<int Op>
__global__ void k(uint64_t* out,uint32_t seed,int iters)
{
    uint32_t a[8];float2 f[8];
    for(int i=0;i<8;++i){a[i]=seed*(threadIdx.x+i);f[i]=make_float2(a[i]*1e-9f,i);}
    const uint64_t t0=__builtin_amdgcn_s_memtime();
    for(int it=0;it<iters;++it)
    {
#pragma unroll
        for(int u=0;u<16;++u)
        {
            if constexpr(Op==0){
#define X(i) asm volatile("v_cvt_pk_f32_fp8 %0, %1" : "=v"(f[i]) : "v"(a[i]));
                REP8(X)
#undef X
            }else if constexpr(Op==1){
#define X(i) asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2" : "+v"(a[i]) : "v"(f[i].x),"v"(f[i].y));
                REP8(X)
#undef X
            }else if constexpr(Op==2){
#define X(i) asm volatile("v_pk_mul_f32 %0, %0, %0" : "+v"(f[i]));
                REP8(X)
#undef X
            }else if constexpr(Op==3){
#define X(i) asm volatile("v_perm_b32 %0, %0, %1, %0" : "+v"(a[i]) : "v"(a[(i+1)&7]));
                REP8(X)
#undef X
            }else if constexpr(Op==4){
#define X(i) asm volatile("v_add_u32 %0, %0, %1" : "+v"(a[i]) : "v"(a[(i+1)&7]));
                REP8(X)
#undef X
            }else if constexpr(Op==5){
#define X(i) asm volatile("v_mul_f32 %0, %0, %0" : "+v"(f[i].x));
                REP8(X)
#undef X
            }
        }
    }
    const uint64_t t1=__builtin_amdgcn_s_memtime();
    uint32_t s=0;for(int i=0;i<8;++i)s+=a[i]+__float_as_uint(f[i].x)+__float_as_uint(f[i].y);
    if(threadIdx.x%64==0)out[threadIdx.x/64]=t1-t0;
    if(s==0x12345678)out[100]=s;
}
template<int Op> void run(const char* name,int waves)
{
    uint64_t* d;hipMalloc(&d,1024*8);int iters=200;
    k<Op><<<1,64*waves>>>(d,3,iters);hipDeviceSynchronize();
    k<Op><<<1,64*waves>>>(d,3,iters);hipDeviceSynchronize();
    uint64_t h[16];hipMemcpy(h,d,8*waves,hipMemcpyDeviceToHost);
    printf("%-18s waves=%d memtime cycles/instr/wave: %.2f\n",name,waves,(double)h[0]/(iters*16*8));
    hipFree(d);
}
int main()
{
    for(int w:{1,4,8}){
    run<0>("cvt_pk_f32_fp8",w);run<1>("cvt_pk_fp8_f32",w);run<2>("pk_mul_f32",w);
    run<3>("perm_b32",w);run<4>("add_u32",w);run<5>("mul_f32",w);}
}
