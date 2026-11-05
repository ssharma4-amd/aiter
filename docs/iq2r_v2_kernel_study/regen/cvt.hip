#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
// out[c*256+b] = E4M3(fp32(e4m3 b) * m[c]) via the hardware pk conversions
__global__ void k(uint8_t* out,const float* m)
{
    const int b=threadIdx.x;  // 256 threads
    const int word=b|(b<<8)|(b<<16)|(b<<24);
    for(int c=0;c<4;++c)
    {
        auto lo=__builtin_amdgcn_cvt_pk_f32_fp8(word,false);
        auto hi=__builtin_amdgcn_cvt_pk_f32_fp8(word,true);
        const float mc=m[c];
        typedef short __attribute__((ext_vector_type(2))) s2;
        s2 r0=__builtin_amdgcn_cvt_scalef32_pk_fp8_f32(s2{0,0},lo[0]*mc,lo[1]*mc,1.0f,false);
        r0=__builtin_amdgcn_cvt_scalef32_pk_fp8_f32(r0,hi[0]*mc,hi[1]*mc,1.0f,true);
        int r=__builtin_bit_cast(int,r0);
        out[c*256+b]=static_cast<uint8_t>(r>>((b&3)*8));
    }
}
int main()
{
    float hm[4];for(int c=0;c<4;++c)hm[c]=static_cast<float>(std::pow(2.0,c/4.0));
    uint8_t* d;float* dm;hipMalloc(&d,1024);hipMalloc(&dm,16);hipMemcpy(dm,hm,16,hipMemcpyHostToDevice);
    k<<<1,256>>>(d,dm);uint8_t h[1024];hipMemcpy(h,d,1024,hipMemcpyDeviceToHost);
    FILE* f=fopen("cvt.bin","wb");fwrite(h,1,1024,f);fclose(f);printf("ok\n");
}
