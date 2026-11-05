#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
typedef float f2 __attribute__((ext_vector_type(2)));
__constant__ uint32_t kD[3][2];
template<int V>
__device__ __forceinline__ void form(uint64_t* lds,const uint4 (&base)[4],int lane)
{
    const uint32_t m_bits[4]={0x3f800000u,0x3f9837f0u,0x3fb504f3u,0x3fd744fdu};
    uint4* out=reinterpret_cast<uint4*>(lds);
#pragma unroll
    for(int i=0;i<4;++i)
    {
        uint32_t w[4]={base[i].x,base[i].y,base[i].z,base[i].w};
        asm volatile("" : "+v"(w[0]),"+v"(w[1]),"+v"(w[2]),"+v"(w[3]));
        out[i*64+lane]=make_uint4(w[0],w[1],w[2],w[3]);
        if constexpr(V==0)
        {
            f2 v[8];
#pragma unroll
            for(int d=0;d<4;++d){v[2*d]=__builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w[d]),false);v[2*d+1]=__builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w[d]),true);}
            uint32_t r[4]={w[0],w[1],w[2],w[3]};
#pragma unroll
            for(int c=1;c<4;++c)
            {
                const float mc=__builtin_bit_cast(float,m_bits[c]);const f2 m2={mc,mc};
#pragma unroll
                for(int d=0;d<4;++d)
                {
                    const f2 lo=v[2*d]*m2,hi=v[2*d+1]*m2;
                    int x=__builtin_amdgcn_cvt_pk_fp8_f32(lo[0],lo[1],static_cast<int>(r[d]),false);
                    r[d]=static_cast<uint32_t>(__builtin_amdgcn_cvt_pk_fp8_f32(hi[0],hi[1],x,true));
                }
                out[c*256+i*64+lane]=make_uint4(r[0],r[1],r[2],r[3]);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        else if constexpr(V==2)
        {
#pragma unroll
            for(int c=1;c<4;++c)out[c*256+i*64+lane]=make_uint4(w[0]+c,w[1],w[2],w[3]);
        }
        else if constexpr(V==3)
        {
            uint32_t m[4];
#pragma unroll
            for(int d=0;d<4;++d)m[d]=w[d]&0x07070707u;
            out[256+i*64+lane]=make_uint4(w[0]+0x02020202u,w[1]+0x02020202u,w[2]+0x02020202u,w[3]+0x02020202u);
            out[512+i*64+lane]=make_uint4(w[0]+__builtin_amdgcn_perm(0x04040404u,0x05040403u,m[0]),w[1]+__builtin_amdgcn_perm(0x04040404u,0x05040403u,m[1]),
                                          w[2]+__builtin_amdgcn_perm(0x04040404u,0x05040403u,m[2]),w[3]+__builtin_amdgcn_perm(0x04040404u,0x05040403u,m[3]));
            out[768+i*64+lane]=make_uint4(w[0]+__builtin_amdgcn_perm(0x06060606u,0x06060605u,m[0]),w[1]+__builtin_amdgcn_perm(0x06060606u,0x06060605u,m[1]),
                                          w[2]+__builtin_amdgcn_perm(0x06060606u,0x06060605u,m[2]),w[3]+__builtin_amdgcn_perm(0x06060606u,0x06060605u,m[3]));
        }
        else
        {
            uint32_t m[4];
#pragma unroll
            for(int d=0;d<4;++d)m[d]=w[d]&0x07070707u;
#pragma unroll
            for(int c=1;c<4;++c)
            {
                uint32_t r[4];
#pragma unroll
                for(int d=0;d<4;++d)r[d]=w[d]+__builtin_amdgcn_perm(kD[c-1][1],kD[c-1][0],m[d]);
                out[c*256+i*64+lane]=make_uint4(r[0],r[1],r[2],r[3]);
            }
        }
    }
}
template<int V>
__global__ __launch_bounds__(576) void k(const uint4* books,uint8_t* res,uint64_t* t,int reps)
{
    __shared__ uint64_t lds[9][2048];
    const int lane=threadIdx.x%64,wv=threadIdx.x/64;
    uint4 base[4];
    for(int i=0;i<4;++i)base[i]=books[wv*256+i*64+lane];
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    const uint64_t t0=__builtin_amdgcn_s_memtime();
    for(int r=0;r<reps;++r){form<V>(lds[wv],base,lane);asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");}
    const uint64_t t1=__builtin_amdgcn_s_memtime();
    for(int j=lane;j<2048;j+=64)reinterpret_cast<uint64_t*>(res)[wv*2048+j]=lds[wv][j];
    if(lane==0)t[wv]=t1-t0;
}
static uint8_t e4m3(float x)  // RNE, saturate (torch semantics) -- positives only here
{
    uint8_t best=0;float bd=1e30f;
    for(int b=0;b<0x7f;++b){int e=(b>>3)&15,mm=b&7;float v=e?std::ldexp(1.0f+mm/8.0f,e-7):std::ldexp(mm/8.0f,-6);
        float d=std::fabs(v-x);if(d<bd||(d==bd&&(b&1)==0)){bd=d;best=b;}}
    return best;
}
static float dec(uint8_t b){int e=(b>>3)&15,mm=b&7;float v=e?std::ldexp(1.0f+mm/8.0f,e-7):std::ldexp(mm/8.0f,-6);return b&0x80?-v:v;}
int main(int argc,char** argv)
{
    const float mul[4]={1.0f,__builtin_bit_cast(float,0x3f9837f0u),__builtin_bit_cast(float,0x3fb504f3u),__builtin_bit_cast(float,0x3fd744fdu)};
    uint32_t D[3][2];
    for(int c=1;c<4;++c){uint8_t t[8];for(int mm=0;mm<8;++mm){uint8_t b=(7<<3)|mm;t[mm]=e4m3(dec(b)*mul[c])-b;}
        memcpy(D[c-1],t,8);}
    // which normal bytes does the table get right?
    int bad=0;for(int c=1;c<4;++c)for(int b=0;b<0x80;++b){uint8_t want=e4m3(dec(b)*mul[c]);uint8_t got=b+((uint8_t*)D[c-1])[b&7];if(want!=got){if(bad<12)printf("table miss c=%d b=0x%02x want 0x%02x got 0x%02x\n",c,b,want,got);++bad;}}
    printf("table misses (positive bytes 0..0x7f) %d\n",bad);
    hipMemcpyToSymbol(HIP_SYMBOL(kD),D,sizeof(D));
    // 9 books of random normal bytes in [0x08, 0x5f]
    const int n=9*4096;uint8_t* h=new uint8_t[n];uint32_t s=1;for(int i=0;i<n;++i){s=s*1664525u+1013904223u;h[i]=0x08+(s>>24)%(0x60-0x08);}
    uint4* db;uint8_t* dr[2];uint64_t* dt;hipMalloc(&db,n);hipMemcpy(db,h,n,hipMemcpyHostToDevice);hipMalloc(&dr[0],9*16384);hipMalloc(&dr[1],9*16384);hipMalloc(&dt,64*8);
    uint8_t* dr2[2];hipMalloc(&dr2[0],9*16384);hipMalloc(&dr2[1],9*16384);
    for(int waves:{1,4,9})for(int v=0;v<4;++v){
        uint8_t* o=v<2?dr[v]:dr2[v-2];
        for(int it=0;it<2;++it){if(v==1)k<1><<<1,64*waves>>>(db,o,dt,8);else if(v==0)k<0><<<1,64*waves>>>(db,o,dt,8);else if(v==2)k<2><<<1,64*waves>>>(db,o,dt,8);else k<3><<<1,64*waves>>>(db,o,dt,8);hipDeviceSynchronize();}
        uint64_t ht[9];hipMemcpy(ht,dt,8*waves,hipMemcpyDeviceToHost);
        printf("V=%d waves=%d memtime cycles per formation: wave0 %.0f wave%d %.0f\n",v,waves,ht[0]/8.0,waves-1,ht[waves-1]/8.0);}
    uint8_t* r0=new uint8_t[9*16384];uint8_t* r1=new uint8_t[9*16384];hipMemcpy(r0,dr[0],9*16384,hipMemcpyDeviceToHost);hipMemcpy(r1,dr[1],9*16384,hipMemcpyDeviceToHost);
    int diff=0,ref=0;for(int w=0;w<9;++w)for(int c=0;c<4;++c)for(int j=0;j<4096;++j){uint8_t want=e4m3(dec(h[w*4096+j])*mul[c]);
        if(r0[w*16384+c*4096+j]!=want)++ref;if(r1[w*16384+c*4096+j]!=want)++diff;}
    printf("mismatch vs host RNE: hw-cvt %d perm %d\n",ref,diff);
    hipMemcpy(r1,dr2[1],9*16384,hipMemcpyDeviceToHost);diff=0;
    for(int w=0;w<9;++w)for(int c=0;c<4;++c)for(int j=0;j<4096;++j){uint8_t want=e4m3(dec(h[w*4096+j])*mul[c]);if(r1[w*16384+c*4096+j]!=want)++diff;}
    printf("mismatch vs host RNE: add/perm2 %d\n",diff);
}
