#!/usr/bin/env bash
# usage: stg.sh NAME STG ORD -> route9 run with staggered book DMA
F=../../splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu; cd "$(dirname "$0")"
sed -i -E "s/^#define GLM53_X_R9STG [0-9]+/#define GLM53_X_R9STG $2/; s/^#define GLM53_X_R9ORD [0-9]+/#define GLM53_X_R9ORD $3/" $F
./r9run.sh $1 4 >/dev/null 2>&1
