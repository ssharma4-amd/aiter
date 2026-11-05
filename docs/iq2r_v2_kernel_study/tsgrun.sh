#!/usr/bin/env bash
# usage: tsgrun.sh NAME "MACRO=V ..." [GPU] [FMT] [INTER] [MS] -- GTS=1 build in aiter-v2-exp, tsg.py, then GTS=0
N=$1; SET=$2; G=${3:-3}; FMT=${4:-v2}; I=${5:-512}; MS=${6:-16,64,256}
cd "$(dirname "$0")"
F=/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu
for kv in GTS=1 $SET; do k=${kv%=*}; v=${kv#*=}; grep -q "^#define GLM53_X_$k " $F || { echo "no macro $k"; exit 1; }; sed -i "s/^#define GLM53_X_$k .*/#define GLM53_X_$k $v/" $F; done
rm -f /home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/aiter/jit/module_iq2r_moe.so
GPU=$G TREE=/workspace/splits/aiter-v2-exp ./optest.sh ts_$N "python3 \$D/tsg.py $FMT $I $MS"
sed -i "s/^#define GLM53_X_GTS .*/#define GLM53_X_GTS 0/" $F
rm -f /home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/aiter-v2-exp/aiter/jit/module_iq2r_moe.so
grep -v "WARN\|import\|build" ts_$N.log
