#!/usr/bin/env bash
# usage: vab.sh NAME TREEDIR "MACRO=V ..." GPU_A GPU_B [MS512] [MS256]
#   sets GLM53_X_<MACRO> in TREEDIR's kernel, rebuilds, runs STAGES (default gate) v2 at I=512 on GPU_A and I=256 on GPU_B.
N=$1; T=$2; SET=$3; GA=$4; GB=$5; A=${6:-1,2,4,8,16,32,64,128,256}; B=${7:-1,2,4,8,16,32,64,128,256,512}; S=${STAGES-gate}
cd "$(dirname "$0")"
H=/home/ssharma4/2bit-support-upstream/GLM5.3_Integration/splits/$T; F=$H/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu
for kv in $SET; do k=${kv%=*}; v=${kv#*=}; grep -q "^#define GLM53_X_$k " $F || { echo "no macro $k"; exit 1; }; sed -i "s/^#define GLM53_X_$k .*/#define GLM53_X_$k $v/" $F; done
rm -f $H/aiter/jit/module_iq2r_moe.so
X=/workspace/splits/$T
GPU=$GA TREE=$X ./optest.sh ${N}_build "STAGES=gate python3 \$D/vbench.py v2 512 2" || exit 1
GPU=$GA TREE=$X ./optest.sh ${N}_v2_512 "STAGES=$S python3 \$D/vbench.py v2 512 $A" &
GPU=$GB TREE=$X ./optest.sh ${N}_v2_256 "STAGES=$S python3 \$D/vbench.py v2 256 $B" &
wait
