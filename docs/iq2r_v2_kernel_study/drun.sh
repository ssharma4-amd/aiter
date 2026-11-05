#!/usr/bin/env bash
# usage: drun.sh NAME [MACRO=VAL ...] -> NAME_v2_{512,256}.log (exp tree, all down M, HIP 3,5)
N=$1; shift; cd "$(dirname "$0")"
F=../../splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu
for kv in "$@"; do k=${kv%%=*}; v=${kv#*=}; sed -i -E "s/^#define GLM53_$k [0-9]+/#define GLM53_$k $v/" $F; done
grep -E "^#define GLM53_X_" $F | sed 's/#define GLM53_X_//' | tr '\n' ' '; echo
X=/workspace/splits/aiter-v2-exp
rm -f ../../splits/aiter-v2-exp/aiter/jit/module_iq2r_moe.so
GPU=3 TREE=$X ./optest.sh ${N}_build "STAGES=down python3 \$D/vbench.py v2 512 1,2,4,8,16,32,64,128,256 && STAGES=down python3 \$D/vbench.py v2 256 1,2,4,8,16,32,64,128,256,512"
grep -q "^v2 I=256 M=  512" ${N}_build.log || { grep -m3 " error" ${N}_build.log; tail -3 ${N}_build.log; exit 1; }
grep -c "MISMATCH\|non-finite" ${N}_build.log
GPU=3 TREE=$X ./optest.sh ${N}_v2_512 "STAGES=down CHECK=0 python3 \$D/vbench.py v2 512 1,2,4,8,16,32,64,128,256" &
GPU=5 TREE=$X ./optest.sh ${N}_v2_256 "STAGES=down CHECK=0 python3 \$D/vbench.py v2 256 1,2,4,8,16,32,64,128,256,512" &
wait
