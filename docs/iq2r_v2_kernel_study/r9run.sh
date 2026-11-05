#!/usr/bin/env bash
# usage: r9run.sh NAME R9T [R9W R9ROT] -> NAME_v2_{512,256}.log (exp tree route9 down, HIP 3,5)
N=$1; cd "$(dirname "$0")"
F=../../splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu
sed -i -E "s/^#define GLM53_X_R9T [0-9]+/#define GLM53_X_R9T $2/; s/^#define GLM53_X_R9W [0-9]+/#define GLM53_X_R9W ${3:-0}/; s/^#define GLM53_X_R9ROT [0-9]+/#define GLM53_X_R9ROT ${4:-0}/; s/^#define GLM53_X_R9VG [0-9]+/#define GLM53_X_R9VG ${5:-0}/; s/^#define GLM53_X_R9ND [0-9]+/#define GLM53_X_R9ND ${6:-0}/; s/^#define GLM53_X_R9SLP [0-9]+/#define GLM53_X_R9SLP ${7:-0}/" $F
grep -E "^#define GLM53_X_R9" $F | tr '\n' ' '; echo
X=/workspace/splits/aiter-v2-exp
rm -f ../../splits/aiter-v2-exp/aiter/jit/module_iq2r_moe.so
GPU=3 TREE=$X ./optest.sh ${N}_build "STAGES=down python3 \$D/vbench.py v2 512 1,2,4 && STAGES=down python3 \$D/vbench.py v2 256 1,2,4"
grep -q "^v2 I=256 M=    4" ${N}_build.log || { grep -m3 " error" ${N}_build.log; exit 1; }
grep "^v2" ${N}_build.log
GPU=3 TREE=$X ./optest.sh ${N}_v2_512 "STAGES=down CHECK=0 python3 \$D/vbench.py v2 512 1,2,4" &
GPU=5 TREE=$X ./optest.sh ${N}_v2_256 "STAGES=down CHECK=0 python3 \$D/vbench.py v2 256 1,2,4" &
wait; grep -h "^v2" ${N}_v2_512.log ${N}_v2_256.log
