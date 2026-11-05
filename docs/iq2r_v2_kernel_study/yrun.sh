#!/usr/bin/env bash
# usage: [NOV1=1] yrun.sh NAME BOOKS DOUBLE [NOCOPY NOD] -> NAME_v2_{512,256}.log (exp tree, HIP 3,5) + NAME_v1m_{512,256}.log (main v1, HIP 6,7)
N=$1; cd "$(dirname "$0")"
F=../../splits/aiter-v2-exp/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu
sed -i -E "s/^#define GLM53_GATE_BOOKS [0-9]+/#define GLM53_GATE_BOOKS $2/; s/^#define GLM53_GATE_DOUBLE [0-9]+/#define GLM53_GATE_DOUBLE $3/; s/^#define GLM53_X_NOCOPY [0-9]+/#define GLM53_X_NOCOPY ${4:-0}/; s/^#define GLM53_X_NOD [0-9]+/#define GLM53_X_NOD ${5:-0}/; s/^#define GLM53_X_ORDER [0-9]+/#define GLM53_X_ORDER ${6:-0}/; s/^#define GLM53_X_DPRE [0-9]+/#define GLM53_X_DPRE ${7:-0}/" $F
grep -E "^#define GLM53_(GATE_BOOKS|GATE_DOUBLE|X_)" $F | tr '\n' ' '; echo
X=/workspace/splits/aiter-v2-exp M=/workspace/splits/aiter-main-v1
rm -f ../../splits/aiter-v2-exp/aiter/jit/module_iq2r_moe.so
GPU=3 TREE=$X ./optest.sh ${N}_build "STAGES=gate python3 \$D/vbench.py v2 512 2" || exit 1
GPU=3 TREE=$X ./optest.sh ${N}_v2_512 "STAGES=gate CHECK=0 python3 \$D/vbench.py v2 512 2,4,8,16,32,64,128,256" &
GPU=5 TREE=$X ./optest.sh ${N}_v2_256 "STAGES=gate CHECK=0 python3 \$D/vbench.py v2 256 2,4,8,16,32,64,128,256,512" &
if [ -z "$NOV1" ]; then
GPU=6 TREE=$M ./optest.sh ${N}_v1m_512 "STAGES=gate CHECK=0 python3 \$D/vbench.py v1 512 2,4,8,16,32,64,128,256" &
GPU=7 TREE=$M ./optest.sh ${N}_v1m_256 "STAGES=gate CHECK=0 python3 \$D/vbench.py v1 256 2,4,8,16,32,64,128,256,512" &
fi
wait
