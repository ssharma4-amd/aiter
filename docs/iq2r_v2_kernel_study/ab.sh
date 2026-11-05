#!/usr/bin/env bash
# usage: ab.sh TAG TREE [MS512] [MS256]  -> TAG_{v1,v2}_{512,256}.log, the four runs concurrently on HIP 3,5,6,7 (gate only unless STAGES set)
TAG=$1; T=$2; A=${3:-2,4,8,16,32,64,128,256}; B=${4:-2,4,8,16,32,64,128,256,512}; S=${STAGES-gate}
cd "$(dirname "$0")"
# The JIT loads an existing module .so as is: rebuild it from the tree's current source first.
rm -f "../..${T#/workspace}/aiter/jit/module_iq2r_moe.so"
GPU=3 TREE=$T ./optest.sh ${TAG}_build "STAGES=gate python3 \$D/vbench.py v2 512 2" || exit 1
GPU=3 TREE=$T ./optest.sh ${TAG}_v2_512 "STAGES=$S python3 \$D/vbench.py v2 512 $A" &
GPU=5 TREE=$T ./optest.sh ${TAG}_v1_512 "STAGES=$S python3 \$D/vbench.py v1 512 $A" &
GPU=6 TREE=$T ./optest.sh ${TAG}_v2_256 "STAGES=$S python3 \$D/vbench.py v2 256 $B" &
GPU=7 TREE=$T ./optest.sh ${TAG}_v1_256 "STAGES=$S python3 \$D/vbench.py v1 256 $B" &
wait
