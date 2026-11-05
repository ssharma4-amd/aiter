#!/usr/bin/env bash
# usage: xrun.sh VAR... -> x_VAR_v2_{512,256}.log: v2 gate of each mkvar.py variant in the exp tree (HIP 3 and 5)
cd "$(dirname "$0")"
for V in "$@"; do
  python3 mkvar.py $V || exit 1
  rm -f ../../splits/aiter-v2-exp/aiter/jit/module_iq2r_moe.so
  GPU=3 TREE=/workspace/splits/aiter-v2-exp ./optest.sh x_${V}_build 'STAGES=gate python3 $D/vbench.py v2 512 2' || exit 1
  GPU=3 TREE=/workspace/splits/aiter-v2-exp ./optest.sh x_${V}_v2_512 'STAGES=gate CHECK=0 python3 $D/vbench.py v2 512 2,8,32,128,256' &
  GPU=5 TREE=/workspace/splits/aiter-v2-exp ./optest.sh x_${V}_v2_256 'STAGES=gate CHECK=0 python3 $D/vbench.py v2 256 2,8,32,128,512' &
  wait
  if [ -n "$V1" ]; then
    GPU=3 TREE=/workspace/splits/aiter-v2-exp ./optest.sh x_${V}_v1_512 'STAGES=gate CHECK=0 python3 $D/vbench.py v1 512 2,8,32,128,256' &
    GPU=5 TREE=/workspace/splits/aiter-v2-exp ./optest.sh x_${V}_v1_256 'STAGES=gate CHECK=0 python3 $D/vbench.py v1 256 2,8,32,128,512' &
    wait
  fi
done
