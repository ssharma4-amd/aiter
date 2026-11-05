# Device assembly of TREE's iq2r_gemm_gfx950.cu into $OUT (build flags from the JIT build.ninja).
T=${TREE:-/workspace/splits/aiter-iq2r-v2}; OUT=${OUT:-/workspace/experiments/iq2r-v2-20261103/iq2r.s}
B=$T/aiter/jit/build/module_iq2r_moe/build
flags=$(sed -n 's/^cuda_cflags = //p' $B/build.ninja)
post=$(sed -n 's/^cuda_post_cflags = //p' $B/build.ninja)
cd $B && eval hipcc $flags -c $T/csrc/kernels/iq2r/iq2r_gemm_gfx950.cu -o $OUT $post --cuda-device-only -S
