# Source inside codex-glm53-iq2r-tp4-20260921. Local GPU rules: HIP 3,5,6,7 only, no core dumps.
for v in $(compgen -e); do
  case $v in AITER_*|ATOM_*|PYTHONSAFEPATH) unset "$v" ;; esac
done
export HIP_VISIBLE_DEVICES=${GPU:-3} GPU_ARCHS=gfx950 MAX_JOBS=64
export TREE=${TREE:-/workspace/splits/aiter-iq2r-v2}
export PYTHONPATH=$TREE:$TREE/op_tests PYTHONDONTWRITEBYTECODE=1
export AITER_MLA_USE_OPUS=1 AITER_DISABLE_FLYDSL_TOPK_DECODE=1 AITER_AR_1STAGE=1 \
  AITER_CUSTOM_AR_MAX_SIZE=131072 AITER_AR_QUANT_MAX_BYTES=131072
export CCACHE_DIR=/ccache
ulimit -c 0
cd $TREE
