#!/usr/bin/env bash
# usage (host): GPU=3 optest.sh LOGNAME CMD...   -- runs CMD in the container with env.sh
D=/workspace/experiments/iq2r-v2-20261103
L=$1; shift
docker exec -e GPU=${GPU:-3} -e TREE=${TREE:-/workspace/splits/aiter-iq2r-v2} codex-glm53-iq2r-tp4-20260921 \
  bash -c "D=$D; source $D/env.sh; $* " > /home/ssharma4/2bit-support-upstream/GLM5.3_Integration/experiments/iq2r-v2-20261103/$L.log 2>&1
echo "rc=$? $L"
