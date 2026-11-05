#!/usr/bin/env bash
# usage (host): isa2.sh NAME  -> NAME.s from $TREE (default v2 tree), then occupancy + wait summary
N=$1; T=${TREE:-/workspace/splits/aiter-iq2r-v2}
docker exec codex-glm53-iq2r-tp4-20260921 bash -c "TREE=$T OUT=/workspace/experiments/iq2r-v2-20261103/$N.s bash /workspace/experiments/iq2r-v2-20261103/isa.sh" > $N.isa.log 2>&1 || { tail $N.isa.log; exit 1; }
cd "$(dirname "$0")"; python3 occ.py $N.s > $N.occ; python3 waits.py $N.s > $N.waits
