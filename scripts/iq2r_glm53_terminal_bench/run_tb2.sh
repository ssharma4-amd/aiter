#!/usr/bin/env bash
# Terminal-Bench 2.0 (Harbor, terminus-2 agent) on GLM-5.3 served by ATOM with the native IQ2R-v2 kernels.
#
# Server: serve.sh arm fp8 on an IQ2R-v2 checkpoint (experts on the native IQ2R-v2 kernels, shared expert stored
#   block FP8), aiter tree AITER_TREE and ATOM tree ATOM_TREE mounted over the image's, TP8, FP8 KV, PTPC FP8
#   non-expert layers, prefix cache, --max-num-seqs 16; gated by check_server.py (log scan + 3 sanity prompts).
# Proxy: budget_proxy.py (16K think / 4K answer) between the agent and ATOM.
# Agent: Harbor terminus-2 (litellm, OpenAI chat completions) -> proxy. Task sandboxes: local Docker.
#
# usage: run_tb2.sh                      full TB2 (89 tasks)
# env:   WORK (~/iq2r_glm53_tb2): venv (harbor==0.25.0), the downloaded dataset, jobs/, logs/. Nothing is written
#          into this directory of the repo.
#        SERVER   atom (default) | stub (CPU stand-in, plumbing test) | none (use what already listens on PORT)
#        ARM (fp8; mxfp4 = MXFP4 baseline on the same stack), STORE (/models/zai-org/GLM-5.3-IQ2Rv2-final)
#        AITER_TREE, ATOM_TREE (aiter / ATOM checkouts mounted into the server; empty = the image's own)
#        GPUS (0..7), PORT (8011), PROXY_PORT (9011), BUDGET (16384), ANSWER (4096), PROXY=0 to skip the proxy
#        TASKS ("" = all 89; else space-separated task names), N_CONC (8), ATTEMPTS (1)
#        AGENT_TIMEOUT_MULT (1.0 = leaderboard timeouts), TEMP (0.95, GLM-5.3 model card), JOB (tb2-ARM-<time>)
#        BATCH (30): tasks per Harbor job. Harbor never removes the prebuilt task images it pulls (48 GB compressed
#          for all 89), so after each batch the TB2 images that batch pulled are removed (images present before
#          the run are kept). A finished batch is skipped when the same JOB is rerun.
#        MIN_FREE_GB (40): Docker-root headroom required before each batch
#        KEEP_SERVER=1 leaves server and proxy running afterwards
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK=${WORK:-$HOME/iq2r_glm53_tb2}
SERVER=${SERVER:-atom}
ARM=${ARM:-fp8}
STORE=${STORE:-/models/zai-org/GLM-5.3-IQ2Rv2-final}
AITER_TREE=${AITER_TREE-/home/ssharma4/2-bit-Upstream/GLM5.3_Integration/experiments/iq2r-v2-20261102/upstream_final/aiter-smoke}
ATOM_TREE=${ATOM_TREE-/home/ssharma4/2-bit-Upstream/GLM5.3_Integration/ATOM-v2}
IMG=${IMAGE:-glm53-ab:aiter-6dbf9e876-atom-17aca439a}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-8011}
PROXY_PORT=${PROXY_PORT:-9011}
N_CONC=${N_CONC:-8}
JOB=${JOB:-tb2-$ARM-$(date -u +%Y%m%dT%H%M)}
SNAME=tb2-$ARM-$PORT
PNAME=tb2-proxy-$PROXY_PORT
mkdir -p "$WORK/logs" "$WORK/jobs"
cd "$WORK"
log() { echo "$(date -u +%FT%TZ) $*"; }

if [[ ! -x venv/bin/harbor ]]; then
  python3 -m venv venv && venv/bin/pip install -q harbor==0.25.0 || { log "harbor install failed"; exit 1; }
fi
if [[ ! -d terminal-bench ]]; then
  venv/bin/harbor download terminal-bench@2.0 > logs/download.log 2>&1 || { log "dataset download failed"; exit 1; }
fi

cleanup() {
  [[ ${KEEP_SERVER:-0} == 1 ]] && return
  docker rm -f $PNAME > /dev/null 2>&1
  [[ $SERVER == atom ]] && docker stop $SNAME > /dev/null 2>&1
  [[ $SERVER == stub ]] && kill %1 2> /dev/null
}
trap cleanup EXIT

case $SERVER in
  atom)
    log "starting ATOM ($ARM, $STORE, GPUs $GPUS, :$PORT)"
    EXTRA="-w /tmp -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1"
    PP=""
    [[ -n $ATOM_TREE ]] && { EXTRA+=" -v $ATOM_TREE:/atom_tree:ro"; PP=/atom_tree; }
    [[ -n $AITER_TREE ]] && { EXTRA+=" -v $AITER_TREE:$AITER_TREE"; PP=${PP:+$PP:}$AITER_TREE; }
    [[ -n $PP ]] && EXTRA+=" -e PYTHONPATH=$PP"
    EXTRA_DOCKER_ARGS="$EXTRA" MAX_NUM_SEQS=16 IMAGE=$IMG LOG_DIR=logs \
      "$HERE/serve.sh" $ARM $STORE $GPUS $PORT > logs/serve-$JOB.txt 2>&1 || { log "serve.sh failed"; exit 1; }
    if ! python3 "$HERE/check_server.py" logs/$SNAME.log $PORT > logs/check-$JOB.txt 2>&1; then
      log "server gate FAILED (logs/check-$JOB.txt)"; exit 1
    fi
    log "server up | $(grep -o "'max_num_seqs': [0-9]*\|'kv_cache_dtype': '[a-z0-9]*'\|'tensor_parallel_size': [0-9]*" logs/$SNAME.log | sort -u | tr '\n' ' ')" ;;
  stub)
    python3 "$HERE/stub_server.py" --port $PORT --log logs/stub-$JOB.jsonl > logs/stub-$JOB.txt 2>&1 &
    sleep 2 ;;
  none) ;;
  *) echo "unknown SERVER $SERVER" >&2; exit 2 ;;
esac
curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null || { log "nothing answers on :$PORT"; exit 1; }

BASE=http://127.0.0.1:$PORT/v1
if [[ ${PROXY:-1} == 1 ]]; then
  docker rm -f $PNAME > /dev/null 2>&1
  docker run -d --name $PNAME --network host -v /models:/models:ro -v "$HERE:/proxy:ro" -v "$WORK/logs:/logs" \
    -e HF_HUB_OFFLINE=1 $IMG python3 /proxy/budget_proxy.py --upstream http://127.0.0.1:$PORT --port $PROXY_PORT \
    --budget ${BUDGET:-16384} --answer-tokens ${ANSWER:-4096} --inject '{}' --log /logs/proxy-$JOB.jsonl > /dev/null
  for _ in $(seq 60); do curl -sf http://127.0.0.1:$PROXY_PORT/v1/models > /dev/null && break; sleep 2; done
  curl -sf http://127.0.0.1:$PROXY_PORT/v1/models > /dev/null || { log "proxy did not come up"; docker logs $PNAME 2>&1 | tail; exit 1; }
  BASE=http://127.0.0.1:$PROXY_PORT/v1
fi

tb2_images() { docker images --format '{{.Repository}}:{{.Tag}}' | grep -xFf "$HERE/tb2_images.txt" || true; }
free_gb() { df -BG --output=avail "$(docker info --format '{{.DockerRootDir}}')" | tail -n 1 | tr -dc '0-9'; }
read -r -a ALL <<< "${TASKS:-$(ls terminal-bench | tr '\n' ' ')}"
BATCH=${BATCH:-30}
[[ -f logs/images-before-$JOB.txt ]] || tb2_images > logs/images-before-$JOB.txt
rc=0
for (( b = 0; b * BATCH < ${#ALL[@]}; b++ )); do
  J=$JOB-b$b
  [[ -s jobs/$J/result.json ]] && grep -q '"finished_at": "2' jobs/$J/result.json && { log "$J already finished"; continue; }
  (( $(free_gb) >= ${MIN_FREE_GB:-40} )) || { log "ERROR: Docker root has $(free_gb) GiB free, need ${MIN_FREE_GB:-40}"; rc=1; break; }
  INC=()
  for t in "${ALL[@]:b*BATCH:BATCH}"; do INC+=(-i "$t"); done
  log "harbor $J: terminal-bench@2.0, terminus-2 -> $BASE, $(( ${#INC[@]} / 2 )) tasks, $N_CONC concurrent, $(free_gb) GiB free"
  OPENAI_API_KEY=local venv/bin/harbor run -p terminal-bench -a terminus-2 -m openai/zai-org/GLM-5.3 \
    --ak api_base=$BASE --ak temperature=${TEMP:-0.95} \
    --ak 'model_info={"max_input_tokens": 245760, "max_output_tokens": 16384, "input_cost_per_token": 0, "output_cost_per_token": 0}' \
    "${INC[@]}" -n $N_CONC -k ${ATTEMPTS:-1} --agent-timeout-multiplier ${AGENT_TIMEOUT_MULT:-1.0} \
    -o jobs --job-name $J -y -q > logs/harbor-$J.log 2>&1 || { log "harbor $J exit $?"; rc=1; }
  log "$J | $(python3 "$HERE/summarize.py" jobs/$J 2>&1 | tail -1)"
  mapfile -t NEW < <(comm -13 <(sort logs/images-before-$JOB.txt) <(tb2_images | sort))
  (( ${#NEW[@]} )) && docker rmi "${NEW[@]}" > /dev/null 2>&1
done
python3 "$HERE/summarize.py" jobs/$JOB-b* | tee logs/summary-$JOB.txt
exit $rc
