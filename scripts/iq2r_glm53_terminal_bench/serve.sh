#!/usr/bin/env bash
# Start the GLM-5.3 ATOM server for one arm in its own container (container tb2-ARM-PORT, log LOG_DIR/tb2-ARM-PORT.log).
# usage: serve.sh ARM MODEL GPUS PORT
#   ARM: fp8   -> experts and shared expert excluded from online quant, served as stored: on an IQ2R-v2 checkpoint the
#                 experts run on the native IQ2R-v2 kernels, the shared expert is the stored block FP8
#        mxfp4 -> MXFP4 baseline (FP8 source, experts quantized to MXFP4 at load)
# Non-expert layers are online PTPC FP8 in both arms. FP8 KV cache, prefix cache, --max-model-len 262144.
# env: IMAGE, CHAT_TEMPLATE, MAX_NUM_SEQS (64), CG_MAX (64: largest CUDA-graph batch), LOG_DIR (.), EXTRA_DOCKER_ARGS, EXTRA_SERVE_ARGS, MODELS_MOUNTS
# Never sets ATOM_LOADER_STRICT_COVERAGE (default true: a checkpoint that misses an expert fails to load).
set -euo pipefail
ARM=$1; MODEL=$2; GPUS=$3; PORT=$4
IMAGE="${IMAGE:-glm53-ab:aiter-6dbf9e876-atom-17aca439a}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/models/zai-org/GLM-5.3-IQ2R/chat_template.jinja}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
LOG_DIR="${LOG_DIR:-.}"
case "$ARM" in
  fp8)
    QUANT='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*.mlp.experts","*.mlp.shared_experts*"]}' ;;
  mxfp4)
    QUANT='{"global_quant_config":"ptpc_fp8","layer_quant_config":{"*expert*":"mxfp4"},"exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate"]}' ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
NAME="tb2-${ARM}-${PORT}"
CG="$(python3 -c "print([c for c in [1,2,4,8,16,32,64,128,256] if c <= ${CG_MAX:-64}])")"
if env | grep -q '^ATOM_LOADER_STRICT_COVERAGE='; then echo "refusing: ATOM_LOADER_STRICT_COVERAGE is set" >&2; exit 2; fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --network host --ipc host --privileged \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit core=0 --shm-size 64g \
  -e HIP_VISIBLE_DEVICES="$GPUS" \
  -e PYTHONUNBUFFERED=1 -e AITER_LOG_LEVEL=WARNING \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
  -e MASTER_PORT=$(( 29500 + (PORT - 8000) * 200 )) \
  ${MODELS_MOUNTS:--v /models:/models:ro} \
  -v "tb2-${ARM}-cache:/root/.cache" \
  ${EXTRA_DOCKER_ARGS:-} \
  "$IMAGE" \
  python3 -m atom.entrypoints.openai_server \
    --model "$MODEL" --served-model-name zai-org/GLM-5.3 \
    --chat-template "$CHAT_TEMPLATE" \
    --host 127.0.0.1 --server-port "$PORT" --port $(( PORT - 880 )) \
    --data-parallel-master-port $(( 29500 + (PORT - 8000) * 200 )) \
    --tensor-parallel-size "$(echo "$GPUS" | tr , "\n" | wc -l)" \
    --max-model-len 262144 --kv_cache_dtype fp8 \
    --online_quant_config "$QUANT" \
    --enable_prefix_caching --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 16384 --cudagraph-capture-sizes "$CG" \
    --gpu-memory-utilization 0.9 \
    --timeout-keep-alive 900 --disable-uvicorn-access-log ${EXTRA_SERVE_ARGS:-} \
  >/dev/null
mkdir -p "$LOG_DIR"
nohup docker logs -f "$NAME" > "$LOG_DIR/$NAME.log" 2>&1 &
echo "started $NAME: $MODEL GPUs $GPUS :$PORT log $LOG_DIR/$NAME.log"
echo "quant: $QUANT | max-num-seqs $MAX_NUM_SEQS | extra serve args: ${EXTRA_SERVE_ARGS:-}"
