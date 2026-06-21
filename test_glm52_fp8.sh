#!/bin/bash
# Launch GLM-5.2-FP8 on 8x MI300X (gfx942) via SGLang with the DSA tilelang backend.
# Single-node TP=8, FP8 weights, bf16 KV cache, no MTP (speculative decoding is
# not enabled on AMD for this model). Verified on ROCm 7.2.0 / SGLang 0.5.13.post1.
set -uo pipefail

# --- image: ROCm 7.2 / MI35x line (supports gfx942 + gfx950) ---
TAG="${TAG:-rocm/sgl-dev:v0.5.13.post1-rocm720-mi35x}"
HF_HUB="${HF_HUB:-/data/hf-cache}"             # host HF cache mounted to /hf-cache
MODEL_REPO="${MODEL_REPO:-zai-org/GLM-5.2-FP8}"
PORT="${PORT:-30000}"
NAME="${NAME:-glm52-fp8-tp8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/glm52-fp8-results}"

mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/launch.log"
echo "=== glm5.2-fp8 launch $(date -u +%FT%TZ) ===" | tee "${LOG}"
echo "image : ${TAG}"  | tee -a "${LOG}"
echo "model : ${MODEL_REPO}" | tee -a "${LOG}"
echo "gpus  : ${GPUS}  tp=${TP}  port=${PORT}" | tee -a "${LOG}"

docker rm -f "${NAME}" 2>/dev/null || true

docker run -d --name "${NAME}" \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --shm-size=32g \
  -v "${HF_HUB}:/hf-cache" \
  -e HIP_VISIBLE_DEVICES="${GPUS}" \
  -e HF_HOME=/hf-cache \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_USE_ROCM700A=1 \
  -e SGLANG_MOE_PADDING=1 \
  "${TAG}" \
  python3 -m sglang.launch_server \
    --model-path "${MODEL_REPO}" \
    --served-model-name glm-5.2 \
    --trust-remote-code \
    --tp "${TP}" \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --kv-cache-dtype bfloat16 \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 64 --max-running-requests 64 \
    --watchdog-timeout 1200 \
    --host 0.0.0.0 --port "${PORT}" 2>&1 | tee -a "${LOG}"

echo "Launched container ${NAME}" | tee -a "${LOG}"
docker logs -f "${NAME}" 2>&1 | tee "${RESULTS_DIR}/server.log"
