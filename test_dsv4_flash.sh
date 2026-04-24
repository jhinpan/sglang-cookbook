#!/bin/bash
# Launch DeepSeek-V4-Flash-FP8 on MI355X via the PR #23608 official recipe.
# Recipe is the verbatim launch command from the PR description, adapted only
# for this node (port 31000 since 30000 is squatted, explicit HF cache mount).
set -uo pipefail

TAG="${TAG:-sglang-dsv4-mi355x:flash-r1}"
HF_HUB="${HF_HUB:-/data/jhinpan-cache/hub}"
RESULTS_DIR="${RESULTS_DIR:-/home/jinpan12@amd.com/dsv4-flash-results-mia1}"
PORT="${PORT:-31000}"
NAME="${NAME:-dsv4-flash-r1}"
MODEL_REPO="${MODEL_REPO:-sgl-project/DeepSeek-V4-Flash-FP8}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
DP="${DP:-4}"

mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/launch.log"

echo "=== dsv4-flash launch $(date -u +%FT%TZ) ===" | tee "${LOG}"
echo "image    : ${TAG}"                              | tee -a "${LOG}"
echo "model    : ${MODEL_REPO}"                       | tee -a "${LOG}"
echo "gpus     : CUDA_VISIBLE_DEVICES=${GPUS}"        | tee -a "${LOG}"
echo "tp/dp    : tp=${TP} dp=${DP}"                    | tee -a "${LOG}"
echo "port     : ${PORT}"                              | tee -a "${LOG}"
echo "hf cache : ${HF_HUB}"                            | tee -a "${LOG}"

SNAP_DIR=$(ls -1d "${HF_HUB}"/models--sgl-project--DeepSeek-V4-Flash-FP8/snapshots/*/ 2>/dev/null | head -1)
if [ -z "${SNAP_DIR}" ] || [ ! -f "${SNAP_DIR}/config.json" ]; then
  echo "ERROR: V4-Flash-FP8 snapshot not found under ${HF_HUB}" | tee -a "${LOG}"
  exit 1
fi
SNAP_DIR="${SNAP_DIR%/}"
SNAP_REV=$(basename "${SNAP_DIR}")
MODEL_IN_CTR="/hf-cache/models--sgl-project--DeepSeek-V4-Flash-FP8/snapshots/${SNAP_REV}"
echo "snapshot : ${SNAP_REV}" | tee -a "${LOG}"
echo "model in ctr: ${MODEL_IN_CTR}" | tee -a "${LOG}"

docker rm -f "${NAME}" 2>/dev/null || true

docker run -d --name "${NAME}" \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --shm-size=32g \
  -v "${HF_HUB}:/hf-cache" \
  -e CUDA_VISIBLE_DEVICES="${GPUS}" \
  -e SGLANG_OPT_USE_FUSED_COMPRESS=false \
  -e SGLANG_OPT_USE_OLD_COMPRESSOR=true \
  -e SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false \
  -e SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false \
  -e SGLANG_OPT_USE_FUSED_HASH_TOPK=false \
  -e SGLANG_HACK_FLASHMLA_BACKEND=torch \
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=false \
  -e SGLANG_OPT_USE_TILELANG_MHC_PRE=false \
  -e SGLANG_OPT_USE_TILELANG_MHC_POST=false \
  -e SGLANG_ENABLE_THINKING=1 \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_USE_ROCM700A=1 \
  -e SGLANG_TOPK_TRANSFORM_512_TORCH=1 \
  -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
  -e SGLANG_DSV4_FP4_EXPERTS=false \
  -e SGLANG_OPT_DPSK_V4_RADIX=0 \
  -e SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false \
  -e SGLANG_OPT_USE_FUSED_STORE_CACHE=false \
  -e SGLANG_FORCE_TRITON_MOE_FP8=1 \
  -e HF_HUB_OFFLINE=1 \
  "${TAG}" \
  python3 -m sglang.launch_server \
    --model-path "${MODEL_IN_CTR}" \
    --served-model-name dsv4-flash \
    --trust-remote-code \
    --tp "${TP}" --dp "${DP}" --enable-dp-attention \
    --disable-radix-cache --attention-backend compressed \
    --max-running-request 256 --page-size 256 --chunked-prefill-size 8192 \
    --kv-cache-dtype auto \
    --host 0.0.0.0 --port "${PORT}" \
    --disable-shared-experts-fusion --disable-cuda-graph \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 2>&1 | tee -a "${LOG}"

echo "Launched container ${NAME}" | tee -a "${LOG}"
docker logs -f "${NAME}" 2>&1 | tee "${RESULTS_DIR}/server.log"
