#!/bin/bash
# Launch Kimi-K2.6 on 8x MI355X with the prebuilt image.
# Default config: TP=8, EP=1 (hits the pre-tuned E=384,N=128,...MI355X,int4_w4a16 MoE configs),
# triton decode + aiter prefill. Override via env vars below.
#
# Quick use:
#   bash test_kimi_k26.sh            # default TP=8 EP=1 on port 30000
#   TAG=ep8 bash test_kimi_k26.sh    # EP=8 + mori a2a backend
#   TAG=tp2ep4 bash test_kimi_k26.sh # 2 expert replicas, 4-way EP each
#   TAG=tp4ep2 bash test_kimi_k26.sh # 4 expert replicas, 2-way EP each

set -uo pipefail

TAG="${TAG:-tp8}"
IMAGE="${IMAGE:-jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420}"
HF_HUB="${HF_HUB:-$HOME/hf-cache/hub}"
PORT="${PORT:-30000}"
NAME="${NAME:-kimi-k26-${TAG}}"

# Auto-detect the Kimi-K2.6 snapshot dir.
shopt -s nullglob
snapshots=("${HF_HUB}"/models--moonshotai--Kimi-K2.6/snapshots/*)
shopt -u nullglob
if [ "${#snapshots[@]}" -ne 1 ]; then
  echo "ERROR: expected exactly one Kimi-K2.6 snapshot under ${HF_HUB}, found ${#snapshots[@]}" >&2
  exit 1
fi
SNAP_HOST="${snapshots[0]}"
SNAP_IN_CTR="${SNAP_HOST/${HF_HUB}/\/hf-cache}"

# Per-config flags
EXTRA_ENV=()
case "${TAG}" in
  tp8)    EP_FLAGS=( --ep-size 1 ) ;;
  ep8)    EP_FLAGS=( --ep-size 8 --moe-a2a-backend mori --disable-cuda-graph --skip-server-warmup --watchdog-timeout 1800 --dist-timeout 3600 ) ;;
  tp2ep4) EP_FLAGS=( --ep-size 4 --moe-dp-size 2 --moe-a2a-backend mori --disable-cuda-graph --skip-server-warmup --watchdog-timeout 1800 --dist-timeout 3600 ) ;;
  tp4ep2) EP_FLAGS=( --ep-size 2 --moe-dp-size 4 --moe-a2a-backend mori --disable-cuda-graph --skip-server-warmup --watchdog-timeout 1800 --dist-timeout 3600 ) ;;
  *)      echo "Unknown TAG: ${TAG} (tp8|ep8|tp2ep4|tp4ep2)" >&2; exit 2 ;;
esac

if [ "${TAG}" != "tp8" ]; then
  EXTRA_ENV+=(
    -e SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384
    -e MORI_SHMEM_MODE=vmm
    -e MORI_SHMEM_HEAP_SIZE=34359738368
    -e TORCH_NCCL_BLOCKING_WAIT=0
    -e NCCL_ASYNC_ERROR_HANDLING=0
  )
fi

echo "image    : ${IMAGE}"
echo "tag/cfg  : ${TAG}"
echo "snapshot : ${SNAP_HOST}"
echo "port     : ${PORT}"
echo "name     : ${NAME}"

docker rm -f "${NAME}" 2>/dev/null || true

docker run -d --name "${NAME}" \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --shm-size=32g \
  -v "${HF_HUB}:/hf-cache" \
  -e HF_HUB_OFFLINE=1 \
  -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_DEEPSEEK_LOAD_MAX_WORKERS=4 \
  "${EXTRA_ENV[@]}" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
    --model-path "${SNAP_IN_CTR}" \
    --served-model-name kimi-k2.6 \
    --tensor-parallel-size 8 \
    "${EP_FLAGS[@]}" \
    --trust-remote-code \
    --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 \
    --decode-attention-backend triton --prefill-attention-backend aiter \
    --host 0.0.0.0 --port "${PORT}"

echo "Container ${NAME} launched. Tail logs:"
echo "  docker logs -f ${NAME}"
echo "Ready check:"
echo "  curl -s http://127.0.0.1:${PORT}/v1/models | python3 -m json.tool"
