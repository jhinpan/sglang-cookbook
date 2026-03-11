#!/bin/bash
# Usage: ./test_kimi.sh <image_tag>
# Example: ./test_kimi.sh v0.5.9-rocm700-mi35x-20260303
set -e
TAG=$1
IMAGE="rocm/sgl-dev:${TAG}"
FIXED="sglang-test:${TAG}"
MODEL_PATH="/sgl-workspace/models/hub/models--moonshotai--Kimi-K2.5/snapshots/54383e83fa343a1331754112fb9e3410c55efa2f"

echo "=== Testing ${IMAGE} ==="

# Pull
echo "[1/4] Pulling image..."
docker pull ${IMAGE} 2>&1 | tail -2

# Build fixed image with aiter stub
echo "[2/4] Building fixed image..."
cd /home/jinpan12/sglang-cookbook
docker build -t ${FIXED} --build-arg BASE=${IMAGE} -f Dockerfile.bisect . 2>&1 | tail -3

# Launch
echo "[3/4] Launching Kimi-K2.5..."
docker rm -f kimi-bisect 2>/dev/null
docker run -d \
    --name kimi-bisect \
    --ipc=host --network=host --privileged \
    --shm-size 32G \
    --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    -e HF_HUB_OFFLINE=1 \
    -v /mnt/dcgpuval/huggingface:/sgl-workspace/models \
    ${FIXED} \
    python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --served-model-name kimi-k2.5 \
        --tp 8 \
        --attention-backend triton \
        --trust-remote-code \
        --mem-fraction-static 0.80 \
        --disable-cuda-graph \
        --watchdog-timeout 1200 \
        --port 30000

echo "[4/4] Waiting for server (checking every 60s, max 25 min)..."
for i in $(seq 1 25); do
    sleep 60
    HEALTH=$(curl -s --max-time 5 http://localhost:30000/health 2>/dev/null)
    LAST_LOG=$(docker logs --tail 1 kimi-bisect 2>&1)
    RUNNING=$(docker ps --format "{{.Names}}" | grep kimi-bisect)

    if [ -n "$HEALTH" ]; then
        echo "SUCCESS at minute ${i}: Server is healthy!"
        echo "Health: ${HEALTH}"
        # Quick inference test
        curl -s http://localhost:30000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"kimi-k2.5","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":32}' 2>/dev/null | python3 -m json.tool 2>/dev/null | head -15
        echo "=== RESULT: ${TAG} -> PASS ==="
        exit 0
    fi

    if [ -z "$RUNNING" ]; then
        echo "CRASHED at minute ${i}!"
        docker logs --tail 5 kimi-bisect 2>&1
        echo "=== RESULT: ${TAG} -> FAIL ==="
        exit 1
    fi

    echo "  Minute ${i}: still loading... (last: ${LAST_LOG:0:80})"
done
echo "=== RESULT: ${TAG} -> TIMEOUT ==="
exit 2
