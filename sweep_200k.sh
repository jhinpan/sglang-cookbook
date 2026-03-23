#!/bin/bash
# sweep_200k.sh - Targeted sweep for 200K long-context support on Kimi K2.5.
#
# Tests whether the server can handle 200K-token inputs without OOM, and
# measures latency at high context lengths.
#
# Usage:
#   ./sweep_200k.sh                   # Run the 200K sweep
#   ./sweep_200k.sh --image <tag>     # Override Docker image
#   ./sweep_200k.sh --dry-run         # List configs only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RESULTS_DIR="${SCRIPT_DIR}/grid_results/200k_${TIMESTAMP}"
RESULTS_CSV="${RESULTS_DIR}/results_200k.csv"
LOG_FILE="${RESULTS_DIR}/sweep_200k.log"
CONTAINER_NAME="k25-200k"
PORT=30000
MODEL_PATH="/sgl-workspace/models/hub/models--moonshotai--Kimi-K2.5/snapshots/54383e83fa343a1331754112fb9e3410c55efa2f"
MODEL_VOLUME="/mnt/dcgpuval/huggingface:/sgl-workspace/models"
HEALTH_TIMEOUT=1800
DRY_RUN=0

IMAGE="rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260320"
ATTN_BACKEND="triton"
TP=8

MEM_FRACTIONS=(0.90 0.93)
CHUNKED_PREFILL_SIZES=(65536 131072)
CONTEXT_LENGTHS=(131072 200000)
LONG_INPUT_LENS=(32768 65536 131072 200000)
LONG_OUTPUT_LEN=256
OOM_DETECT_TIMEOUT=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)   IMAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *)         echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

teardown() {
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    sleep 5
}

detect_oom() {
    local container="$1"
    local logs
    logs=$(docker logs --tail 100 "$container" 2>&1 || echo "")

    if echo "$logs" | grep -qiE "out of memory|OOM|CUDA error.*out of memory|MemoryError|HIP.*out of memory"; then
        return 0  # OOM detected
    fi

    if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        local exit_code
        exit_code=$(docker inspect "$container" --format='{{.State.ExitCode}}' 2>/dev/null || echo "unknown")
        if [ "$exit_code" = "137" ] || [ "$exit_code" = "139" ]; then
            return 0  # killed by OOM-killer or segfault
        fi
    fi

    return 1  # no OOM
}

wait_for_health_or_oom() {
    local max_wait="$1"
    local interval=30
    local elapsed=0
    log "  Waiting for health (max ${max_wait}s, checking for OOM)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            log "  Server healthy after ${elapsed}s"
            return 0
        fi

        if detect_oom "$CONTAINER_NAME"; then
            log "  OOM DETECTED after ${elapsed}s"
            docker logs --tail 20 "$CONTAINER_NAME" >> "$LOG_FILE" 2>&1 || true
            return 2
        fi

        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            log "  Container exited after ${elapsed}s"
            docker logs --tail 20 "$CONTAINER_NAME" >> "$LOG_FILE" 2>&1 || true
            return 1
        fi

        sleep $interval
        elapsed=$((elapsed + interval))
    done
    log "  Timed out after ${max_wait}s"
    return 1
}

launch_server() {
    local mem_frac="$1" chunked="$2" ctx_len="$3"

    local use_aiter=0
    if [ "$ATTN_BACKEND" = "aiter" ]; then use_aiter=1; fi

    local built_image
    local tag="grid-${IMAGE##*:}-aiter${use_aiter}"
    tag="${tag//\//-}"

    if ! docker images --format '{{.Repository}}' | grep -q "^${tag}$"; then
        log "Building image ${tag}..."
        docker build -t "$tag" \
            --build-arg "BASE=${IMAGE}" \
            --build-arg "USE_AITER=${use_aiter}" \
            -f "${SCRIPT_DIR}/Dockerfile.grid" \
            "$SCRIPT_DIR" >> "$LOG_FILE" 2>&1
    fi

    local server_args=(
        python3 -m sglang.launch_server
        --model-path "$MODEL_PATH"
        --served-model-name kimi-k2.5
        --tp "$TP"
        --attention-backend "$ATTN_BACKEND"
        --trust-remote-code
        --mem-fraction-static "$mem_frac"
        --watchdog-timeout 1200
        --host 0.0.0.0
        --port "$PORT"
        --tool-call-parser kimi_k2
        --reasoning-parser kimi_k2
        --chunked-prefill-size "$chunked"
        --context-length "$ctx_len"
    )

    docker run -d \
        --name "$CONTAINER_NAME" \
        --ipc=host --network=host --privileged \
        --shm-size 32G \
        --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --security-opt seccomp=unconfined \
        --security-opt apparmor=unconfined \
        -e HF_HUB_OFFLINE=1 \
        -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
        -e SGLANG_USE_AITER=${use_aiter} \
        -v "$MODEL_VOLUME" \
        "$tag" \
        "${server_args[@]}" >> "$LOG_FILE" 2>&1
}

run_long_context_bench() {
    local il="$1" label="$2"

    log "  Benchmarking IL=${il} OL=${LONG_OUTPUT_LEN}..."

    local output
    output=$(timeout "$OOM_DETECT_TIMEOUT" docker exec "$CONTAINER_NAME" \
        python3 -m sglang.bench_one_batch_server \
            --model None --base-url "http://localhost:${PORT}" \
            --batch-size 1 --input-len "$il" --output-len "$LONG_OUTPUT_LEN" 2>&1) || {
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log "  TIMEOUT at IL=${il}"
            echo "${label},bench_one_batch,${il},${LONG_OUTPUT_LEN},TIMEOUT,NA,NA,NA" >> "$RESULTS_CSV"
            return 1
        fi

        if detect_oom "$CONTAINER_NAME"; then
            log "  OOM at IL=${il}"
            echo "${label},bench_one_batch,${il},${LONG_OUTPUT_LEN},OOM,NA,NA,NA" >> "$RESULTS_CSV"
            return 2
        fi

        log "  ERROR at IL=${il}"
        echo "${label},bench_one_batch,${il},${LONG_OUTPUT_LEN},ERROR,NA,NA,NA" >> "$RESULTS_CSV"
        return 1
    }

    local ttft decode_tps total_lat
    ttft=$(echo "$output" | grep -oP 'Prefill\.\s+latency:\s+\K[\d.]+' | head -1)
    decode_tps=$(echo "$output" | grep -oP 'Decode\.\s+median.*?throughput:\s+\K[\d.]+' | head -1)
    if [ -z "$decode_tps" ]; then
        decode_tps=$(echo "$output" | grep -oP 'Decode.*throughput:\s+\K[\d.]+' | tail -1)
    fi
    total_lat=$(echo "$output" | grep -oP 'Total\.\s+latency:\s+\K[\d.]+' | head -1)

    echo "${label},bench_one_batch,${il},${LONG_OUTPUT_LEN},OK,${ttft:-NA},${decode_tps:-NA},${total_lat:-NA}" >> "$RESULTS_CSV"
    log "  -> ttft=${ttft:-NA}s decode=${decode_tps:-NA} tok/s total=${total_lat:-NA}s"
}

main() {
    log "========================================"
    log "200K Long-Context Sweep"
    log "Image: ${IMAGE}"
    log "TP: ${TP}, Backend: ${ATTN_BACKEND}"
    log "========================================"

    echo "label,benchmark,input_len,output_len,status,ttft_s,decode_tps,total_lat_s" > "$RESULTS_CSV"

    local total=$(( ${#MEM_FRACTIONS[@]} * ${#CHUNKED_PREFILL_SIZES[@]} * ${#CONTEXT_LENGTHS[@]} ))
    local config_num=0

    if [ "$DRY_RUN" -eq 1 ]; then
        for mem in "${MEM_FRACTIONS[@]}"; do
        for cp in "${CHUNKED_PREFILL_SIZES[@]}"; do
        for ctx in "${CONTEXT_LENGTHS[@]}"; do
            config_num=$((config_num+1))
            log "[${config_num}/${total}] mem=${mem} cp=${cp} ctx=${ctx}"
            for il in "${LONG_INPUT_LENS[@]}"; do
                if [ "$il" -le "$ctx" ]; then
                    log "  -> bench IL=${il} OL=${LONG_OUTPUT_LEN}"
                fi
            done
        done; done; done
        return 0
    fi

    for mem in "${MEM_FRACTIONS[@]}"; do
    for cp in "${CHUNKED_PREFILL_SIZES[@]}"; do
    for ctx in "${CONTEXT_LENGTHS[@]}"; do
        config_num=$((config_num+1))
        local label="200k_mem${mem}_cp${cp}_ctx${ctx}"

        log "════════════════════════════════════════"
        log "Config [${config_num}/${total}]: ${label}"
        log "════════════════════════════════════════"

        teardown

        if ! launch_server "$mem" "$cp" "$ctx"; then
            log "FAILED to launch: ${label}"
            echo "${label},LAUNCH_FAILED,NA,NA,LAUNCH_FAILED,NA,NA,NA" >> "$RESULTS_CSV"
            teardown
            continue
        fi

        local health_result=0
        wait_for_health_or_oom "$HEALTH_TIMEOUT" || health_result=$?
        if [ $health_result -eq 2 ]; then
            log "Server OOM during startup for ${label}"
            echo "${label},STARTUP,NA,NA,OOM,NA,NA,NA" >> "$RESULTS_CSV"
            docker logs --tail 30 "$CONTAINER_NAME" > "${RESULTS_DIR}/${label}_oom.log" 2>&1 || true
            teardown
            continue
        elif [ $health_result -ne 0 ]; then
            log "Server failed to start for ${label}"
            echo "${label},STARTUP,NA,NA,FAILED,NA,NA,NA" >> "$RESULTS_CSV"
            docker logs --tail 30 "$CONTAINER_NAME" > "${RESULTS_DIR}/${label}_fail.log" 2>&1 || true
            teardown
            continue
        fi

        local oom_hit=0
        for il in "${LONG_INPUT_LENS[@]}"; do
            if [ "$il" -gt "$ctx" ]; then
                log "  Skipping IL=${il} > ctx=${ctx}"
                continue
            fi
            if [ $oom_hit -eq 1 ]; then
                log "  Skipping IL=${il} (previous OOM)"
                echo "${label},bench_one_batch,${il},${LONG_OUTPUT_LEN},SKIPPED_OOM,NA,NA,NA" >> "$RESULTS_CSV"
                continue
            fi

            local bench_result=0
            run_long_context_bench "$il" "$label" || bench_result=$?
            if [ $bench_result -eq 2 ]; then
                oom_hit=1
            fi
        done

        docker logs --tail 30 "$CONTAINER_NAME" > "${RESULTS_DIR}/${label}_server.log" 2>&1 || true
        teardown

    done; done; done

    log "========================================"
    log "200K sweep complete. Results: ${RESULTS_CSV}"
    log "========================================"

    # Summary
    log ""
    log "=== Max viable context per config ==="
    while IFS=, read -r label bench il ol status ttft decode total; do
        if [ "$status" = "OK" ]; then
            echo "  ${label}: IL=${il} OK (decode=${decode} tok/s)"
        fi
    done < <(tail -n +2 "$RESULTS_CSV") | sort -t= -k2 -rn | tee -a "$LOG_FILE"
}

main "$@"
