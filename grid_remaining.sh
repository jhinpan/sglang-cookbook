#!/bin/bash
# grid_remaining.sh - Run only the remaining untested configs (cp=0 only).
# Appends to the existing results CSV from the first run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/grid_results/20260321_082724"
RESULTS_CSV="${RESULTS_DIR}/results.csv"
LOG_FILE="${RESULTS_DIR}/grid_search.log"
CONTAINER_NAME="k25-grid"
PORT=30000
MODEL_PATH="/sgl-workspace/models/hub/models--moonshotai--Kimi-K2.5/snapshots/54383e83fa343a1331754112fb9e3410c55efa2f"
MODEL_VOLUME="/mnt/dcgpuval/huggingface:/sgl-workspace/models"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

teardown() {
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    sleep 5
}

launch_server() {
    local image="$1" mem_frac="$2"
    local tag="grid-${image##*:}-aiter0"
    tag="${tag//\//-}"

    if ! docker images --format '{{.Repository}}' | grep -q "^${tag}$"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building image ${tag}..." >> "$LOG_FILE"
        docker build -t "$tag" \
            --build-arg "BASE=${image}" \
            --build-arg "USE_AITER=0" \
            -f "${SCRIPT_DIR}/Dockerfile.grid" \
            "$SCRIPT_DIR" >> "$LOG_FILE" 2>&1
    fi

    docker run -d \
        --name "$CONTAINER_NAME" \
        --ipc=host --network=host --privileged \
        --shm-size 32G \
        --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --security-opt seccomp=unconfined \
        --security-opt apparmor=unconfined \
        -v "$MODEL_VOLUME" \
        -e HF_HUB_OFFLINE=1 \
        -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
        -e SGLANG_USE_AITER=0 \
        "$tag" \
        python3 -m sglang.launch_server \
            --model-path "$MODEL_PATH" \
            --served-model-name kimi-k2.5 \
            --tp 8 \
            --attention-backend triton \
            --trust-remote-code \
            --mem-fraction-static "$mem_frac" \
            --watchdog-timeout 1200 \
            --host 0.0.0.0 \
            --port "$PORT" \
            --tool-call-parser kimi_k2 \
            --reasoning-parser kimi_k2 \
            --schedule-policy lpm \
        >> "$LOG_FILE" 2>&1
}

CONFIGS=(
    "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260320|0.85|rocm720_20260320_triton_tp8_mem0.85_cp0_cgon_fmla0_lpm"
    "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260320|0.90|rocm720_20260320_triton_tp8_mem0.90_cp0_cgon_fmla0_lpm"
)

log "========================================"
log "Remaining configs sweep (cp=0 only)"
log "Total: ${#CONFIGS[@]}"
log "Appending to: ${RESULTS_CSV}"
log "========================================"

start_time=$(date +%s)
n=0
for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r image mem_frac label <<< "$entry"
    n=$((n+1))

    log "════════════════════════════════════════"
    log "Config [${n}/${#CONFIGS[@]}]: ${label}"
    log "════════════════════════════════════════"

    teardown

    if ! launch_server "$image" "$mem_frac"; then
        log "FAILED to launch: ${label}"
        echo "${label},LAUNCH_FAILED,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA" >> "$RESULTS_CSV"
        teardown
        continue
    fi

    if ! bash "${SCRIPT_DIR}/run_bench.sh" "$CONTAINER_NAME" "$PORT" "$RESULTS_CSV" "$label"; then
        log "Benchmark failed for: ${label}"
    fi

    docker logs --tail 50 "$CONTAINER_NAME" > "${RESULTS_DIR}/${label}_server.log" 2>&1 || true
    teardown

    elapsed=$(( $(date +%s) - start_time ))
    log "Total elapsed: ${elapsed}s"
done

end_time=$(date +%s)
log "========================================"
log "Remaining sweep complete in $(( end_time - start_time ))s"
log "========================================"

log ""
log "=== FINAL RESULTS SUMMARY ==="
log ""
log "Top decode throughput (IL=1024 OL=1024, tok/s):"
grep "latency,1024,1024" "$RESULTS_CSV" | sort -t',' -k7 -rn | head -10 | while IFS=',' read -r lbl bench il ol conc ttft dec tot rest; do
    log "  ${lbl}  decode=${dec} tok/s  ttft=${ttft}s"
done

log ""
log "Top output throughput (conc=8, tok/s):"
grep "throughput.*,8," "$RESULTS_CSV" | sort -t',' -k9 -rn | head -10 | while IFS=',' read -r lbl bench il ol conc ttft dec tot otps rest; do
    log "  ${lbl}  out_tps=${otps} tok/s"
done
