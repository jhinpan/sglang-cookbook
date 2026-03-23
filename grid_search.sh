#!/bin/bash
# grid_search.sh - Brute-force grid search for optimal SGLang serving config
# for Kimi K2.5 on MI355X.
#
# Usage: ./grid_search.sh [--tier 1|2|3] [--dry-run]
#
# Iterates over Docker images, attention backends, TP, mem-fraction, and
# chunked-prefill-size.  For each combo it builds the image, launches the
# server, runs benchmarks via run_bench.sh, and tears down.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RESULTS_DIR="${SCRIPT_DIR}/grid_results/${TIMESTAMP}"
RESULTS_CSV="${RESULTS_DIR}/results.csv"
LOG_FILE="${RESULTS_DIR}/grid_search.log"
CONTAINER_NAME="k25-grid"
PORT=30000
MODEL_PATH="/sgl-workspace/models/hub/models--moonshotai--Kimi-K2.5/snapshots/54383e83fa343a1331754112fb9e3410c55efa2f"
MODEL_VOLUME="/mnt/dcgpuval/huggingface:/sgl-workspace/models"
HEALTH_TIMEOUT=1500
DRY_RUN=0
TIER=1

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)   TIER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Tier 1: High-impact parameters ──────────────────────────────────
IMAGES_T1=(
    "rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260319"
    "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260320"
)
ATTN_BACKENDS_T1=("triton")
# aiter is not functional in current images (PR1 not merged); add once available:
# ATTN_BACKENDS_T1=("triton" "aiter")
TP_VALUES_T1=(8)
MEM_FRACTIONS_T1=(0.80 0.85 0.90)
CHUNKED_PREFILL_T1=(0 65536 131072)
CUDA_GRAPH_T1=("on")
FUSED_MLA_T1=(0)
SCHEDULE_T1=("lpm")

# ── Tier 2: Medium-impact parameters (added on top of Tier 1) ──────
IMAGES_T2=("${IMAGES_T1[@]}")
ATTN_BACKENDS_T2=("triton")
TP_VALUES_T2=(4 8)
MEM_FRACTIONS_T2=(0.70 0.80 0.85 0.90 0.93)
CHUNKED_PREFILL_T2=(0 32768 65536 131072)
CUDA_GRAPH_T2=("on" "off")
FUSED_MLA_T2=(0 1)
SCHEDULE_T2=("lpm" "fcfs")

# ── Tier 3: Fine-tuning (everything) ───────────────────────────────
IMAGES_T3=(
    "rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260318"
    "rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260319"
    "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260318"
    "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260319"
    "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260320"
)
ATTN_BACKENDS_T3=("triton")
TP_VALUES_T3=(4 8)
MEM_FRACTIONS_T3=(0.70 0.75 0.80 0.85 0.90 0.93)
CHUNKED_PREFILL_T3=(0 16384 32768 65536 131072)
CUDA_GRAPH_T3=("on" "off")
FUSED_MLA_T3=(0 1)
SCHEDULE_T3=("lpm" "fcfs" "random")

# ── Select tier ─────────────────────────────────────────────────────
select_tier() {
    case "$TIER" in
        1) IMAGES=("${IMAGES_T1[@]}"); ATTN_BACKENDS=("${ATTN_BACKENDS_T1[@]}")
           TP_VALUES=("${TP_VALUES_T1[@]}"); MEM_FRACTIONS=("${MEM_FRACTIONS_T1[@]}")
           CHUNKED_PREFILL=("${CHUNKED_PREFILL_T1[@]}"); CUDA_GRAPH=("${CUDA_GRAPH_T1[@]}")
           FUSED_MLA=("${FUSED_MLA_T1[@]}"); SCHEDULE=("${SCHEDULE_T1[@]}") ;;
        2) IMAGES=("${IMAGES_T2[@]}"); ATTN_BACKENDS=("${ATTN_BACKENDS_T2[@]}")
           TP_VALUES=("${TP_VALUES_T2[@]}"); MEM_FRACTIONS=("${MEM_FRACTIONS_T2[@]}")
           CHUNKED_PREFILL=("${CHUNKED_PREFILL_T2[@]}"); CUDA_GRAPH=("${CUDA_GRAPH_T2[@]}")
           FUSED_MLA=("${FUSED_MLA_T2[@]}"); SCHEDULE=("${SCHEDULE_T2[@]}") ;;
        3) IMAGES=("${IMAGES_T3[@]}"); ATTN_BACKENDS=("${ATTN_BACKENDS_T3[@]}")
           TP_VALUES=("${TP_VALUES_T3[@]}"); MEM_FRACTIONS=("${MEM_FRACTIONS_T3[@]}")
           CHUNKED_PREFILL=("${CHUNKED_PREFILL_T3[@]}"); CUDA_GRAPH=("${CUDA_GRAPH_T3[@]}")
           FUSED_MLA=("${FUSED_MLA_T3[@]}"); SCHEDULE=("${SCHEDULE_T3[@]}") ;;
        *) echo "Invalid tier: $TIER"; exit 1 ;;
    esac
}

# ── Helpers ──────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

count_combos() {
    echo $(( ${#IMAGES[@]} * ${#ATTN_BACKENDS[@]} * ${#TP_VALUES[@]} \
             * ${#MEM_FRACTIONS[@]} * ${#CHUNKED_PREFILL[@]} * ${#CUDA_GRAPH[@]} \
             * ${#FUSED_MLA[@]} * ${#SCHEDULE[@]} ))
}

teardown() {
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    sleep 5
}

build_image() {
    local base_image="$1"
    local use_aiter="$2"
    local tag="grid-${base_image##*:}-aiter${use_aiter}"
    tag="${tag//\//-}"

    if docker images --format '{{.Repository}}' | grep -q "^${tag}$"; then
        echo "$tag"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building image ${tag} from ${base_image} (USE_AITER=${use_aiter})..." >> "$LOG_FILE"
    docker build -t "$tag" \
        --build-arg "BASE=${base_image}" \
        --build-arg "USE_AITER=${use_aiter}" \
        -f "${SCRIPT_DIR}/Dockerfile.grid" \
        "$SCRIPT_DIR" >> "$LOG_FILE" 2>&1

    echo "$tag"
}

pull_image() {
    local image="$1"
    if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${image}$"; then
        return 0
    fi
    log "Pulling ${image}..."
    docker pull "$image" >> "$LOG_FILE" 2>&1
}

launch_server() {
    local image="$1" tp="$2" mem_frac="$3" chunked="$4"
    local attn_backend="$5" cuda_graph="$6" fused_mla="$7" schedule="$8"

    local gpu_devices
    if [ "$tp" -eq 4 ]; then
        gpu_devices="0,1,2,3"
    else
        gpu_devices=""  # all GPUs
    fi

    local server_args=(
        python3 -m sglang.launch_server
        --model-path "$MODEL_PATH"
        --served-model-name kimi-k2.5
        --tp "$tp"
        --attention-backend "$attn_backend"
        --trust-remote-code
        --mem-fraction-static "$mem_frac"
        --watchdog-timeout 1200
        --host 0.0.0.0
        --port "$PORT"
        --tool-call-parser kimi_k2
        --reasoning-parser kimi_k2
        --schedule-policy "$schedule"
    )

    if [ "$chunked" -gt 0 ]; then
        server_args+=(--chunked-prefill-size "$chunked")
    fi

    if [ "$cuda_graph" = "off" ]; then
        server_args+=(--disable-cuda-graph)
    fi

    local env_args=(
        -e HF_HUB_OFFLINE=1
        -e "SGLANG_ROCM_FUSED_DECODE_MLA=${fused_mla}"
    )

    local use_aiter=0
    if [ "$attn_backend" = "aiter" ]; then
        use_aiter=1
        env_args+=(-e SGLANG_USE_AITER=1)
    else
        env_args+=(-e SGLANG_USE_AITER=0)
    fi

    local built_image
    built_image=$(build_image "$image" "$use_aiter")

    local docker_args=(
        docker run -d
        --name "$CONTAINER_NAME"
        --ipc=host --network=host --privileged
        --shm-size 32G
        --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE
        --device=/dev/kfd --device=/dev/dri
        --group-add video
        --security-opt seccomp=unconfined
        --security-opt apparmor=unconfined
        -v "$MODEL_VOLUME"
    )

    if [ -n "$gpu_devices" ]; then
        docker_args+=(-e "HIP_VISIBLE_DEVICES=${gpu_devices}")
    fi

    docker_args+=("${env_args[@]}")
    docker_args+=("$built_image")
    docker_args+=("${server_args[@]}")

    log "Launching: ${docker_args[*]}"
    "${docker_args[@]}" >> "$LOG_FILE" 2>&1
}

make_label() {
    local image="$1" attn="$2" tp="$3" mem="$4" chunked="$5"
    local cg="$6" fmla="$7" sched="$8"

    local rocm_tag
    rocm_tag=$(echo "$image" | grep -oP 'rocm\d+')
    local date_tag
    date_tag=$(echo "$image" | grep -oP '\d{8}$')

    echo "${rocm_tag}_${date_tag}_${attn}_tp${tp}_mem${mem}_cp${chunked}_cg${cg}_fmla${fmla}_${sched}"
}

# ── Main loop ────────────────────────────────────────────────────────
main() {
    select_tier
    mkdir -p "$RESULTS_DIR"

    local total
    total=$(count_combos)
    log "========================================"
    log "K2.5 Grid Search - Tier ${TIER}"
    log "Total configurations: ${total}"
    log "Results: ${RESULTS_CSV}"
    log "========================================"

    if [ "$DRY_RUN" -eq 1 ]; then
        log "DRY RUN - listing all configurations:"
        local n=0
        for img in "${IMAGES[@]}"; do
        for attn in "${ATTN_BACKENDS[@]}"; do
        for tp in "${TP_VALUES[@]}"; do
        for mem in "${MEM_FRACTIONS[@]}"; do
        for cp in "${CHUNKED_PREFILL[@]}"; do
        for cg in "${CUDA_GRAPH[@]}"; do
        for fmla in "${FUSED_MLA[@]}"; do
        for sched in "${SCHEDULE[@]}"; do
            n=$((n+1))
            local label
            label=$(make_label "$img" "$attn" "$tp" "$mem" "$cp" "$cg" "$fmla" "$sched")
            log "  [${n}/${total}] ${label}"
        done; done; done; done; done; done; done; done
        log "Dry run complete."
        return 0
    fi

    # Pre-pull all images
    log "Pre-pulling images..."
    for img in "${IMAGES[@]}"; do
        pull_image "$img"
    done

    local config_num=0
    local start_time
    start_time=$(date +%s)

    for img in "${IMAGES[@]}"; do
    for attn in "${ATTN_BACKENDS[@]}"; do
    for tp in "${TP_VALUES[@]}"; do
    for mem in "${MEM_FRACTIONS[@]}"; do
    for cp in "${CHUNKED_PREFILL[@]}"; do
    for cg in "${CUDA_GRAPH[@]}"; do
    for fmla in "${FUSED_MLA[@]}"; do
    for sched in "${SCHEDULE[@]}"; do
        config_num=$((config_num + 1))
        local label
        label=$(make_label "$img" "$attn" "$tp" "$mem" "$cp" "$cg" "$fmla" "$sched")

        log "════════════════════════════════════════"
        log "Config [${config_num}/${total}]: ${label}"
        log "════════════════════════════════════════"

        teardown

        local config_start
        config_start=$(date +%s)

        if ! launch_server "$img" "$tp" "$mem" "$cp" "$attn" "$cg" "$fmla" "$sched"; then
            log "FAILED to launch: ${label}"
            echo "${label},LAUNCH_FAILED,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA" >> "$RESULTS_CSV"
            teardown
            continue
        fi

        if ! bash "${SCRIPT_DIR}/run_bench.sh" "$CONTAINER_NAME" "$PORT" "$RESULTS_CSV" "$label"; then
            log "Benchmark failed for: ${label}"
        fi

        # Capture server logs for debugging
        docker logs --tail 50 "$CONTAINER_NAME" > "${RESULTS_DIR}/${label}_server.log" 2>&1 || true

        teardown

        local config_end elapsed_config elapsed_total remaining_est
        config_end=$(date +%s)
        elapsed_config=$((config_end - config_start))
        elapsed_total=$((config_end - start_time))
        if [ $config_num -gt 0 ]; then
            remaining_est=$(( (elapsed_total / config_num) * (total - config_num) ))
        else
            remaining_est=0
        fi
        log "Config took ${elapsed_config}s | Total: ${elapsed_total}s | ETA remaining: ~$((remaining_est/60))min"

    done; done; done; done; done; done; done; done

    local end_time
    end_time=$(date +%s)
    log "========================================"
    log "Grid search complete in $((end_time - start_time))s"
    log "Results: ${RESULTS_CSV}"
    log "========================================"

    generate_summary
}

generate_summary() {
    local summary="${RESULTS_DIR}/summary.md"
    log "Generating summary at ${summary}"

    cat > "$summary" <<'HEADER'
# Grid Search Results Summary

HEADER

    if [ ! -f "$RESULTS_CSV" ]; then
        echo "No results CSV found." >> "$summary"
        return
    fi

    echo '## Best decode throughput (bench_one_batch, tok/s)' >> "$summary"
    echo '```' >> "$summary"
    grep "bench_one_batch" "$RESULTS_CSV" 2>/dev/null \
        | sort -t',' -k7 -rn \
        | head -10 \
        | awk -F',' '{printf "%-60s  decode=%s tok/s  ttft=%ss  IL=%s OL=%s\n", $1, $7, $6, $3, $4}' \
        >> "$summary"
    echo '```' >> "$summary"

    echo '' >> "$summary"
    echo '## Best serving throughput (bench_serving, output tok/s)' >> "$summary"
    echo '```' >> "$summary"
    grep "bench_serving" "$RESULTS_CSV" 2>/dev/null \
        | sort -t',' -k9 -rn \
        | head -10 \
        | awk -F',' '{printf "%-60s  out_tps=%s  conc=%s  p50=%ss\n", $1, $9, $5, $12}' \
        >> "$summary"
    echo '```' >> "$summary"

    echo '' >> "$summary"
    echo "Generated: $(date)" >> "$summary"

    log "Summary written to ${summary}"
}

main "$@"
