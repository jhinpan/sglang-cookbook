#!/bin/bash
# run_bench.sh - Run benchmarks against a running SGLang server.
#
# Uses curl + a lightweight Python timing harness (no sglang imports needed).
# This avoids all module-import issues inside Docker containers.
#
# Usage: ./run_bench.sh <container_name> <port> <results_csv> [extra_label]

set -euo pipefail

CONTAINER="$1"
PORT="$2"
RESULTS_CSV="$3"
LABEL="${4:-}"

SERVER_URL="http://localhost:${PORT}"

INPUT_LENS=(1024 8192 16384)
OUTPUT_LENS=(512 1024)
CONCURRENCY_LEVELS=(1 2 4 8)
SERVING_NUM_PROMPTS=16
SERVING_ISL=1024
SERVING_OSL=512

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_health() {
    local max_wait=${1:-1500}
    local interval=30
    local elapsed=0
    log "Waiting for server health at ${SERVER_URL}/health (max ${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s --max-time 5 "${SERVER_URL}/health" >/dev/null 2>&1; then
            log "Server healthy after ${elapsed}s"
            return 0
        fi
        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
            log "ERROR: container ${CONTAINER} exited"
            docker logs --tail 20 "${CONTAINER}" 2>&1 || true
            return 1
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    log "ERROR: health check timed out after ${max_wait}s"
    return 1
}

BENCH_PY=$(cat <<'PYEOF'
import sys, json, time, argparse, random, string, concurrent.futures, statistics
import urllib.request, urllib.error

def random_text(n_tokens):
    words = []
    for _ in range(n_tokens):
        words.append("".join(random.choices(string.ascii_lowercase, k=random.randint(2, 8))))
    return " ".join(words)

def single_request(url, model, input_len, output_len, stream=True):
    prompt = random_text(input_len)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": stream,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t_start = time.perf_counter()
    ttft = None
    total_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            if stream:
                for line in resp:
                    line = line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text and ttft is None:
                        ttft = time.perf_counter() - t_start
                    if text:
                        total_tokens += 1
            else:
                body = json.loads(resp.read())
                ttft = time.perf_counter() - t_start
                usage = body.get("usage", {})
                total_tokens = usage.get("completion_tokens", 0) or usage.get("total_tokens", 0)
    except Exception as e:
        return {"error": str(e)}
    t_end = time.perf_counter()
    total_time = t_end - t_start
    decode_time = total_time - (ttft or total_time)
    decode_tps = (total_tokens - 1) / decode_time if decode_time > 0 and total_tokens > 1 else 0
    return {
        "ttft": round(ttft, 4) if ttft else None,
        "total_time": round(total_time, 4),
        "total_tokens": total_tokens,
        "decode_tps": round(decode_tps, 2),
    }

def bench_latency(url, model, input_len, output_len):
    r = single_request(url, model, input_len, output_len)
    if "error" in r:
        return {"ttft": "NA", "decode_tps": "NA", "total_time": "NA"}
    return r

def bench_throughput(url, model, input_len, output_len, num_prompts, concurrency):
    results = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(single_request, url, model, input_len, output_len, False)
            for _ in range(num_prompts)
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    wall_time = time.perf_counter() - t0
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"output_tps": "NA", "req_tps": "NA", "avg_lat": "NA", "p50_lat": "NA", "p99_lat": "NA", "mean_ttft": "NA"}
    total_out_tokens = sum(r["total_tokens"] for r in ok)
    lats = sorted(r["total_time"] for r in ok)
    ttfts = [r["ttft"] for r in ok if r.get("ttft")]
    return {
        "output_tps": round(total_out_tokens / wall_time, 2),
        "req_tps": round(len(ok) / wall_time, 4),
        "avg_lat": round(statistics.mean(lats), 4),
        "p50_lat": round(lats[len(lats)//2], 4),
        "p99_lat": round(lats[int(len(lats)*0.99)], 4),
        "mean_ttft": round(statistics.mean(ttfts), 4) if ttfts else "NA",
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["latency", "throughput"])
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="kimi-k2.5")
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--output-len", type=int, required=True)
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=1)
    args = p.parse_args()

    if args.mode == "latency":
        r = bench_latency(args.url, args.model, args.input_len, args.output_len)
        print(json.dumps(r))
    else:
        r = bench_throughput(args.url, args.model, args.input_len, args.output_len, args.num_prompts, args.concurrency)
        print(json.dumps(r))
PYEOF
)

run_bench_latency() {
    log "=== Latency sweep (bench_one_batch equivalent) ==="
    for il in "${INPUT_LENS[@]}"; do
        for ol in "${OUTPUT_LENS[@]}"; do
            log "  IL=${il} OL=${ol} ..."
            local output
            output=$(python3 -c "$BENCH_PY" --mode latency \
                --url "$SERVER_URL" --input-len "$il" --output-len "$ol" 2>&1) || true

            local ttft decode_tps total_time
            ttft=$(echo "$output" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('ttft','NA'))" 2>/dev/null || echo "NA")
            decode_tps=$(echo "$output" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('decode_tps','NA'))" 2>/dev/null || echo "NA")
            total_time=$(echo "$output" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('total_time','NA'))" 2>/dev/null || echo "NA")

            echo "${LABEL},latency,${il},${ol},1,${ttft},${decode_tps},${total_time}" >> "$RESULTS_CSV"
            log "  -> ttft=${ttft}s decode=${decode_tps} tok/s total=${total_time}s"
        done
    done
}

run_bench_throughput() {
    log "=== Throughput sweep (bench_serving equivalent) ==="
    for conc in "${CONCURRENCY_LEVELS[@]}"; do
        log "  concurrency=${conc} ISL=${SERVING_ISL} OSL=${SERVING_OSL} ..."
        local output
        output=$(python3 -c "$BENCH_PY" --mode throughput \
            --url "$SERVER_URL" --input-len "$SERVING_ISL" --output-len "$SERVING_OSL" \
            --num-prompts "$SERVING_NUM_PROMPTS" --concurrency "$conc" 2>&1) || true

        local out_tps req_tps avg_lat p50 p99 mean_ttft
        out_tps=$(echo "$output" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('output_tps','NA'))" 2>/dev/null || echo "NA")
        req_tps=$(echo "$output" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('req_tps','NA'))" 2>/dev/null || echo "NA")
        avg_lat=$(echo "$output" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('avg_lat','NA'))" 2>/dev/null || echo "NA")
        p50=$(echo "$output" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('p50_lat','NA'))" 2>/dev/null || echo "NA")
        p99=$(echo "$output" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('p99_lat','NA'))" 2>/dev/null || echo "NA")
        mean_ttft=$(echo "$output" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('mean_ttft','NA'))" 2>/dev/null || echo "NA")

        echo "${LABEL},throughput,${SERVING_ISL},${SERVING_OSL},${conc},${mean_ttft},${out_tps},NA,${out_tps},${req_tps},${avg_lat},${p50},${p99},${mean_ttft}" >> "$RESULTS_CSV"
        log "  -> out_tps=${out_tps} req_tps=${req_tps} p50=${p50}s"
    done
}

if [ ! -f "$RESULTS_CSV" ]; then
    echo "label,benchmark,input_len,output_len,concurrency,ttft_s,decode_tps,total_lat_s,output_tps,req_tps,avg_lat_s,p50_lat_s,p99_lat_s,mean_ttft_s" > "$RESULTS_CSV"
fi

if ! wait_for_health; then
    echo "${LABEL},FAILED,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA" >> "$RESULTS_CSV"
    exit 1
fi

run_bench_latency
run_bench_throughput

log "Benchmarks complete for ${LABEL}"
