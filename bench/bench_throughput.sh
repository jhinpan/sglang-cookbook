#!/usr/bin/env bash
# Online throughput vs concurrency for a RUNNING SGLang server (bench_serving).
# Produces TTFT / TPOT / tok-s-per-GPU at each concurrency level.
#
#   PORT=30000 bash bench/bench_throughput.sh
#
# Optional: ISL=8192  OSL=1024  CONC="1 16 64"
set -uo pipefail
PORT="${PORT:-30000}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONC="${CONC:-1 16 64}"

for C in $CONC; do
  echo "==================== concurrency=${C} (ISL ${ISL} / OSL ${OSL}) ===================="
  python3 -m sglang.bench_serving --backend sglang --dataset-name random \
    --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1.0 \
    --num-prompts $((C * 2)) --max-concurrency "$C" \
    --host 127.0.0.1 --port "$PORT"
done
