#!/usr/bin/env bash
# Single-request latency for a RUNNING SGLang server (offline bench_one_batch_server).
# Produces the prefill tok/s + per-token decode (TPOT) numbers used by the
# "Measured performance" table and the roofline gauge.
#
#   MODEL=<hf-repo-or-local-path> PORT=30000 bash bench/bench_latency.sh
#
# Optional: ISL="1024 8192 16384"  OSL="1024"
set -uo pipefail
MODEL="${MODEL:?set MODEL=<hf repo id or local snapshot path>}"
PORT="${PORT:-30000}"
ISL="${ISL:-1024 8192 16384}"
OSL="${OSL:-1024}"
OUT="${OUT:-/tmp/bench_latency_${PORT}.jsonl}"

python3 -m sglang.bench_one_batch_server \
  --model-path "$MODEL" \
  --base-url "http://127.0.0.1:${PORT}" \
  --batch-size 1 \
  --input-len $ISL \
  --output-len $OSL \
  --dataset-name random --skip-warmup \
  --result-filename "$OUT"

echo "results -> $OUT"
