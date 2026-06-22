#!/usr/bin/env bash
# Accuracy probes for a RUNNING SGLang server: GSM8K (in-tree) + AIME25 (sgl-eval).
#
#   PORT=30000 bash bench/eval_accuracy.sh
#
# For GLM thinking models pass THINK="--thinking-mode glm-45".
# IMPORTANT: AIME uses sgl-eval, NOT the in-tree run_eval — the in-tree
# answer-extraction regex badly undercounts reasoning models (e.g. 62.5% vs 90.6%).
set -uo pipefail
PORT="${PORT:-30000}"
THINK="${THINK:-}"
N="${N:-1319}"

echo "==================== GSM8K (n=${N}) ===================="
python3 -m sglang.test.run_eval --port "$PORT" --eval-name gsm8k \
  $THINK --max-tokens 8192 --temperature 0 --num-examples "$N"

echo "==================== AIME25 (sgl-eval) ===================="
python3 -c "import sgl_eval" 2>/dev/null || pip install -q git+https://github.com/sgl-project/sgl-eval
sgl-eval run aime25 --api-key EMPTY --base-url "http://localhost:${PORT}/v1" \
  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking
