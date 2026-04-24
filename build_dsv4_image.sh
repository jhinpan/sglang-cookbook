#!/bin/bash
# Build sglang-dsv4-mi355x:flash-r1 overlay image.
# Base: rocm/sgl-dev:deepseek-v4-mi35x (official AMD MI355X image for PR #23608).
# Delta: overlay PR #23608 head 26fbc93 over /sgl-workspace/sglang, drop
# @dataclass from DeepSeekV4Config, stub kernelkit/bench.py.
set -euo pipefail

PR_TREE="/home/jinpan12@amd.com/sglang-fork-dsv4"
COOKBOOK="/home/jinpan12@amd.com/sglang-cookbook"
CTX="/tmp/dsv4-build-r1"
TAG="sglang-dsv4-mi355x:flash-r1"

if [ ! -d "${PR_TREE}/python/sglang" ]; then
  echo "ERROR: PR #23608 tree missing at ${PR_TREE}." >&2
  exit 1
fi

CUR_HEAD=$(git -C "${PR_TREE}" rev-parse HEAD)
if [ "${CUR_HEAD}" != "26fbc935300a3bfba34f3dfa8925310929f82680" ]; then
  echo "WARN: ${PR_TREE} is at ${CUR_HEAD}, expected 26fbc935300a3bfba34f3dfa8925310929f82680" >&2
fi

echo "[1/3] Staging build context at ${CTX} ..."
rm -rf "${CTX}"
mkdir -p "${CTX}/sglang-fork-dsv4"
cp -r "${PR_TREE}/python" "${CTX}/sglang-fork-dsv4/python"
cp "${COOKBOOK}/Dockerfile.dsv4" "${CTX}/Dockerfile"

echo "[2/3] Build context: $(du -sh "${CTX}" | cut -f1)"

echo "[3/3] docker build -t ${TAG} ..."
docker build -t "${TAG}" "${CTX}"

echo "Done. Tag: ${TAG}"
docker images "${TAG}"
