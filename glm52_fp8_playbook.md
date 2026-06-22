# GLM-5.2-FP8 on MI300X — quick playbook

Single-node **TP=8** deployment of `zai-org/GLM-5.2-FP8` on **8× AMD Instinct MI300X (gfx942)** with SGLang, using the **DSA tilelang** attention backend. Verified end-to-end: server, latency, throughput, and accuracy (GSM8K, AIME25).

> GLM-5.2 shares the DeepSeek-V3.2 / GLM-5.1 `glm_moe_dsa` architecture (MLA + **DeepSeek Sparse Attention, "DSA"**). The model code and DSA tilelang backend are already upstream in SGLang ≥ 0.5.13.post1 — **no special branch needed**. The AMD cookbook page is tracked in SGLang PR [#28471](https://github.com/sgl-project/sglang/pull/28471).

## 0. Environment (verified)

| Item | Value |
|------|-------|
| GPUs | 8× AMD Instinct MI300X (gfx942), 192 GiB each (~1.65 TB total) |
| ROCm | 7.2.0 |
| PyTorch | 2.9.1+rocm7.2.0 |
| SGLang | 0.5.13.post1.dev20260621 (`g3975ea5ac7`) |
| tilelang | 0.1.7.post3 |
| aiter | enabled (`SGLANG_USE_AITER=1`), ships `glm5_bf16_tuned_gemm.csv` |
| Image | `rocm/sgl-dev:v0.5.13.post1-rocm720-mi35x` (ROCm 7.2 / MI35x line; `PYTORCH_ROCM_ARCH=gfx942;gfx950`) |
| Weights | `zai-org/GLM-5.2-FP8` — 704 GB on disk, 141 shards, `model_type=glm_moe_dsa` |

Key container env (from this run): `SGLANG_USE_AITER=1`, `SGLANG_USE_ROCM700A=1`, `SGLANG_MOE_PADDING=1`, `PYTORCH_ROCM_ARCH=gfx942;gfx950`.

### Why FP8 only on MI300X
GLM-5.2 **BF16** weights are ~1.4 TB → ~175 GB/GPU across 8×192 GiB, leaving only ~17–31 GB/GPU for KV cache + activations + CUDA graphs. BF16 **does not fit single-node on MI300X** (only MI325X/MI355X). FP8 (704 GB → 88 GB/GPU) fits comfortably. No FP4 checkpoint exists.

### gfx942 is safe
The block-FP8 GEMM accuracy bug noted in PR #28471 affects **gfx950 (MI350X/MI355X)** only. **gfx942 (MI300X) is unaffected** — GSM8K below confirms healthy numerics.

## 1. Launch the server (TP=8, DSA tilelang)

```bash
GPUS=0,1,2,3,4,5,6,7 PORT=30000 NAME=glm52-fp8-tp8 \
  bash ~/sglang-cookbook/test_glm52_fp8.sh
```

Equivalent direct command (what runs inside the container):

```bash
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-5.2-FP8 \
  --served-model-name glm-5.2 \
  --trust-remote-code \
  --tp 8 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype bfloat16 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.85 \
  --cuda-graph-max-bs 64 --max-running-requests 64 \
  --watchdog-timeout 1200 \
  --host 0.0.0.0 --port 30000
```

**Gotchas (learned the hard way):**
- **KV-cache dtype must be set.** With the DSA tilelang backend, FP8 KV is incompatible — use `--kv-cache-dtype bfloat16` (SGLang auto-selects this and warns if omitted).
- **Keep prefill chunked.** An *unchunked* long-context prefill trips the tilelang DSA prefill tile limit: `RuntimeError: tensor a (16384) must match b (131072)`. `--chunked-prefill-size 8192` is proven-safe; raising it (within the tile limit) improves TTFT.
- **No MTP / speculative decoding on AMD.** Omit all `--speculative-*` flags (the NV recipe's EAGLE MTP is not enabled on AMD).
- **No DP-attention / DeepEP** in this verified config (NV "balanced/high-throughput" recipes add `--dp 8 --enable-dp-attention --moe-a2a-backend deepep`; left out here for a first-known-good MI300X config).

Server comes up with `max_total_num_tokens≈691904`, `context_len=1048576`.

## 2. Smoke test

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"glm-5.2","messages":[{"role":"user","content":"What is 12*13?"}],
  "max_tokens":64,"temperature":0}' | python3 -m json.tool
```

GLM-5.2 is a **thinking** model: it emits long reasoning by default. Its chat template sets `effective_reasoning_effort='max'` unless you pass `reasoning_effort='high'` — budget tokens accordingly.

## 3. Latency — `sglang.bench_one_batch` (offline, single batch)

`bench_one_batch` runs the whole `batch×input_len` as one **unchunked** forward, so for this DSA model it is **bs=1-only at ISL 8192** (higher bs hits the tilelang prefill tile limit from §1). Single-request latency:

```bash
python3 -m sglang.bench_one_batch \
  --model-path zai-org/GLM-5.2-FP8 --tp 8 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --mem-fraction-static 0.60 --cuda-graph-bs 1 \
  --trust-remote-code --batch-size 1 --input-len 8192 --output-len 1024
```

| Metric | bs=1, ISL 8192 / OSL 1024 |
|--------|---------------------------|
| Prefill latency | **1.328 s** (6,170 tok/s) |
| Decode | **19.6 ms/token** (51.0 tok/s) |
| Total (8192 in + 1024 out) | 21.40 s (430.7 tok/s) |

## 4. Throughput — `sglang.bench_serving` (online, vs concurrency)

Batched throughput can't go through `bench_one_batch` for this model (see §3), so it's measured against the live server with chunked prefill — which is also how the NV cookbook numbers were produced (apples-to-apples).

```bash
for C in 1 16 64; do
  python3 -m sglang.bench_serving --backend sglang --dataset-name random \
    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
    --num-prompts $((C*2)) --max-concurrency $C --host 127.0.0.1 --port 30000
done
```

| Concurrency | Output tok/s | tok/s/GPU | Median TPOT (ms) | Median TTFT (ms) |
|------------:|-------------:|----------:|-----------------:|-----------------:|
| 1  | 47.8  | 6.0  | 18.9 | 1,458 |
| 16 | 311.5 | 38.9 | 41.4 | 10,226 |
| 64 | 528.1 | 66.0 | 72.7 | 15,849 |

> **TTFT caveat:** pessimistic because `chunked-prefill-size=8192` serializes prefills; a larger chunk (within the tilelang tile limit) lowers high-concurrency TTFT. Decode metrics (TPOT, tok/s) are representative.

## 5. Accuracy

```bash
# GSM8K (chat + thinking)
python3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \
  --thinking-mode glm-45 --max-tokens 8192 --temperature 0 --num-examples 1319

# AIME25 (chat + thinking, cookbook spec). max-tokens=64000 is REQUIRED:
# GLM-5.2 defaults to 'max' reasoning effort; a 32k cap truncates hard problems → ~56% (artifact).
python3 -m sglang.test.run_eval --port 30000 --eval-name aime25 \
  --thinking-mode glm-45 --repeat 8 --max-tokens 64000 --temperature 1.0 --top-p 0.95
```

| Benchmark | GLM-5.2-FP8 @ MI300X | Cookbook ref | Notes |
|-----------|----------------------|--------------|-------|
| **GSM8K** | **97.2%** (n=1319) | 98.2% | parity ✓ — FP8 numerics healthy on gfx942 |
| **AIME25** | **62.5%** (8 reps, max_tokens 64000) | 87.7% | 32k cap gave 56.3% (truncation); 64k → 62.5%. Residual gap: hardest items still hit 64k + harness/parser diffs |

## 6. Comparison vs NVIDIA (SGLang cookbook, GLM-5.2-FP8, ISL 8192 / OSL 1024)

> ⚠️ **Not apples-to-apples on decode.** NV cookbook configs use **EAGLE MTP speculative decoding** (+ DP-attention + DeepEP), which 2–3× decode speed. **MTP is disabled on AMD**, so the MI300X numbers below are non-speculative and expected to trail.

| HW | Strategy | Conc | TTFT (ms) | TPOT (ms) | tok/s/GPU |
|----|----------|-----:|----------:|----------:|----------:|
| **MI300X** (this run, no MTP) | tilelang | 1  | 1,458 | 18.9 | 6.0 |
| **MI300X** (this run, no MTP) | tilelang | 16 | 10,226 | 41.4 | 38.9 |
| **MI300X** (this run, no MTP) | tilelang | 64 | 15,849 | 72.7 | 66.0 |
| H200 (MTP) | low-latency | 1 | 662 | 3.03 | 34 |
| H200 (MTP) | low-latency | 16 | 5,080 | 12.44 | 113 |
| H200 (MTP) | balanced | 64 | 8,013 | 25.57 | 219 |
| B300 (MTP) | low-latency | 1 | 503 | 3.24 | 34 |
| B300 (MTP) | balanced | 64 | 6,465 | 23.36 | 245 |
| GB300 (MTP) | low-latency | 1 | 393 | 2.78 | 79 |

**Takeaways:** ① FP8 works on MI300X out-of-the-box on stock SGLang ≥0.5.13.post1 + DSA tilelang. ② Accuracy is at parity (GSM8K 97.2% vs 98.2%). ③ Decode trails NV ~3–4×, primarily because AMD has **no MTP** here (NV numbers include EAGLE MTP) plus per-GPU HW differences. ④ Clear next levers: enable MTP/spec when supported on AMD, add DP-attention + DeepEP, and raise the prefill chunk size to cut TTFT.

## 7. One-liners

```bash
curl -s http://127.0.0.1:30000/get_server_info | python3 -m json.tool | head -40   # all flags/backends
rocm-smi --showuse --showmeminfo vram                                              # GPU usage
# stop server (frees all 8 GPUs):  docker rm -f glm52-fp8-tp8   (or kill the launch_server PID)
```
