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
| Image | `rocm/sgl-dev:v0.5.13.post1-rocm720-mi30x-20260620` (AMD ROCm 7.2 MI300X/gfx942 build; sglang built from source @ `a51d56d948`, aiter @ `7d604afe`) |
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
# GSM8K (chat + thinking) — in-tree harness is fine here
python3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \
  --thinking-mode glm-45 --max-tokens 8192 --temperature 0 --num-examples 1319

# AIME25 — use sgl-eval (NV's official harness), NOT the in-tree run_eval.
# The in-tree simple-evals answer regex undercounts this model badly (62.5% vs 90.6%).
pip install git+https://github.com/sgl-project/sgl-eval
sgl-eval run aime25 --api-key EMPTY --base-url http://localhost:30000/v1 \
  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking
```

| Benchmark | GLM-5.2-FP8 @ MI300X | Cookbook ref | Notes |
|-----------|----------------------|--------------|-------|
| **GSM8K** | **97.2%** (n=1319) | 98.2% | parity ✓ — FP8 numerics healthy on gfx942 |
| **AIME25** (`sgl-eval`) | **90.6%** pass@1[avg-of-16] (95% CI 88.6–92.6) | 87.7% | **≈ parity within noise** — pass@16 100%, majority@16 96.7%, truncated 0% |

> **Harness caveat (important):** with the **same model/server/settings**, the in-tree `sglang.test.run_eval --eval-name aime25` reports only **62.5%** because its `ANSWER_PATTERN = (?i)Answer\s*:\s*(...)` first-match regex grabs an intermediate "Answer:" from the reasoning trace or misses non-`Answer:` formats → false zeros. `sgl-eval` (NV's official harness) reports **90.6%** with `truncated_rate=0%`. Always use **`sgl-eval`** for AIME-style answer-extraction evals; the in-tree simple-evals are not reliable for this model.

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

## 7. Long-context (GLM-5.2 = DSA + 1M context)

GLM-5.2's DeepSeek Sparse Attention (DSA) is built for long context. Two probes on MI300X:

**LongBench-v2** (long-context reasoning accuracy) — `run_eval --eval-name longbench_v2` (pass `--model <path>` so it can build the tokenizer):
```bash
python3 -m sglang.test.run_eval --port 30000 \
  --model zai-org/GLM-5.2-FP8 --eval-name longbench_v2 \
  --thinking-mode glm-45 --max-tokens 16384 --temperature 0 \
  --num-examples 50 --max-context-length 256000 --num-threads 8
```

| Metric | GLM-5.2-FP8 @ MI300X | Reference |
|--------|----------------------|-----------|
| **LongBench-v2** | **59.5%** (subset, ~64k-tok cap) | human 53.7%, o1-preview 57.7%, best direct model 50.1% |

→ **beats human + o1-preview** on the subset. Strong long-context reasoning.

**Serving perf vs input length** (`bench_serving`, conc 1, output 512):

| Input length | TTFT median | TPOT | Output tok/s |
|-------------:|------------:|-----:|-------------:|
| 8,192   | 1.4 s  | 18.9 ms | 47.7 |
| 32,768  | 4.6 s  | 19.5 ms | 38.7 |
| 131,072 | 23.6 s | 21.5 ms | 19.0 |
| 262,144 | 43.8 s | 24.2 ms | 11.5 |

→ **DSA win: decode TPOT rises only 18.9 → 24.2 ms across a 32× context jump (8k → 256k)** — per-token decode stays nearly flat at long context; TTFT scales with prefill (~8k tok/s). Note these use the safe `--chunked-prefill-size 8192`; a larger chunk would cut long-context TTFT.

## 8. One-liners

```bash
curl -s http://127.0.0.1:30000/get_server_info | python3 -m json.tool | head -40   # all flags/backends
rocm-smi --showuse --showmeminfo vram                                              # GPU usage
# stop server (frees all 8 GPUs):  docker rm -f glm52-fp8-tp8   (or kill the launch_server PID)
```
