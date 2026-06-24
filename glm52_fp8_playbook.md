# GLM-5.2-FP8 on AMD (MI300X / MI355X) — quick playbook

Single-node **TP=8** deployment of `zai-org/GLM-5.2-FP8` on **8× AMD Instinct** GPUs with SGLang, using the **DSA tilelang** attention backend. Verified end-to-end on **both** MI300X (gfx942) and MI355X (gfx950): server, latency, throughput, and accuracy (GSM8K, AIME25).

> **Which GPU am I on?** `rocminfo | grep -m1 gfx` (or `rocm-smi --showproductname` for the marketing name) → `gfx942` = MI300X, `gfx950` = MI355X.
>
> | | MI300X (gfx942) | MI355X (gfx950) |
> |---|---|---|
> | VRAM/GPU | 192 GiB | 288 GiB |
> | Launch flags | identical (§1) | **identical (§1)** |
> | Source patches needed | **none** | **2 patches first** (§1b) — else GSM8K ≈ 0.0 |
> | GSM8K (n=1319) | 97.2% | 97.7% |
> | AIME25 (sgl-eval, avg@16) | 90.6% | **91.5%** |
> | Single-stream decode | 48 tok/s | **67 tok/s** |
> | c64 throughput | 528 tok/s | **1009 tok/s** |
>
> **The launch command is the same on both.** The ONLY MI355X-specific step is applying the two mandatory gfx950 bpreshuffle patches in §1b before the first launch. Everything else in this playbook applies to both unless a section says otherwise.

> GLM-5.2 shares the DeepSeek-V3.2 / GLM-5.1 `glm_moe_dsa` architecture (MLA + **DeepSeek Sparse Attention, "DSA"**). The model code and DSA tilelang backend are already upstream in SGLang ≥ 0.5.13.post1 — **no special branch needed**. The AMD cookbook page is tracked in SGLang PR [#28471](https://github.com/sgl-project/sglang/pull/28471).

## 0. Environment (verified)

| Item | Value |
|------|-------|
| GPUs | **MI300X:** 8× gfx942, 192 GiB each (~1.65 TB) · **MI355X:** 8× gfx950, 288 GiB each (~2.3 TB) |
| ROCm | 7.2.0 |
| PyTorch | 2.9.1+rocm7.2.0 |
| SGLang | 0.5.13.post1 (MI300X `g3975ea5ac7`; MI355X `g4923bb93ae` editable + 2 mandatory gfx950 bpreshuffle patches, see §1b) |
| tilelang | 0.1.7.post3 |
| aiter | enabled (`SGLANG_USE_AITER=1`), ships `glm5_bf16_tuned_gemm.csv`; on gfx950 the bpreshuffle block-FP8 path is force-disabled in source (§1b) |
| Image | `rocm/sgl-dev:v0.5.13.post1-rocm720-mi30x-20260620` (AMD ROCm 7.2 build; covers gfx942 **and** gfx950; sglang @ `a51d56d948`, aiter @ `7d604afe`) |
| Weights | `zai-org/GLM-5.2-FP8` — 704 GB on disk, 141 shards, `model_type=glm_moe_dsa` |

Key container env (from this run): `SGLANG_USE_AITER=1`, `SGLANG_USE_ROCM700A=1`, `SGLANG_MOE_PADDING=1`, `PYTORCH_ROCM_ARCH=gfx942;gfx950`.

### Why FP8 (both GPUs)
GLM-5.2 **BF16** weights are ~1.4 TB → ~175 GB/GPU across 8 GPUs, leaving only ~17–31 GB/GPU on MI300X (192 GiB) for KV cache + activations + CUDA graphs. BF16 **does not fit single-node on MI300X** (only MI325X/MI355X). FP8 (704 GB → 88 GB/GPU) fits comfortably on both MI300X and MI355X. No FP4 checkpoint exists. We use FP8 on both for an apples-to-apples comparison.

### gfx942 is safe; gfx950 needs two patches
The block-FP8 GEMM accuracy bug from PR #28471 affects **gfx950 (MI350X/MI355X)** only — **gfx942 (MI300X) is unaffected** (GSM8K confirms healthy numerics with no patches). On **gfx950 you MUST apply the two bpreshuffle patches in §1b first**, or GSM8K collapses to ~0.0. Once patched, gfx950 reaches the same accuracy as gfx942.

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

Server comes up with `max_total_num_tokens≈691904`, `context_len=1048576` on MI300X (≈1417344 on MI355X — bigger KV pool from 288 GiB GPUs).

## 1b. MI355X (gfx950) — apply TWO mandatory source patches FIRST

> **MI300X users: skip this section.** gfx942 never touches the broken kernel paths; the §1 command works as-is. **MI355X users: the two bpreshuffle patches below are mandatory** — without them GSM8K collapses to ~0.0 with token-garbage. The launch command itself is **identical to §1** (same flags); only these source patches differ. (`SGLANG_USE_AITER=1` is set on both — via the container env / `test_glm52_fp8.sh` on MI300X, and explicitly here.)

The SGLang install is editable, so patches take effect at the **next server start — no rebuild**. Run this once:

```bash
SRT=$(python3 -c 'import os,sglang.srt as m; print(os.path.dirname(m.__file__))')

# Patch #1 — block-FP8 bpreshuffle GEMM is miscompiled on gfx950/ROCm 7.2 (sglang #28685).
# ROCm 7.2 hipcc drops `-mllvm -amdgpu-coerce-illegal-types` on aiter
# gemm_a8w8_blockscale_bpreshuffle → silently wrong output. Force the gate OFF.
sed -i 's/^_use_aiter_bpreshuffle_gfx95 = .*/_use_aiter_bpreshuffle_gfx95 = False  # FORCED OFF (gfx950 #28685)/' \
  "$SRT/layers/quantization/fp8_utils.py"

# Patch #1b — SAME flag, SECOND definition. GLM-5.2 is GlmMoeDsaForCausalLM →
# DeepseekV2ForCausalLM (models/glm4_moe.py), and its MLA/activation-quant path
# reads the flag from deepseek_common/utils.py, NOT from fp8_utils.py. Patching
# only #1 leaves GSM8K=0.0. THIS is the one that's easy to miss.
sed -i 's/^_use_aiter_bpreshuffle_gfx95 = .*/_use_aiter_bpreshuffle_gfx95 = False  # FORCED OFF (gfx950 #28685)/' \
  "$SRT/models/deepseek_common/utils.py"

# Verify BOTH now say "= False" (the fix is the VALUE, not just the variable name):
grep -n '^_use_aiter_bpreshuffle_gfx95' \
  "$SRT/layers/quantization/fp8_utils.py" "$SRT/models/deepseek_common/utils.py"
```

Then launch with the **exact same §1 command** plus `export SGLANG_USE_AITER=1`.

> **Optional — only if you ALSO serve DeepSeek-V4 from this tree.**
> `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` must be `False` for long-context correctness. **For GLM-5.2 this is already the default and needs no patch:** the `.set(True)` line in `server_args.py` lives in the `model_arch in ["DeepseekV4ForCausalLM"]` branch (~line 3786), but GLM-5.2's `model_arch` is `GlmMoeDsaForCausalLM`, so that branch never runs and the global `EnvBool` default (`False`) holds. We verified a long-context needle-recall **PASS** on the stock default with no patch. Apply the env-aware patch below **only** if you serve DeepSeek-V4 from the same source tree (there `.set(True)` would otherwise clobber a shell `export …=0`):
>
> ```bash
> python3 - "$SRT/server_args.py" <<'PY'
> import sys; p=sys.argv[1]; L=open(p).read().split("\n"); hip=False
> for i,l in enumerate(L):
>     s=l.strip()
>     if s=='elif is_hip():': hip=True; continue
>     if hip and (s.startswith('elif ') or s.startswith('else:')): hip=False
>     if hip and 'SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)' in l:
>         ind=l[:len(l)-len(l.lstrip())]
>         L[i]=ind+'if not envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.is_set():\n'+ind+'    envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(False)'
>         break
> open(p,'w').write("\n".join(L))
> PY
> ```

**Why these are gfx950-only:**
- The bpreshuffle kernel miscompile is **M-tile-sensitive** (wrong rows cluster at 16-row tile boundaries and shift run-to-run), so a `bs=1` single-token answer can look *correct* while batched/longer prompts corrupt — it is **one** bug, not a separate "batching bug." Don't trust a passing smoke test alone; run GSM8K.
- The correct fallback is `ck_gemm_a8w8_blockscale` / `triton_gemm_a8w8_blockscale` (verified cosine 1.0 on gfx950).
- **gfx942 (MI300X) never reaches the bpreshuffle path**, so none of this applies there.
- Upstream permanent fix is the CK kernel rewrite [ROCm/rocm-libraries#8639](https://github.com/ROCm/rocm-libraries/pull/8639) (scalar-FMA/VGPR accumulator); not yet in the aiter build here (`7d604afe5`), so the source workaround is required for now.

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

**MI300X (gfx942):**

| Metric | bs=1, ISL 8192 / OSL 1024 |
|--------|---------------------------|
| Prefill latency | **1.328 s** (6,170 tok/s) |
| Decode | **19.6 ms/token** (51.0 tok/s) |
| Total (8192 in + 1024 out) | 21.40 s (430.7 tok/s) |

> MI355X `bench_one_batch` not captured separately; see the §4 `bench_serving` conc=1 numbers (single-stream decode 14.4 ms/token / 66.9 tok/s) for the MI355X single-request figure.

## 4. Throughput — `sglang.bench_serving` (online, vs concurrency)

Batched throughput can't go through `bench_one_batch` for this model (see §3), so it's measured against the live server with chunked prefill — which is also how the NV cookbook numbers were produced (apples-to-apples).

```bash
for C in 1 16 64; do
  python3 -m sglang.bench_serving --backend sglang --dataset-name random \
    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
    --num-prompts $((C*2)) --max-concurrency $C --host 127.0.0.1 --port 30000
done
```

**MI300X (gfx942):**

| Concurrency | Output tok/s | tok/s/GPU | Median TPOT (ms) | Median TTFT (ms) |
|------------:|-------------:|----------:|-----------------:|-----------------:|
| 1  | 47.8  | 6.0  | 18.9 | 1,458 |
| 16 | 311.5 | 38.9 | 41.4 | 10,226 |
| 64 | 528.1 | 66.0 | 72.7 | 15,849 |

**MI355X (gfx950)** — same flags + the §1b patches; output verified correct (GSM8K 97.7%):

| Concurrency | Output tok/s | tok/s/GPU | Median TPOT (ms) | Median TTFT (ms) |
|------------:|-------------:|----------:|-----------------:|-----------------:|
| 1  | 66.9   | 8.4   | 14.4 | 652    |
| 16 | 535.7  | 67.0  | 25.2 | 4,790  |
| 64 | 1,009  | 126.1 | 41.5 | 13,795 |

> **MI355X vs MI300X:** ~1.4× single-stream decode (67 vs 48 tok/s), ~1.9× c64 throughput (1009 vs 528 tok/s), and lower TPOT/TTFT across the board — the larger, faster CDNA4 GPU plus a bigger KV pool (288 vs 192 GiB).
>
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

| Benchmark | MI300X (gfx942) | MI355X (gfx950) | Cookbook ref | Notes |
|-----------|-----------------|-----------------|--------------|-------|
| **GSM8K** (n=1319) | **97.2%** | **97.7%** | 98.2% | parity ✓ on both — FP8 numerics healthy (MI355X requires the §1b patches) |
| **AIME25** (`sgl-eval`, avg@16) | **90.6%** (95% CI 88.6–92.6) | **91.5%** (95% CI 89.1–93.8) | 87.7% | both **≈ parity within noise**, both beat ref. MI300X: pass@16 100%, majority@16 96.7%, trunc 0%. MI355X: pass@16 100%, majority@16 93.3%, trunc 0.21%, error 0% (n=30×16, 9.46M completion tok) |

> **Same model, same recipe, same numerics:** MI355X (gfx950) matches MI300X (gfx942) on accuracy **once the two mandatory §1b bpreshuffle patches are applied**. Without them GSM8K on gfx950 is ~0.0 — the gap is a kernel miscompile, not a real model/accuracy difference.
>
> **Harness caveat (important):** with the **same model/server/settings**, the in-tree `sglang.test.run_eval --eval-name aime25` reports only **62.5%** because its `ANSWER_PATTERN = (?i)Answer\s*:\s*(...)` first-match regex grabs an intermediate "Answer:" from the reasoning trace or misses non-`Answer:` formats → false zeros. `sgl-eval` (NV's official harness) reports the real ~90% with a near-zero `truncated_rate` (MI300X 0%, MI355X 0.21%). Always use **`sgl-eval`** for AIME-style answer-extraction evals; the in-tree simple-evals are not reliable for this model.

## 6. Comparison vs NVIDIA (SGLang cookbook, GLM-5.2-FP8, ISL 8192 / OSL 1024)

> ⚠️ **Not apples-to-apples on decode.** NV cookbook configs use **EAGLE MTP speculative decoding** (+ DP-attention + DeepEP), which 2–3× decode speed. **MTP is disabled on AMD**, so the MI300X/MI355X numbers below are non-speculative and expected to trail.

| HW | Strategy | Conc | TTFT (ms) | TPOT (ms) | tok/s/GPU |
|----|----------|-----:|----------:|----------:|----------:|
| **MI300X** (this run, no MTP) | tilelang | 1  | 1,458 | 18.9 | 6.0 |
| **MI300X** (this run, no MTP) | tilelang | 16 | 10,226 | 41.4 | 38.9 |
| **MI300X** (this run, no MTP) | tilelang | 64 | 15,849 | 72.7 | 66.0 |
| **MI355X** (this run, no MTP) | tilelang | 1  | 652 | 14.4 | 8.4 |
| **MI355X** (this run, no MTP) | tilelang | 16 | 4,790 | 25.2 | 67.0 |
| **MI355X** (this run, no MTP) | tilelang | 64 | 13,795 | 41.5 | 126.1 |
| H200 (MTP) | low-latency | 1 | 662 | 3.03 | 34 |
| H200 (MTP) | low-latency | 16 | 5,080 | 12.44 | 113 |
| H200 (MTP) | balanced | 64 | 8,013 | 25.57 | 219 |
| B300 (MTP) | low-latency | 1 | 503 | 3.24 | 34 |
| B300 (MTP) | balanced | 64 | 6,465 | 23.36 | 245 |
| GB300 (MTP) | low-latency | 1 | 393 | 2.78 | 79 |

**Takeaways:** ① FP8 works on **MI300X** out-of-the-box on stock SGLang ≥0.5.13.post1 + DSA tilelang; **MI355X** needs the two mandatory §1b bpreshuffle patches but then runs the identical recipe. ② Accuracy is at parity on both (GSM8K 97.2% / 97.7% vs 98.2% ref; AIME25 ≈90% via sgl-eval). ③ **MI355X is ~1.4–1.9× faster than MI300X** (single-stream 67 vs 48 tok/s; c64 1009 vs 528 tok/s). ④ Both still trail NV ~3–4× on decode, primarily because AMD has **no MTP** here (NV numbers include EAGLE MTP) plus per-GPU HW differences. ⑤ Clear next levers: enable MTP/spec when supported on AMD, add DP-attention + DeepEP, and raise the prefill chunk size to cut TTFT.

## 7. Long-context (GLM-5.2 = DSA + 1M context)

GLM-5.2's DeepSeek Sparse Attention (DSA) is built for long context. Two probes (LongBench-v2 accuracy verified on MI300X; serving-perf-vs-length captured on both GPUs below):

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

MI300X (gfx942):

| Input length | TTFT median | TPOT | Output tok/s |
|-------------:|------------:|-----:|-------------:|
| 8,192   | 1.4 s  | 18.9 ms | 47.7 |
| 32,768  | 4.6 s  | 19.5 ms | 38.7 |
| 131,072 | 23.6 s | 21.5 ms | 19.0 |
| 262,144 | 43.8 s | 24.2 ms | 11.5 |

MI355X (gfx950):

| Input length | TTFT median | TPOT | Output tok/s |
|-------------:|------------:|-----:|-------------:|
| 8,192   | 0.41 s | 14.4 ms | 65.7 |
| 32,768  | 1.18 s | 14.9 ms | 58.2 |
| 131,072 | 5.46 s | 16.6 ms | 36.7 |
| 262,144 | 9.58 s | 18.9 ms | 26.6 |

→ **DSA win:** per-token decode stays nearly flat at long context — MI300X TPOT rises only 18.9 → 24.2 ms across a 32× context jump (8k → 256k); MI355X is both flatter and faster (14.4 → 18.9 ms). TTFT scales with prefill. Note these use the safe `--chunked-prefill-size 8192`; a larger chunk would cut long-context TTFT.

## 8. One-liners

```bash
curl -s http://127.0.0.1:30000/get_server_info | python3 -m json.tool | head -40   # all flags/backends
rocm-smi --showuse --showmeminfo vram                                              # GPU usage
# stop server (frees all 8 GPUs):  docker rm -f glm52-fp8-tp8   (or kill the launch_server PID)
```
