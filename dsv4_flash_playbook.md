# DeepSeek-V4-Flash on MI355X — quick playbook

Server is up at `http://127.0.0.1:31000` on `mia1-p02-g45`, container `dsv4-flash-tp8`, TP=8 DP=8 with DP-attention across all 8 GPUs.

## 1. What's running

```bash
# list models
curl -s http://127.0.0.1:31000/v1/models | python3 -m json.tool

# full server info (all flags, dtypes, memory, backends)
curl -s http://127.0.0.1:31000/get_server_info | python3 -m json.tool | head -40

# health
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:31000/health_generate
```

`v1/models` returns `{"id":"dsv4-flash", "max_model_len":1048576}`. The served-model-name `dsv4-flash` is what you pass in `"model":` on chat calls.

## 2. Chat completions

```bash
curl -s -X POST http://127.0.0.1:31000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-flash",
    "messages": [{"role":"user","content":"What is 2+2?"}],
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 256
  }' | python3 -m json.tool
```

Thinking mode is ON (`SGLANG_ENABLE_THINKING=1` in the container env). The model will emit `<think>…reasoning…</think>answer`. Both end up in `choices[0].message.content` because the `deepseek-v4` reasoning parser is not splitting them yet (upstream polish issue noted in the PR comment draft).

Switch to non-thinking mode per-request if you want direct answers:

```bash
curl -s -X POST http://127.0.0.1:31000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-flash",
    "messages": [{"role":"user","content":"What is 2+2?"}],
    "chat_template_kwargs": {"thinking": false},
    "max_tokens": 32
  }' | python3 -m json.tool
```

## 3. Raw /generate (bypasses chat template)

If you want to hand-format the V4 prompt with the special tokens:

```bash
curl -s -X POST http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d "{
    \"text\": \"<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜></think>\",
    \"sampling_params\": {\"max_new_tokens\": 64, \"temperature\": 0.0,
                          \"stop\": [\"<｜end▁of▁sentence｜>\"]}
  }"
```

Prompt template note: `<｜User｜>…<｜Assistant｜><think>` for thinking-mode, `<｜User｜>…<｜Assistant｜></think>` for direct-answer mode.

## 4. Benchmark sweep (already done for you)

Command that ran inside the container (needs `tabulate` installed first):

```bash
docker exec dsv4-flash-tp8 pip install tabulate --quiet
docker exec -e HF_HUB_OFFLINE=0 dsv4-flash-tp8 python3 -m sglang.bench_one_batch_server \
  --model-path /hf-cache/models--sgl-project--DeepSeek-V4-Flash-FP8/snapshots/ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17 \
  --base-url http://127.0.0.1:31000 \
  --batch-size 1 \
  --input-len 2048 4096 8192 \
  --output-len 1024 2048 \
  --dataset-name random --skip-warmup \
  --result-filename /tmp/bench_tp8_sweep.jsonl
```

Raw JSONL at `~/dsv4-flash-results-mia1/bench_tp8_sweep.jsonl`.

### TP=8 DP=8, BS=1, single-request end-to-end

| ISL (in) | OSL (out) | latency (s) | prefill (tok/s) | decode (tok/s) | TTFT (s) | overall (tok/s) |
|---------:|----------:|------------:|----------------:|---------------:|---------:|----------------:|
|    2048  |    1024   |      262.4  |          374.4  |          3.99  |    5.47  |          11.71  |
|    2048  |    2048   |      521.2  |          382.1  |          3.97  |    5.36  |           7.86  |
|    4096  |    1024   |      264.4  |          429.9  |          4.02  |    9.53  |          19.37  |
|    4096  |    2048   |      523.5  |          401.2  |          3.99  |   10.21  |          11.74  |
|    8192  |    1024   |      276.9  |          409.0  |          3.99  |   20.03  |          33.28  |
|    8192  |    2048   |      533.3  |          420.3  |          3.99  |   19.49  |          19.20  |

Observations:
- Decode throughput is essentially flat (~4 tok/s) across all ISL/OSL at BS=1. That's expected — V4-Flash has `num_key_value_heads=1` (MQA), so attention is tiny regardless of context; the bottleneck is per-token MoE-FP8 matmul + torch-reference FlashMLA. This is the same ~4 tok/s we saw at TP=4 DP=4; TP=8 does not help a BS=1 workload because MoE per-expert work is already small.
- Prefill scales with ISL: 2k→4k raises ISL throughput 374→430 tok/s; further gains are eaten by attention quadratic cost. 8k prefill takes ~20 s TTFT.
- Overall tok/s grows with ISL because ISL tokens are counted toward total throughput too.
- Every row has the same `chunked_prefill_size=1024` because DP-attention in this image auto-lowers the PR-recipe 8192 to 1024 to dodge an MoE-kernel sizing issue (warning `server_args.py:2057`).

## 5. Where the performance could come from

Remember: this config runs V4 on AMD with **every** JIT fast-path disabled (11 `SGLANG_OPT_USE_*=false` envs), `SGLANG_HACK_FLASHMLA_BACKEND=torch` (pure-PyTorch FlashMLA reference), `--disable-cuda-graph`, and `--disable-radix-cache`. It is the "correctness first" config. Order-of-magnitude speed-ups require:

1. Re-enabling the JIT kernels once their ROCm ports land in PR #23608.
2. Larger batch sizes (MoE amortizes across tokens; BS=1 is the worst case).
3. Radix cache re-enabled once the compressed-attention radix path is stable on HIP.
4. CUDA-graph replay for decode (ROCm HIP-graph needs driver support).

## 6. Useful one-liners

```bash
# tail server logs
docker logs -f dsv4-flash-tp8

# resource usage
rocm-smi --showuse --showmeminfo vram

# stop server (frees all 8 GPUs)
docker rm -f dsv4-flash-tp8

# relaunch identical config
TP=8 DP=8 GPUS=0,1,2,3,4,5,6,7 PORT=31000 NAME=dsv4-flash-tp8 \
  bash ~/sglang-cookbook/test_dsv4_flash.sh
```

## 7. Container env snapshot

Image: `sglang-dsv4-mi355x:flash-r1`
Base : `rocm/sgl-dev:deepseek-v4-mi35x` (digest `sha256:a5f71877...`)
SGLang: `0.5.8.dev20260129+gf959851eb` + PR #23608 head `26fbc93` overlay + 2 patches
(drop @dataclass, stub kernelkit/bench.py)

Weights : `/data/jhinpan-cache/hub/models--sgl-project--DeepSeek-V4-Flash-FP8/
           snapshots/ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17`  (274 GiB, 46 shards)
