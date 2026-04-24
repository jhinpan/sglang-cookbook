# Kimi-K2.6 on MI355X — quick playbook

Verified on 8× AMD Instinct MI355X (gfx950) with the prebuilt
`jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420` image (includes
aiter `3125d3b01` MLA TP=8 fix + 13-bucket `E=384,N=128,…MI355X,int4_w4a16`
MoE configs).

## 1. Launch

```bash
# defaults: TP=8 EP=1 on port 30000, weights under $HOME/hf-cache/hub
bash test_kimi_k26.sh
```

Override via env:

```bash
TAG=ep8     bash test_kimi_k26.sh   # EP=8 + mori all-to-all
TAG=tp2ep4  bash test_kimi_k26.sh   # 2 expert replicas x 4-way EP
TAG=tp4ep2  bash test_kimi_k26.sh   # 4 expert replicas x 2-way EP
PORT=30001  bash test_kimi_k26.sh   # custom port
```

## 2. What's running

```bash
curl -s http://127.0.0.1:30000/v1/models | python3 -m json.tool
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:30000/health_generate

curl -s -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}' \
  | python3 -m json.tool
```

## 3. Benchmark (TP=8, BS=1, single-request)

```bash
for IN in 1024 2048 4096 8192 16384 32768; do
  docker exec kimi-k26-tp8 python3 -m sglang.bench_one_batch_server \
    --model-path "$(docker exec kimi-k26-tp8 ls -d /hf-cache/models--moonshotai--Kimi-K2.6/snapshots/*/ | head -1)" \
    --base-url http://127.0.0.1:30000 \
    --batch-size 1 --input-len $IN --output-len 1024 \
    --dataset-name random --skip-warmup \
    --result-filename /tmp/bench_k26_tp8.jsonl
done
```

Reference numbers from our run:

| Input | Output | TTFT   | Decode        | Total   |
|------:|-------:|-------:|--------------:|--------:|
| 1024  | 1024   | 0.30 s | 45.23 tok/s   | 22.6 s  |
| 2048  | 1024   | 0.35 s | 44.69 tok/s   | 22.9 s  |
| 4096  | 1024   | 0.43 s | 43.62 tok/s   | 23.5 s  |
| 8192  | 1024   | 0.65 s | 41.88 tok/s   | 24.5 s  |
| 16384 | 1024   | 1.10 s | 38.93 tok/s   | 26.3 s  |
| 32768 | 1024   | 2.23 s | 34.00 tok/s   | 30.1 s  |

## 4. Key flags explained

- `--ep-size 1` — MoE TP-sharded 8-way, hits the pre-tuned `E=384,N=128` config.
- `--decode-attention-backend triton --prefill-attention-backend aiter` —
  MI355X sweet spot (triton MLA for decode, aiter for heavy prefill).
- `SGLANG_ROCM_FUSED_DECODE_MLA=0` — avoids the triton MLA tuple-unpack crash.
- `SGLANG_DEEPSEEK_LOAD_MAX_WORKERS=4` — keeps weight-load RAM pressure bounded.
- `--reasoning-parser kimi_k2 --tool-call-parser kimi_k2` — splits `<think>` blocks
  and DSML-style tool calls out of `choices[0].message.content`.

## 5. Teardown

```bash
docker rm -f kimi-k26-tp8
rocm-smi --showmeminfo vram | grep "Total Used"   # confirm ~280 MB per GPU
```
