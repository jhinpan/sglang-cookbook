# bench/ — fill in the coverage gaps

These are the harness scripts behind the **Roadmap** section of the site. Each
produces numbers for one verified row. Bring the model's server up first (the
launch command is in its recipe card on the site), then run against it.

| Script | Produces | Run |
|--------|----------|-----|
| `bench_latency.sh` | prefill tok/s + decode TPOT (BS=1) | `MODEL=zai-org/GLM-5.2-FP8 PORT=30000 bash bench/bench_latency.sh` |
| `bench_throughput.sh` | TTFT / TPOT / tok-s-per-GPU vs concurrency | `PORT=30000 bash bench/bench_throughput.sh` |
| `eval_accuracy.sh` | GSM8K + AIME25 | `PORT=30000 THINK="--thinking-mode glm-45" bash bench/eval_accuracy.sh` |

`THINK="--thinking-mode glm-45"` is for GLM thinking models; leave it empty for others.

## Putting results back on the site

The site renders entirely from `../models.js` (`window.MODELS`). Find the model,
then add measured rows to its config — **verified data only**, every number traced
to a run.

A benchmark row:

```js
{ isl: 8192, osl: 1024, concurrency: 64,
  ttft_ms: 15849, tpot_ms: 72.7, decode_tok_s: 528.1, tok_s_per_gpu: 66.0,
  source: "bench_throughput.sh, 2026-06-22, 8x MI355X" }
```

An accuracy row:

```js
{ name: "GSM8K", value: "97.2%", ref: "98.2%", note: "n=1319, sgl-eval" }
```

To light up a **new cell** in the coverage matrix (e.g. a `balanced` strategy or a
`gfx950` target), push a second object into the model's `configs` array with that
`gfx` + `strategy` and `verified: true`. Once a config is verified, delete the
matching entry from the model's `gaps` list so the roadmap stays honest.

The roofline gauge is computed automatically from `active_params_billions`,
`bytes_per_param`, and the HW bandwidth in `window.HW` — no need to fill it in.
