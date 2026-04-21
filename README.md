# SGLang AMD Cookbook

Copy-paste ready commands for deploying and benchmarking large language models on AMD Instinct MI355X GPUs with [SGLang](https://github.com/sgl-project/sglang).

## Models Covered

| Model | HF Path | Architecture | Precision |
|-------|---------|-------------|-----------|
| Qwen3.5-397B-A17B | `Qwen/Qwen3.5-397B-A17B` | MoE (397B total, 17B active) | BF16 |
| Kimi-K2.5 | `moonshotai/Kimi-K2.5` | MoE (1T total, 32B active) | W4A16 |
| Kimi-K2.6 | `moonshotai/Kimi-K2.6` | MoE (1T total, 32B active) | W4A16 |
| GLM-5-FP8 | `zai-org/GLM-5-FP8` | MoE + NSA (744B total, 40B active) | FP8 |

## View the Cookbook

Visit **[jhinpan.github.io/sglang-cookbook](https://jhinpan.github.io/sglang-cookbook/)** for the full interactive guide.

## License

MIT
