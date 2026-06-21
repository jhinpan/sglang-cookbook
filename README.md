# SGLang AMD Cookbook

Copy-paste ready commands for deploying and benchmarking large language models on AMD Instinct MI355X GPUs with [SGLang](https://github.com/sgl-project/sglang).

## Models Covered

| Model | HF Path | Architecture | Precision |
|-------|---------|-------------|-----------|
| Qwen3.5-397B-A17B | `Qwen/Qwen3.5-397B-A17B` | MoE (397B total, 17B active) | BF16 |
| Kimi-K2.5 | `moonshotai/Kimi-K2.5` | MoE (1T total, 32B active) | W4A16 |
| Kimi-K2.6 | `moonshotai/Kimi-K2.6` | MoE (1T total, 32B active) | W4A16 |
| DeepSeek-V4-Flash | `sgl-project/DeepSeek-V4-Flash-FP8` | MoE + MLA/MQA (~430B total, ~17B active) | FP8 |
| GLM-5-FP8 | `zai-org/GLM-5-FP8` | MoE + NSA (744B total, 40B active) | FP8 |
| GLM-5.2-FP8 | `zai-org/GLM-5.2-FP8` | MoE + MLA/DSA (`glm_moe_dsa`) | FP8 |

## Playbooks

- [GLM-5.2-FP8 on MI300X](glm52_fp8_playbook.md) — TP=8, DSA tilelang backend, FP8; latency/throughput + GSM8K/AIME25 vs NVIDIA H200/B300 ([`test_glm52_fp8.sh`](test_glm52_fp8.sh))
- [DeepSeek-V4-Flash on MI355X](dsv4_flash_playbook.md) ([`test_dsv4_flash.sh`](test_dsv4_flash.sh))
- [Kimi-K2.6 on MI355X](kimi_k26_playbook.md) ([`test_kimi_k26.sh`](test_kimi_k26.sh))

## View the Cookbook

Visit **[jhinpan.github.io/sglang-cookbook](https://jhinpan.github.io/sglang-cookbook/)** for the full interactive guide.

## License

MIT
