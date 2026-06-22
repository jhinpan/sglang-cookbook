/* =====================================================================
   models.js — GENERATED, verified-only data.
   Source: adversarial extraction + verify workflow over the cookbook
   repo (playbooks, test scripts, index.html, grid_results). Every
   benchmark number is traced to a source file; un-measured cells are
   marked not-benchmarked. HW specs from AMD Instinct datasheets.
   Regenerate via the extraction workflow — do not hand-edit numbers.
   ===================================================================== */
window.HW = {
  "hardware": [
    {
      "name": "MI300X",
      "gfx": "gfx942",
      "arch": "CDNA3",
      "hbm_gb": 192,
      "hbm_type": "HBM3",
      "mem_bw_tbps": 5.325,
      "fp8_tflops": 2614.9,
      "bf16_tflops": 1307.4,
      "fp4_tflops": null,
      "sparsity_note": "Dense (without sparsity) values, per AMD Performance Labs footnote dated Nov 11 2023. CDNA3 matrix cores do not implement 2:4 structured sparsity acceleration, so these dense figures are the hardware peak. (Some OEM tables, e.g. Lenovo, list a second 'with sparsity' column of 5,220/2,610 FP8/BF16 that is a 2x marketing figure, not a real hardware sparsity mode for MI300X.) No native FP4 support on CDNA3.",
      "source": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf (corroborated by https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)"
    },
    {
      "name": "MI355X",
      "gfx": "gfx950",
      "arch": "CDNA4",
      "hbm_gb": 288,
      "hbm_type": "HBM3E",
      "mem_bw_tbps": 8,
      "fp8_tflops": 5033,
      "bf16_tflops": 2517,
      "fp4_tflops": 10066,
      "sparsity_note": "Dense (without sparsity) values, taken from the left column of AMD's MI355X AI peak-performance table. AMD's datasheet quotes dense/with-sparsity pairs: FP4 (MXFP4) 10,066/20,133, FP8 (MXFP8 & OCP-FP8) 5,033/10,066, BF16 2,517/5,033 TFLOPS. The headline marketing figures (e.g. 20 PFLOPS FP4, 10 PFLOPS FP8 per GPU) are the WITH-sparsity numbers; the dense values reported here are exactly half of those. CDNA4 supports FP6/FP4 (MXFP) datatypes natively.",
      "source": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf (corroborated by https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html and verbatim AMD datasheet mirror)"
    }
  ]
};

window.MODELS = [
  {
    "id": "glm-5.2-fp8",
    "name": "GLM-5.2-FP8",
    "family": "GLM",
    "hf_path": "zai-org/GLM-5.2-FP8",
    "architecture": "MoE + MLA with DeepSeek Sparse Attention (DSA), model_type=glm_moe_dsa (DeepSeek-V3.2 / GLM-5.1 architecture). 1M context window.",
    "precision": "FP8 (block-FP8 weights), bf16 KV cache",
    "status": "verified",
    "params_active": "39B",
    "params_total": "743B",
    "active_params_billions": 39,
    "bytes_per_param": 1,
    "weights_gb": 704,
    "context_len": "1048576",
    "summary": "Single-node TP=8 deployment of zai-org/GLM-5.2-FP8 (MoE + MLA/DSA, glm_moe_dsa) on 8x MI300X (gfx942) with SGLang and the DSA tilelang prefill+decode backend. FP8 weights (704 GB -> 88 GB/GPU) fit single-node; BF16 (~1.4 TB -> ~175 GB/GPU) does not. bf16 KV cache, chunked-prefill 8192, no MTP/speculative decoding on AMD. Verified end-to-end: server, latency, throughput, accuracy (GSM8K 97.2%, AIME25 90.6% pass@1 avg-of-16), and long-context (LongBench-v2 59.5%, near-flat decode TPOT 18.9->24.2 ms from 8k to 256k ctx). The gfx950 block-FP8 GEMM accuracy bug does NOT affect gfx942.",
    "configs": [
      {
        "gfx": "gfx942",
        "hw_name": "MI300X",
        "gpus": 8,
        "quant": "FP8 (block-FP8 MoE weights), bf16 KV cache",
        "strategy": "low-latency",
        "nodes": "single",
        "verified": true,
        "docker_image": "rocm/sgl-dev:v0.5.13.post1-rocm720-mi30x-20260620",
        "launch_python": "export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True\npython3 -m sglang.launch_server \\\n  --model-path zai-org/GLM-5.2-FP8 \\\n  --served-model-name glm-5.2 \\\n  --trust-remote-code \\\n  --tp 8 \\\n  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \\\n  --kv-cache-dtype bfloat16 \\\n  --chunked-prefill-size 8192 \\\n  --mem-fraction-static 0.85 \\\n  --cuda-graph-max-bs 64 --max-running-requests 64 \\\n  --watchdog-timeout 1200 \\\n  --host 0.0.0.0 --port 30000",
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "DSA tilelang (--dsa-prefill-backend tilelang --dsa-decode-backend tilelang)",
        "moe_backend": null,
        "aiter": {
          "enabled": true,
          "commit": "7d604afe",
          "kernels": [
            "GEMM (BF16 tuned GEMM via glm5_bf16_tuned_gemm.csv)"
          ],
          "tuned_artifacts": [
            "glm5_bf16_tuned_gemm.csv"
          ],
          "summary": "AITER is enabled (SGLANG_USE_AITER=1) and ships a tuned BF16 GEMM table (glm5_bf16_tuned_gemm.csv); aiter pinned at commit 7d604afe in the container image. DSA attention itself runs through the tilelang backend, not AITER."
        },
        "env": [
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "Enable AITER kernels; ships the glm5_bf16_tuned_gemm.csv tuned GEMM table."
          },
          {
            "key": "SGLANG_USE_ROCM700A",
            "value": "1",
            "why": "ROCm 7.x build flag set for this verified run on the rocm720 image."
          },
          {
            "key": "SGLANG_MOE_PADDING",
            "value": "1",
            "why": "MoE padding enabled for this run."
          },
          {
            "key": "PYTORCH_HIP_ALLOC_CONF",
            "value": "expandable_segments:True",
            "why": "Reduce HIP allocator fragmentation for the large MoE weights."
          },
          {
            "key": "PYTORCH_ROCM_ARCH",
            "value": "gfx942;gfx950",
            "why": "Image targets both MI300X and MI35x; gfx942 used here."
          }
        ],
        "accuracy": [
          {
            "name": "GSM8K",
            "value": "97.2%",
            "note": "n=1319, chat+thinking; in-tree run_eval --eval-name gsm8k --thinking-mode glm-45 --max-tokens 8192 --temperature 0. Parity, FP8 numerics healthy on gfx942.",
            "ref": "98.2% (cookbook ref)"
          },
          {
            "name": "AIME25",
            "value": "90.6%",
            "note": "pass@1 avg-of-16 via sgl-eval (NV official harness), 95% CI 88.6-92.6; pass@16 100%, majority@16 96.7%, truncated 0%. CAVEAT: in-tree run_eval reports only 62.5% on the same model/server due to a strict Answer: first-match regex -- harness artifact, not the model. Always use sgl-eval for AIME-style answer-extraction evals. Run with --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking.",
            "ref": "87.7% (cookbook ref) -- near parity within noise"
          },
          {
            "name": "LongBench-v2",
            "value": "59.5%",
            "note": "subset, ~64k-tok cap; in-tree run_eval --eval-name longbench_v2 --thinking-mode glm-45 --num-examples 50 --max-context-length 256000. Beats human and o1-preview on the subset.",
            "ref": "human 53.7%, o1-preview 57.7%, best direct model 50.1%"
          }
        ],
        "benchmarks": [
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "prefill_tok_s": 6170,
            "decode_tok_s": 51,
            "tpot_ms": 19.6,
            "total_tok_s": 430.7,
            "source": "glm52_fp8_playbook.md (bench_one_batch, bs=1)",
            "tok_s_per_gpu": null,
            "ttft_ms": null
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "total_tok_s": 47.8,
            "tok_s_per_gpu": 6,
            "tpot_ms": 18.9,
            "ttft_ms": 1458,
            "source": "glm52_fp8_playbook.md (bench_serving)"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 16,
            "total_tok_s": 311.5,
            "tok_s_per_gpu": 38.9,
            "tpot_ms": 41.4,
            "ttft_ms": 10226,
            "source": "glm52_fp8_playbook.md (bench_serving)"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 64,
            "total_tok_s": 528.1,
            "tok_s_per_gpu": 66,
            "tpot_ms": 72.7,
            "ttft_ms": 15849,
            "source": "glm52_fp8_playbook.md (bench_serving)"
          },
          {
            "isl": 32768,
            "osl": 512,
            "concurrency": 1,
            "tpot_ms": 19.5,
            "ttft_ms": 4600,
            "total_tok_s": 38.7,
            "source": "glm52_fp8_playbook.md (long-context)"
          },
          {
            "isl": 131072,
            "osl": 512,
            "concurrency": 1,
            "tpot_ms": 21.5,
            "ttft_ms": 23600,
            "total_tok_s": 19,
            "source": "glm52_fp8_playbook.md (long-context)"
          },
          {
            "isl": 262144,
            "osl": 512,
            "concurrency": 1,
            "tpot_ms": 24.2,
            "ttft_ms": 43800,
            "total_tok_s": 11.5,
            "source": "glm52_fp8_playbook.md (long-context)"
          }
        ],
        "vs_nvidia": [
          {
            "hw": "MI300X (this run, no MTP)",
            "strategy": "tilelang",
            "concurrency": 1,
            "ttft_ms": 1458,
            "tpot_ms": 18.9,
            "tok_s_per_gpu": 6,
            "speculative": "none (no MTP on AMD)"
          },
          {
            "hw": "MI300X (this run, no MTP)",
            "strategy": "tilelang",
            "concurrency": 16,
            "ttft_ms": 10226,
            "tpot_ms": 41.4,
            "tok_s_per_gpu": 38.9,
            "speculative": "none (no MTP on AMD)"
          },
          {
            "hw": "MI300X (this run, no MTP)",
            "strategy": "tilelang",
            "concurrency": 64,
            "ttft_ms": 15849,
            "tpot_ms": 72.7,
            "tok_s_per_gpu": 66,
            "speculative": "none (no MTP on AMD)"
          },
          {
            "hw": "H200",
            "strategy": "low-latency",
            "concurrency": 1,
            "ttft_ms": 662,
            "tpot_ms": 3.03,
            "tok_s_per_gpu": 34,
            "speculative": "EAGLE MTP"
          },
          {
            "hw": "H200",
            "strategy": "low-latency",
            "concurrency": 16,
            "ttft_ms": 5080,
            "tpot_ms": 12.44,
            "tok_s_per_gpu": 113,
            "speculative": "EAGLE MTP"
          },
          {
            "hw": "H200",
            "strategy": "balanced",
            "concurrency": 64,
            "ttft_ms": 8013,
            "tpot_ms": 25.57,
            "tok_s_per_gpu": 219,
            "speculative": "EAGLE MTP"
          },
          {
            "hw": "B300",
            "strategy": "low-latency",
            "concurrency": 1,
            "ttft_ms": 503,
            "tpot_ms": 3.24,
            "tok_s_per_gpu": 34,
            "speculative": "EAGLE MTP"
          },
          {
            "hw": "B300",
            "strategy": "balanced",
            "concurrency": 64,
            "ttft_ms": 6465,
            "tpot_ms": 23.36,
            "tok_s_per_gpu": 245,
            "speculative": "EAGLE MTP"
          },
          {
            "hw": "GB300",
            "strategy": "low-latency",
            "concurrency": 1,
            "ttft_ms": 393,
            "tpot_ms": 2.78,
            "tok_s_per_gpu": 79,
            "speculative": "EAGLE MTP"
          }
        ],
        "gotchas": [
          "KV-cache dtype must be bfloat16 with the DSA tilelang backend -- FP8 KV is incompatible (SGLang auto-selects bf16 and warns if --kv-cache-dtype is omitted).",
          "Keep prefill chunked (--chunked-prefill-size 8192). An unchunked long-context prefill trips the tilelang DSA prefill tile limit: 'RuntimeError: tensor a (16384) must match b (131072)'. Raising the chunk (within the tile limit) improves TTFT.",
          "No MTP / speculative decoding on AMD -- omit all --speculative-* flags. The NV recipe's EAGLE MTP is not enabled on AMD, so AMD decode trails NV ~3-4x.",
          "gfx942 (MI300X) is NOT affected by the block-FP8 GEMM accuracy bug from PR #28471 -- that bug is gfx950 (MI350X/MI355X) ONLY. GSM8K 97.2% confirms healthy numerics on gfx942.",
          "AIME-style accuracy: ALWAYS use sgl-eval (NV official harness), not in-tree run_eval. In-tree run_eval reports only 62.5% vs sgl-eval's 90.6% on the same server, due to its strict ANSWER_PATTERN first-match regex grabbing intermediate 'Answer:' from the reasoning trace.",
          "GLM-5.2 is a thinking model defaulting to effective_reasoning_effort='max' -- budget tokens accordingly (use --max-tokens 64000 for AIME).",
          "bench_one_batch runs the whole batch x input_len as one unchunked forward, so for this DSA model it is bs=1-only at ISL 8192; higher bs hits the tilelang prefill tile limit. Batched throughput must be measured against the live server with chunked prefill.",
          "TTFT in the throughput table is pessimistic because chunked-prefill-size=8192 serializes prefills; decode metrics (TPOT, tok/s) are representative.",
          "BF16 (~1.4 TB -> ~175 GB/GPU) does not fit single-node on MI300X (only MI325X/MI355X). Use FP8 (704 GB -> 88 GB/GPU). No FP4 checkpoint exists.",
          "No DP-attention / DeepEP in this verified config (NV balanced/high-throughput recipes add --dp 8 --enable-dp-attention --moe-a2a-backend deepep); left out for a first-known-good MI300X config."
        ],
        "provenance": {
          "image": "rocm/sgl-dev:v0.5.13.post1-rocm720-mi30x-20260620",
          "pr": "https://github.com/sgl-project/sglang/pull/28471",
          "sglang": "0.5.13.post1.dev20260621 (g3975ea5ac7); built from source @ a51d56d948",
          "aiter": "7d604afe",
          "rocm": "7.2.0",
          "date": "2026-06-20/2026-06-21",
          "node": "8x MI300X (gfx942), 192 GiB each; PyTorch 2.9.1+rocm7.2.0; tilelang 0.1.7.post3"
        }
      }
    ],
    "gaps": [
      {
        "title": "gfx950 (MI355X) FP8",
        "kind": "hardware",
        "note": "Same recipe on MI355X is unmeasured. The block-FP8 GEMM accuracy bug is gfx950-only — re-run GSM8K first to confirm numerics before trusting perf.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \\\n  --thinking-mode glm-45 --max-tokens 8192 --temperature 0 --num-examples 1319"
      },
      {
        "title": "balanced / high-throughput",
        "kind": "strategy",
        "note": "Match the NV balanced recipe: add DP-attention + DeepEP, then sweep concurrency. Relaunch with --dp 8 --enable-dp-attention --moe-a2a-backend deepep, then:",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      },
      {
        "title": "MTP / speculative decode",
        "kind": "dependency",
        "note": "Not enabled on AMD for glm_moe_dsa yet — this is most of the ~3–4× decode gap vs NVIDIA. Tracking upstream; no AMD script until EAGLE/MTP lands on ROCm.",
        "cmd": null
      }
    ]
  },
  {
    "id": "glm-5-fp8",
    "name": "GLM-5-FP8",
    "family": "GLM",
    "hf_path": "zai-org/GLM-5-FP8",
    "architecture": "MoE + Native Sparse Attention (NSA); model_type glm_moe_dsa (described as DeepSeek-V2 architecture in the section). 744B total params, 40B active. NSA backend = tilelang (prefill + decode).",
    "precision": "FP8",
    "status": "not_benchmarked",
    "summary": "FP8 MoE + Native Sparse Attention model (744B total / 40B active), served on 8x MI355X (gfx950) via SGLang TP=8 with tilelang NSA prefill/decode backends. Cookbook gives copy-paste launch commands (TP=8 recommended, TP=4 alternative) plus a verification curl, but contains no measured benchmark, accuracy, or vs-NVIDIA numbers, so nothing is verified.",
    "params_total": "744B",
    "params_active": "40B",
    "active_params_billions": 40,
    "bytes_per_param": 1,
    "weights_gb": 705,
    "context_len": null,
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "quant": "FP8",
        "strategy": "balanced",
        "nodes": "single",
        "verified": false,
        "launch_python": "python3 -m sglang.launch_server \\\n    --model-path zai-org/GLM-5-FP8 \\\n    --served-model-name glm-5-fp8 \\\n    --tp 8 \\\n    --tool-call-parser glm47 \\\n    --reasoning-parser glm45 \\\n    --mem-fraction-static 0.80 \\\n    --nsa-prefill-backend tilelang \\\n    --nsa-decode-backend tilelang \\\n    --chunked-prefill-size 131072 \\\n    --watchdog-timeout 1200 \\\n    --port 30000",
        "aiter": {
          "enabled": false,
          "summary": "No AITER usage in the GLM-5-FP8 section. SGLANG_USE_AITER is not set, no aiter commit hash, no tuned GEMM/MoE artifacts. Attention is NSA via the tilelang backend (prefill + decode); no AITER MLA/GEMM/MoE/attention kernels are invoked for this config.",
          "commit": null,
          "kernels": [],
          "tuned_artifacts": []
        },
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "tilelang (NSA prefill + decode)",
        "moe_backend": null,
        "docker_image": null,
        "env": [],
        "benchmarks": [],
        "accuracy": [],
        "vs_nvidia": [],
        "gotchas": [
          "GLM-5 uses a glm_moe_dsa model_type that stock HuggingFace Transformers does not recognize natively; it is registered in SGLang's config loader. Ensure your SGLang build includes the fix from PR #18911 (Day-0 PR). Also pip install --upgrade transformers for GLM-5 tokenizer support.",
          "TP=8 is strongly recommended. TP=4 (HIP_VISIBLE_DEVICES=0,1,2,3) is a tight fit -- 705 GB model in 1,152 GB -- so drop --mem-fraction-static to 0.60 to leave KV-cache room and add --disable-cuda-graph."
        ],
        "provenance": {
          "image": null,
          "pr": "#18911",
          "sglang": "v0.5.8-v0.5.10rc (cookbook-wide range; not pinned in-section)",
          "aiter": null,
          "rocm": "ROCm 7.0-7.2 (cookbook-wide range; not pinned in-section)",
          "date": null,
          "node": "8x MI355X"
        }
      }
    ],
    "gaps": [
      {
        "title": "Latency (BS=1)",
        "kind": "metric",
        "note": "Documented launch, zero measured numbers. Bring the server up (TP=8 command above) then:",
        "cmd": "# single-request latency (offline)\npython3 -m sglang.bench_one_batch_server \\\n  --model-path zai-org/GLM-5-FP8 --base-url http://127.0.0.1:30000 \\\n  --batch-size 1 --input-len 1024 8192 16384 --output-len 1024 \\\n  --dataset-name random --skip-warmup"
      },
      {
        "title": "Throughput sweep",
        "kind": "metric",
        "note": "Concurrency 1/16/64 against the live server.",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      },
      {
        "title": "Accuracy (GSM8K / AIME25)",
        "kind": "metric",
        "note": "No accuracy yet.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \\\n  --thinking-mode glm-45 --max-tokens 8192 --temperature 0 --num-examples 1319\n\n# AIME25 — use sgl-eval (NV official harness), NOT in-tree run_eval\npip install git+https://github.com/sgl-project/sgl-eval\nsgl-eval run aime25 --api-key EMPTY --base-url http://localhost:30000/v1 \\\n  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking"
      }
    ]
  },
  {
    "id": "deepseek-v4-flash-fp8",
    "name": "DeepSeek-V4-Flash-FP8",
    "family": "DeepSeek",
    "hf_path": "sgl-project/DeepSeek-V4-Flash-FP8",
    "architecture": "MoE + Compressed MLA/MQA (num_key_value_heads=1), 256 routed experts, 43 layers, 1M context, native FP8 e4m3",
    "precision": "FP8 (e4m3)",
    "status": "verified",
    "params_total": null,
    "params_active": null,
    "active_params_billions": null,
    "bytes_per_param": 1,
    "weights_gb": 274,
    "context_len": "1M (max_model_len=1048576)",
    "summary": "DeepSeek-V4-Flash-FP8 (FP8 MoE, 256 routed experts, MQA/Compressed MLA with 1 KV head, 43 layers, 274 GiB weights) served on 8x MI355X (gfx950) via SGLang PR #23608. The \"correctness first\" config runs with every JIT fast-path disabled: torch-reference FlashMLA, Triton-forced MoE-FP8, CUDA graph off, radix cache off, 11 SGLANG_OPT_USE_*=false. Decode is flat at ~4 tok/s at BS=1 because attention is tiny (1 KV head) and the bottleneck is per-token MoE-FP8 matmul plus the pure-PyTorch FlashMLA reference; TP=8 does not help a BS=1 workload since per-expert MoE work is already small. Verified ISL/OSL latency sweep measured with bench_one_batch_server at TP=8 DP=8, BS=1.",
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "nodes": "single",
        "quant": "FP8 (e4m3) weights; MoE via Triton FP8; kv-cache silently fp8_e4m3",
        "strategy": "low-latency",
        "verified": true,
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": 8
        },
        "attention_backend": "compressed (torch FlashMLA reference, SGLANG_HACK_FLASHMLA_BACKEND=torch)",
        "moe_backend": "Triton FP8 (SGLANG_FORCE_TRITON_MOE_FP8=1)",
        "docker_image": "sglang-dsv4-mi355x:flash-r1",
        "launch_python": "python3 -m sglang.launch_server \\\n    --model-path /hf-cache/models--sgl-project--DeepSeek-V4-Flash-FP8/snapshots/ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17 \\\n    --served-model-name dsv4-flash \\\n    --trust-remote-code \\\n    --tp 8 --dp 8 --enable-dp-attention \\\n    --disable-radix-cache --attention-backend compressed \\\n    --max-running-request 256 --page-size 256 --chunked-prefill-size 8192 \\\n    --kv-cache-dtype auto \\\n    --host 0.0.0.0 --port 31000 \\\n    --disable-shared-experts-fusion --disable-cuda-graph \\\n    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4",
        "aiter": {
          "enabled": true,
          "commit": null,
          "kernels": [],
          "tuned_artifacts": [],
          "summary": "SGLANG_USE_AITER=1 is set, but in this correctness-first config AITER does not provide the hot kernels: MoE-FP8 is forced to Triton (SGLANG_FORCE_TRITON_MOE_FP8=1), FlashMLA is forced to a pure-PyTorch reference (SGLANG_HACK_FLASHMLA_BACKEND=torch), and 11 SGLANG_OPT_USE_*=false switches disable every JIT fast-path. No aiter commit hash or tuned GEMM/MoE artifacts are recorded in any source."
        },
        "env": [
          {
            "key": "CUDA_VISIBLE_DEVICES",
            "value": "0,1,2,3,4,5,6,7",
            "why": "All 8 MI355X GPUs for the TP=8 DP=8 benchmarked topology (script default is 0,1,2,3 for TP=4)"
          },
          {
            "key": "SGLANG_OPT_USE_FUSED_COMPRESS",
            "value": "false",
            "why": "Disable fused compress JIT fast-path (correctness-first, no CUDA toolchain port)"
          },
          {
            "key": "SGLANG_OPT_USE_OLD_COMPRESSOR",
            "value": "true",
            "why": "Use the old non-JIT compressor path"
          },
          {
            "key": "SGLANG_OPT_USE_TILELANG_SWA_PREPARE",
            "value": "false",
            "why": "Disable TileLang SWA-prepare JIT kernel"
          },
          {
            "key": "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK",
            "value": "false",
            "why": "Disable JIT fused top-k kernel"
          },
          {
            "key": "SGLANG_OPT_USE_FUSED_HASH_TOPK",
            "value": "false",
            "why": "Disable fused hash top-k kernel"
          },
          {
            "key": "SGLANG_HACK_FLASHMLA_BACKEND",
            "value": "torch",
            "why": "Force pure-PyTorch FlashMLA reference instead of a JIT/CK kernel (correctness, slow)"
          },
          {
            "key": "SGLANG_OPT_DEEPGEMM_HC_PRENORM",
            "value": "false",
            "why": "Disable DeepGEMM HC prenorm fast-path"
          },
          {
            "key": "SGLANG_OPT_USE_TILELANG_MHC_PRE",
            "value": "false",
            "why": "Disable TileLang MHC pre JIT kernel"
          },
          {
            "key": "SGLANG_OPT_USE_TILELANG_MHC_POST",
            "value": "false",
            "why": "Disable TileLang MHC post JIT kernel"
          },
          {
            "key": "SGLANG_ENABLE_THINKING",
            "value": "1",
            "why": "Thinking mode on; model emits <think>...</think>answer"
          },
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "Enable AITER (though MoE/MLA are forced to Triton/torch here)"
          },
          {
            "key": "SGLANG_USE_ROCM700A",
            "value": "1",
            "why": "ROCm 7.0.0-alpha path selection for the MI355X base image"
          },
          {
            "key": "SGLANG_TOPK_TRANSFORM_512_TORCH",
            "value": "1",
            "why": "Use torch path for the 512-wide top-k transform"
          },
          {
            "key": "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH",
            "value": "1",
            "why": "Use torch path for FP8 paged MQA logits"
          },
          {
            "key": "SGLANG_DSV4_FP4_EXPERTS",
            "value": "false",
            "why": "Do not use FP4 experts; keep FP8 experts"
          },
          {
            "key": "SGLANG_OPT_DPSK_V4_RADIX",
            "value": "0",
            "why": "Disable V4 radix optimization (compressed-attention radix not stable on HIP)"
          },
          {
            "key": "SGLANG_OPT_USE_OVERLAP_STORE_CACHE",
            "value": "false",
            "why": "Disable overlap store-cache fast-path"
          },
          {
            "key": "SGLANG_OPT_USE_FUSED_STORE_CACHE",
            "value": "false",
            "why": "Disable fused store-cache fast-path"
          },
          {
            "key": "SGLANG_FORCE_TRITON_MOE_FP8",
            "value": "1",
            "why": "Force Triton MoE-FP8 kernel (the AMD-available MoE path)"
          },
          {
            "key": "HF_HUB_OFFLINE",
            "value": "1",
            "why": "Use local HF cache mount, no network"
          }
        ],
        "benchmarks": [
          {
            "isl": 2048,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 5470,
            "decode_tok_s": 3.99,
            "prefill_tok_s": 374.4,
            "total_tok_s": 11.71,
            "source": "dsv4_flash_playbook.md"
          },
          {
            "isl": 2048,
            "osl": 2048,
            "concurrency": 1,
            "ttft_ms": 5360,
            "decode_tok_s": 3.97,
            "prefill_tok_s": 382.1,
            "total_tok_s": 7.86,
            "source": "dsv4_flash_playbook.md"
          },
          {
            "isl": 4096,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 9530,
            "decode_tok_s": 4.02,
            "prefill_tok_s": 429.9,
            "total_tok_s": 19.37,
            "source": "dsv4_flash_playbook.md"
          },
          {
            "isl": 4096,
            "osl": 2048,
            "concurrency": 1,
            "ttft_ms": 10210,
            "decode_tok_s": 3.99,
            "prefill_tok_s": 401.2,
            "total_tok_s": 11.74,
            "source": "dsv4_flash_playbook.md"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 20030,
            "decode_tok_s": 3.99,
            "prefill_tok_s": 409,
            "total_tok_s": 33.28,
            "source": "dsv4_flash_playbook.md"
          },
          {
            "isl": 8192,
            "osl": 2048,
            "concurrency": 1,
            "ttft_ms": 19490,
            "decode_tok_s": 3.99,
            "prefill_tok_s": 420.3,
            "total_tok_s": 19.2,
            "source": "dsv4_flash_playbook.md"
          }
        ],
        "accuracy": [],
        "vs_nvidia": [],
        "provenance": {
          "image": "sglang-dsv4-mi355x:flash-r1 (base rocm/sgl-dev:deepseek-v4-mi35x, digest sha256:a5f71877...)",
          "pr": "#23608",
          "sglang": "0.5.8.dev20260129+gf959851eb + PR #23608 head 26fbc935300a3bfba34f3dfa8925310929f82680 overlay + 2 AMD patches (drop @dataclass from deepseek_v4.py, stub kernelkit/bench.py)",
          "aiter": null,
          "rocm": "ROCm 7.0 (SGLANG_USE_ROCM700A=1)",
          "date": "April 2026",
          "node": "mia1-p02-g45 (container dsv4-flash-tp8), 8x MI355X"
        },
        "gotchas": [
          "Decode is flat at ~4 tok/s across all ISL/OSL at BS=1 because V4-Flash is MQA (num_key_value_heads=1) so attention is tiny regardless of context; the real bottleneck is per-token MoE-FP8 matmul plus the pure-PyTorch torch-reference FlashMLA. TP=8 gives the same decode speed as TP=4 because per-expert MoE work is already small and all-to-all overhead eats any TP gain. This is the correctness-first config, not a tuned one.",
          "--enable-dp-attention is MANDATORY: V4 uses MQA (1 KV head) so attention cannot be TP-sharded; DP-attention replicates attention across all GPUs.",
          "DP-attention in this image silently auto-lowers --chunked-prefill-size from 8192 to 1024 to dodge an MoE-kernel sizing issue (warning at server_args.py:2057); every benchmark row actually ran at chunked_prefill_size=1024.",
          "--kv-cache-dtype auto is silently overridden to fp8_e4m3 for V4 (server_args.py:1193); the 'no scaling factors provided' warning is cosmetic for short contexts.",
          "The deepseek-v4 reasoning parser does not split <think>...</think> from the answer; both end up in choices[0].message.content and reasoning_content is always empty.",
          "Two AMD-side patches are REQUIRED on top of PR #23608 head 26fbc93 (baked into Dockerfile.dsv4): (1) drop @dataclass from python/sglang/srt/configs/deepseek_v4.py or import fails with TypeError because PretrainedConfig's metaclass strips field defaults; (2) create a stub for python/sglang/srt/flashmla_tests/kernelkit/bench.py (NotImplementedError bench_by_cuda_events / bench_kineto) because kernelkit/__init__.py unconditionally imports it but no file exists in the image or PR branch.",
          "Order-of-magnitude speed-ups require: re-enabling JIT kernels once their ROCm ports land in PR #23608, larger batch sizes (MoE amortizes across tokens, BS=1 is worst case), radix cache re-enabled once compressed-attention radix is stable on HIP, and CUDA-graph replay for decode (ROCm HIP-graph needs driver support).",
          "Served-model-name is 'dsv4-flash'; pass that as the model field in chat calls, not the HF repo id.",
          "Total and active parameter counts are NOT documented in any cookbook source; only 256 routed experts / 43 layers / 1 KV head / FP8 e4m3 / 274 GiB weights are stated. Any '~430B total, ~17B active' figure is unverified and was removed from this record."
        ]
      }
    ],
    "gaps": [
      {
        "title": "Tuned (not correctness-first) config",
        "kind": "perf",
        "note": "Current config disables every JIT fast-path (~4 tok/s decode). Re-enable kernels as their ROCm ports land in PR #23608, then re-bench latency + throughput.",
        "cmd": "# single-request latency (offline)\npython3 -m sglang.bench_one_batch_server \\\n  --model-path sgl-project/DeepSeek-V4-Flash-FP8 --base-url http://127.0.0.1:31000 \\\n  --batch-size 1 --input-len 1024 8192 16384 --output-len 1024 \\\n  --dataset-name random --skip-warmup"
      },
      {
        "title": "Batch > 1 throughput",
        "kind": "metric",
        "note": "BS=1 is the worst case for MoE; sweep concurrency where per-expert work amortizes.",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 31000\ndone"
      },
      {
        "title": "Accuracy (GSM8K / AIME25)",
        "kind": "metric",
        "note": "No accuracy measured.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 31000 --eval-name gsm8k \\\n  --max-tokens 8192 --temperature 0 --num-examples 1319\n\n# AIME25 — use sgl-eval (NV official harness), NOT in-tree run_eval\npip install git+https://github.com/sgl-project/sgl-eval\nsgl-eval run aime25 --api-key EMPTY --base-url http://localhost:31000/v1 \\\n  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking"
      }
    ]
  },
  {
    "id": "qwen35-397b-a17b",
    "name": "Qwen3.5-397B-A17B",
    "family": "Qwen",
    "hf_path": "Qwen/Qwen3.5-397B-A17B",
    "architecture": "Mixture-of-Experts with hybrid DeltaNet attention: 397B total params, 17B active per token. 60 layers total = 45 DeltaNet recurrent (linear-attention) layers + 15 GQA layers. Served BF16.",
    "precision": "BF16",
    "status": "verified",
    "params_total": "397B",
    "params_active": "17B",
    "active_params_billions": 17,
    "bytes_per_param": 2,
    "weights_gb": 752,
    "context_len": null,
    "summary": "Qwen3.5-397B-A17B (397B MoE, 17B active) served BF16 at TP=8 on 8x MI355X (gfx950) via SGLang. Hybrid DeltaNet architecture: 45 recurrent linear-attention layers + 15 GQA layers, run with the triton attention backend (NOT aiter — the verified image is a fixed build that disables the broken aiter stub and patches quark imports). --max-mamba-cache-size 128 is the load-bearing flag (+45% perf vs the default 64) because of the 45 DeltaNet recurrent layers. SGLANG_ROCM_FUSED_DECODE_MLA must be forced to 0 (the base image ships it as 1, which crashes the triton backend with a ForwardMetadata unpacking error). Verified decode ~52-59 tok/s at BS=1 across input lengths 1k-16k. Sub-note: a kernel-engineering fine-tune ships as the JinnP/Qwen3.5-397B-A17B-LoRA-SFT-v4 LoRA adapter (rank 32 / alpha 64, 128.5M trainable params = 0.032%, 13 target module types, 270-example AMD-GPU-kernel-engineering dataset, best eval loss 0.0547 at epoch 8). SGLang runtime LoRA cannot serve it (--lora-paths fails at init_lora_shapes() because the adapter targets 6 unsupported modules across the DeltaNet layers + MoE shared_expert_gate; see sglang#9897). Workaround: merge the adapter offline with LLaMA-Factory (llamafactory-cli export merge_qwen35_lora.yaml, ~25 min on CPU, ~800 GB RAM, output 743 GB / 122 shards) and serve the merged checkpoint with the identical base launch command (just swap --model-path and --served-model-name to Qwen3.5-397B-A17B-SFT-v4). This is NOT a separate verified config — no independent benchmark numbers were measured for the merged model.",
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "nodes": "single",
        "quant": "BF16",
        "strategy": "low-latency",
        "verified": true,
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "triton",
        "moe_backend": null,
        "docker_image": "sglang-test:v0.5.9-rocm700-mi35x-20260310",
        "launch_python": "python3 -m sglang.launch_server \\\n    --model-path /sgl-workspace/models/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/98d1a504ba52e88924b3a3a008447cf2fdbd518c \\\n    --served-model-name Qwen3.5-397B-A17B \\\n    --tp 8 \\\n    --trust-remote-code \\\n    --attention-backend triton \\\n    --mem-fraction-static 0.80 \\\n    --max-mamba-cache-size 128 \\\n    --reasoning-parser qwen3 \\\n    --tool-call-parser qwen3_coder \\\n    --watchdog-timeout 1200 \\\n    --host 0.0.0.0 \\\n    --port 30000",
        "aiter": {
          "enabled": false,
          "summary": "AITER is intentionally disabled for this config. The verified image (sglang-test:v0.5.9-rocm700-mi35x-20260310) is a fixed build produced by Dockerfile.bisect that disables the broken aiter stub and patches quark imports; attention runs on the triton backend instead. No aiter commit, tuned GEMM CSV, or MoE bucket config applies.",
          "commit": null,
          "kernels": [],
          "tuned_artifacts": []
        },
        "env": [
          {
            "key": "SGLANG_ROCM_FUSED_DECODE_MLA",
            "value": "0",
            "why": "Base image ships this as 1, which crashes the triton attention backend with a ForwardMetadata unpacking error; must be forced to 0."
          },
          {
            "key": "HF_HUB_OFFLINE",
            "value": "1",
            "why": "Serve from locally mounted weights without contacting the HF Hub."
          }
        ],
        "benchmarks": [
          {
            "isl": 1024,
            "osl": 512,
            "concurrency": 1,
            "decode_tok_s": 58.9,
            "ttft_ms": 110,
            "source": "index.html"
          },
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 1,
            "decode_tok_s": 59.17,
            "ttft_ms": 110,
            "source": "index.html"
          },
          {
            "isl": 8192,
            "osl": 512,
            "concurrency": 1,
            "decode_tok_s": 55.71,
            "ttft_ms": 310,
            "source": "index.html"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "decode_tok_s": 56.68,
            "ttft_ms": 250,
            "source": "index.html"
          },
          {
            "isl": 16384,
            "osl": 512,
            "concurrency": 1,
            "decode_tok_s": 52.32,
            "ttft_ms": 510,
            "source": "index.html"
          },
          {
            "isl": 16384,
            "osl": 1024,
            "concurrency": 1,
            "decode_tok_s": 53.79,
            "ttft_ms": 450,
            "source": "index.html"
          }
        ],
        "accuracy": [],
        "vs_nvidia": [],
        "gotchas": [
          "--max-mamba-cache-size 128 is critical for DeltaNet's 45 recurrent layers (+45% perf vs the default 64).",
          "SGLANG_ROCM_FUSED_DECODE_MLA must be set to 0 — the base image ships it as 1, which crashes the triton backend with a ForwardMetadata unpacking error.",
          "The stock rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260310 base image has a broken aiter stub; use the fixed build (Dockerfile.bisect) that disables aiter and patches quark imports.",
          "DeltaNet's 45 recurrent layers leak memory over long runtime, causing NCCL process-group failures after ~10-12 hours (sglang issue #20010 / PR #20182). Workaround: periodic docker restart qwen35-serve.",
          "Use --ulimit core=0:0 to prevent GPU core dumps (~200 GB each) from filling the disk on crash.",
          "CUDA graph is enabled (default) — left on for this verified config.",
          "Use max_tokens >= 512 when testing inference; it is a reasoning model.",
          "LoRA-SFT-v4: SGLang runtime LoRA (--lora-paths) fails at init_lora_shapes() because the adapter targets DeltaNet modules (in_proj_a/b/z/qkv, out_proj) and MoE shared_expert_gate that SGLang LoRA does not support (sglang#9897). Merge offline with LLaMA-Factory and serve the merged checkpoint instead."
        ],
        "provenance": {
          "image": "sglang-test:v0.5.9-rocm700-mi35x-20260310",
          "pr": null,
          "sglang": "v0.5.9",
          "aiter": "disabled (Dockerfile.bisect stub)",
          "rocm": "7.0",
          "date": "March 2026",
          "node": "8x MI355X"
        }
      }
    ],
    "gaps": [
      {
        "title": "Throughput / concurrency sweep",
        "kind": "metric",
        "note": "Only BS=1 latency is measured; add the online sweep.",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      },
      {
        "title": "Accuracy (GSM8K / AIME25)",
        "kind": "metric",
        "note": "No accuracy yet.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \\\n  --max-tokens 8192 --temperature 0 --num-examples 1319\n\n# AIME25 — use sgl-eval (NV official harness), NOT in-tree run_eval\npip install git+https://github.com/sgl-project/sgl-eval\nsgl-eval run aime25 --api-key EMPTY --base-url http://localhost:30000/v1 \\\n  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking"
      },
      {
        "title": "Merged LoRA-SFT-v4 numbers",
        "kind": "variant",
        "note": "The kernel-engineering adapter cannot serve via runtime LoRA; merge it offline, then benchmark the merged checkpoint with the identical launch (swap --model-path / --served-model-name).",
        "cmd": "llamafactory-cli export merge_qwen35_lora.yaml   # ~25 min CPU, ~800 GB RAM, 743 GB out\n# then relaunch with the merged path and re-run the latency + accuracy commands above"
      }
    ]
  },
  {
    "id": "kimi-k2.6",
    "name": "Kimi-K2.6",
    "family": "Moonshot",
    "hf_path": "moonshotai/Kimi-K2.6",
    "architecture": "Mixture-of-Experts (MLA attention, 384 routed experts), 1T total params / 32B active per token",
    "precision": "W4A16",
    "status": "verified",
    "summary": "Kimi-K2.6 (1T MoE, 32B active, W4A16, 384 routed experts) verified on 8x MI355X (gfx950) with the prebuilt jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420 image. Default TP=8 EP=1 hits the pre-tuned 13-bucket E=384,N=128 int4_w4a16 MoE configs; triton MLA for decode + aiter for prefill. Measured BS=1 single-request decode 34-45 tok/s across 1k-32k context. EP8/tp2ep4/tp4ep2 mori-a2a variants exist but lack tuned N=2048 configs and are slower at BS=1.",
    "params_total": "1T",
    "params_active": "32B",
    "active_params_billions": 32,
    "bytes_per_param": 0.5,
    "weights_gb": 555,
    "context_len": "32768 (max tested context in benchmark; model supports longer)",
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "quant": "W4A16 (int4_w4a16 MoE)",
        "strategy": "low-latency",
        "nodes": "single",
        "verified": true,
        "docker_image": "jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420",
        "attention_backend": "triton (decode) / aiter (prefill)",
        "moe_backend": "TP-sharded MoE (ep-size 1), pre-tuned int4_w4a16 E=384,N=128 configs",
        "parallelism": {
          "tp": 8,
          "ep": 1,
          "dp": null
        },
        "launch_python": "python3 -m sglang.launch_server \\\n    --model-path /hf-cache/models--moonshotai--Kimi-K2.6/snapshots/<rev> \\\n    --served-model-name kimi-k2.6 \\\n    --tensor-parallel-size 8 \\\n    --ep-size 1 \\\n    --trust-remote-code \\\n    --reasoning-parser kimi_k2 \\\n    --tool-call-parser kimi_k2 \\\n    --decode-attention-backend triton \\\n    --prefill-attention-backend aiter \\\n    --host 0.0.0.0 \\\n    --port 30000",
        "env": [
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "Enable AITER kernels (aiter MLA TP=8 fix + tuned int4_w4a16 MoE configs ship in this image)"
          },
          {
            "key": "SGLANG_ROCM_FUSED_DECODE_MLA",
            "value": "0",
            "why": "Avoids the triton MLA tuple-unpack crash during fused decode MLA on ROCm"
          },
          {
            "key": "SGLANG_DEEPSEEK_LOAD_MAX_WORKERS",
            "value": "4",
            "why": "Keeps weight-load RAM pressure bounded while loading the 1T MoE checkpoint"
          },
          {
            "key": "HF_HUB_OFFLINE",
            "value": "1",
            "why": "Use the local HF cache snapshot, no network fetch"
          }
        ],
        "aiter": {
          "enabled": true,
          "commit": "3125d3b01",
          "kernels": [
            "MLA",
            "MoE",
            "attention (prefill)"
          ],
          "tuned_artifacts": [
            "13-bucket E=384,N=128,...MI355X,int4_w4a16 MoE configs",
            "aiter 3125d3b01 MLA TP=8 fix"
          ],
          "summary": "Prebuilt image ships aiter commit 3125d3b01 with the MLA TP=8 fix and 13-bucket E=384,N=128 int4_w4a16 MoE configs tuned for MI355X; aiter provides the heavy prefill attention and the TP-sharded MoE GEMMs, while decode MLA runs on triton."
        },
        "benchmarks": [
          {
            "isl": 1024,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 300,
            "decode_tok_s": 45.23,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          },
          {
            "isl": 2048,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 350,
            "decode_tok_s": 44.69,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          },
          {
            "isl": 4096,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 430,
            "decode_tok_s": 43.62,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          },
          {
            "isl": 8192,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 650,
            "decode_tok_s": 41.88,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          },
          {
            "isl": 16384,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 1100,
            "decode_tok_s": 38.93,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          },
          {
            "isl": 32768,
            "osl": 1024,
            "concurrency": 1,
            "ttft_ms": 2230,
            "decode_tok_s": 34,
            "total_tok_s": null,
            "source": "kimi_k26_playbook.md / index.html"
          }
        ],
        "accuracy": [],
        "vs_nvidia": [],
        "gotchas": [
          "SGLANG_ROCM_FUSED_DECODE_MLA=0 is required - fused decode MLA triton path hits a tuple-unpack crash on ROCm.",
          "Keep --ep-size 1 for BS=1: it keeps MoE TP-sharded across 8 ranks and hits the pre-tuned E=384,N=128 int4_w4a16 config.",
          "EP variants (ep8 / tp2ep4 / tp4ep2) all need --moe-a2a-backend mori plus the mori env group (SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384, MORI_SHMEM_MODE=vmm, MORI_SHMEM_HEAP_SIZE=34359738368, TORCH_NCCL_BLOCKING_WAIT=0, NCCL_ASYNC_ERROR_HANDLING=0) and use --disable-cuda-graph --skip-server-warmup --watchdog-timeout 1800 --dist-timeout 3600. tp2ep4 adds --moe-dp-size 2 (--ep-size 4); tp4ep2 adds --moe-dp-size 4 (--ep-size 2).",
          "EP variants lack tuned MoE configs for N=2048 (only the default TP=8 N=128 config ships), so they fall back to generic kernels and are slower at BS=1.",
          "--reasoning-parser kimi_k2 --tool-call-parser kimi_k2 are needed to split <think> blocks and DSML-style tool calls out of choices[0].message.content.",
          "Decode runs on triton MLA; prefill on aiter - this split is the MI355X sweet spot for best TTFT.",
          "The separate Dockerfile.kimi-opt build (base v0.5.9-rocm700-mi35x) stubs aiter and sets SGLANG_USE_AITER=0; it is NOT the verified serving image - use the prebuilt v0.5.10rc0-rocm720 image with SGLANG_USE_AITER=1 instead."
        ],
        "provenance": {
          "image": "jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420",
          "pr": "#19552 (Kimi-K2/K2.5 tool-call parser fixes; applied in Dockerfile.kimi-opt build, not the prebuilt serving image)",
          "sglang": "v0.5.10rc0",
          "aiter": "3125d3b01",
          "rocm": "7.2.0",
          "date": "April 2026 (image tag 20260420)",
          "node": "8x AMD Instinct MI355X (gfx950), single node"
        }
      }
    ],
    "gaps": [
      {
        "title": "Accuracy (GSM8K / AIME25)",
        "kind": "metric",
        "note": "No accuracy yet.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \\\n  --max-tokens 8192 --temperature 0 --num-examples 1319\n\n# AIME25 — use sgl-eval (NV official harness), NOT in-tree run_eval\npip install git+https://github.com/sgl-project/sgl-eval\nsgl-eval run aime25 --api-key EMPTY --base-url http://localhost:30000/v1 \\\n  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking"
      },
      {
        "title": "Throughput sweep",
        "kind": "metric",
        "note": "Only BS=1 latency measured.",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      },
      {
        "title": "EP variants (ep8 / tp2ep4 / tp4ep2)",
        "kind": "strategy",
        "note": "Need N=2048 tuned MoE configs; today they fall back to generic kernels and lose at BS=1. Launch a variant, then sweep:",
        "cmd": "TAG=ep8 bash test_kimi_k26.sh    # or tp2ep4 / tp4ep2\n\n# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      }
    ]
  },
  {
    "id": "kimi-k2.5",
    "name": "Kimi-K2.5",
    "family": "Moonshot",
    "hf_path": "moonshotai/Kimi-K2.5",
    "architecture": "Mixture-of-Experts (MoE) with MLA attention; 384 experts (E=384, N=128), W4A16 INT4 quantized weights",
    "precision": "W4A16 (INT4 weight, A16 activation)",
    "params_total": "1T",
    "params_active": "32B",
    "active_params_billions": 32,
    "bytes_per_param": 0.5,
    "weights_gb": 555,
    "context_len": "131072",
    "status": "verified",
    "summary": "Kimi-K2.5 (1T total / 32B active W4A16 INT4 MoE + MLA) served on 8x MI355X (gfx950) at TP=8. Optimized config hits 23.5ms decode median (42.6 tok/s) at BS=1 vs 38.3ms baseline (+38.6%) using hybrid attention (triton decode + aiter prefill), GEMM A16W16 small-M tuning, and MoE Triton config tuning. Optional Eagle3 speculative decoding adds ~1.8x on short-context coding/math.",
    "configs": [
      {
        "gfx": "gfx950",
        "hw_name": "MI355X",
        "gpus": 8,
        "nodes": "single",
        "quant": "W4A16 INT4",
        "strategy": "low-latency",
        "verified": true,
        "docker_image": "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317",
        "launch_python": "/opt/venv/bin/python3 -m sglang.launch_server \\\n    --model-path moonshotai/Kimi-K2.5 \\\n    --tp 8 \\\n    --trust-remote-code \\\n    --decode-attention-backend triton \\\n    --prefill-attention-backend aiter \\\n    --mem-fraction-static 0.85 \\\n    --reasoning-parser kimi_k2 \\\n    --tool-call-parser kimi_k2 \\\n    --host 0.0.0.0 --port 30000",
        "parallelism": {
          "tp": 8,
          "ep": null,
          "dp": null
        },
        "attention_backend": "triton decode + aiter prefill (hybrid)",
        "moe_backend": "triton (E=384, N=128 tuned configs)",
        "aiter": {
          "enabled": true,
          "commit": null,
          "summary": "AITER provides the prefill attention path (ASM kernels) plus optimized GEMM A16W16 small-M configs for M=1 decode; built from the Arist12/aiter:kimi-k25-optimize-v2 branch with non-editable pip install.",
          "kernels": [
            "prefill attention (aiter ASM)",
            "GEMM A16W16 small-M"
          ],
          "tuned_artifacts": [
            "GEMM A16W16 small-M configs (M_LEQ_4/8/16/32/64 with BLOCK_SIZE_M=16-64, default was 256)",
            "Arist12/aiter:kimi-k25-optimize-v2 branch"
          ]
        },
        "env": [
          {
            "key": "SGLANG_USE_AITER",
            "value": "1",
            "why": "enables the aiter prefill attention path"
          },
          {
            "key": "SGLANG_ROCM_FUSED_DECODE_MLA",
            "value": "0",
            "why": "required; the image default is 1 and must be disabled for this hybrid attention config"
          },
          {
            "key": "GPU_COREDUMP_ENABLE",
            "value": "0",
            "why": "set on docker run to disable GPU coredumps"
          }
        ],
        "benchmarks": [
          {
            "isl": 8192,
            "osl": 2048,
            "concurrency": 1,
            "decode_tok_s": 42.6,
            "tpot_ms": 23.5,
            "prefill_tok_s": 12847,
            "ttft_ms": 637,
            "source": "index.html",
            "total_tok_s": null,
            "tok_s_per_gpu": null
          },
          {
            "isl": 1024,
            "osl": 2048,
            "concurrency": 1,
            "decode_tok_s": 45.19,
            "ttft_ms": 270,
            "source": "index.html",
            "tpot_ms": null,
            "prefill_tok_s": null,
            "total_tok_s": null,
            "tok_s_per_gpu": null
          },
          {
            "isl": 2048,
            "osl": 2048,
            "concurrency": 1,
            "decode_tok_s": 44.67,
            "ttft_ms": 330,
            "source": "index.html",
            "tpot_ms": null,
            "prefill_tok_s": null,
            "total_tok_s": null,
            "tok_s_per_gpu": null
          }
        ],
        "accuracy": [],
        "gotchas": [
          "aiter must be installed with 'pip install .' (non-editable). 'pip install -e .' creates a broken namespace package that fails to resolve compiled C extensions. Verify with: python3 -c \"from aiter import dynamic_per_tensor_quant; print('OK')\".",
          "SGLANG_ROCM_FUSED_DECODE_MLA=0 is required; the image default is 1 and the hybrid attention config breaks otherwise.",
          "MoE Triton config tuning is the final optimization rung in the ladder: baseline triton attn 38.3ms -> +aiter prefill 34.4ms (10.2%) -> +GEMM A16W16 small-M tuning 24.3ms (36.6%) -> +MoE Triton config tuning 23.5ms / 42.6 tok/s (38.6%).",
          "Eagle3 must use --speculative-algorithm EAGLE3 (not EAGLE); EAGLE silently degrades accept_length to 1.0. Eagle3 also needs --mem-fraction-static 0.75 (down from 0.85) for the draft model, and accept_length degrades to 1.6-2.0 at 8K+ input tokens making it slower than baseline on long context.",
          "Eagle3 non-greedy (temp>0) on ROCm needs PyTorch fallback kernels for 3 missing sgl_kernel C++ ops via patch_eagle_rocm.py (or PR #21275); without it temp>0 silently falls back to greedy."
        ],
        "provenance": {
          "image": "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317",
          "pr": null,
          "sglang": "Arist12/sglang:kimi-k25-optimize-v2 (MoE Triton configs E=384,N=128, BLOCK_SIZE_M=16 for batch=1 decode)",
          "aiter": "Arist12/aiter:kimi-k25-optimize-v2 (GEMM A16W16 small-M configs)",
          "rocm": "7.2",
          "date": "2026-03 (March 2026)",
          "node": "8x MI355X"
        },
        "vs_nvidia": []
      }
    ],
    "gaps": [
      {
        "title": "Accuracy (GSM8K / AIME25)",
        "kind": "metric",
        "note": "No accuracy yet.",
        "cmd": "# GSM8K (chat + thinking)\npython3 -m sglang.test.run_eval --port 30000 --eval-name gsm8k \\\n  --max-tokens 8192 --temperature 0 --num-examples 1319\n\n# AIME25 — use sgl-eval (NV official harness), NOT in-tree run_eval\npip install git+https://github.com/sgl-project/sgl-eval\nsgl-eval run aime25 --api-key EMPTY --base-url http://localhost:30000/v1 \\\n  --n-repeats 16 --max-tokens 64000 --temperature 1.0 --top-p 0.95 --thinking"
      },
      {
        "title": "Throughput sweep",
        "kind": "metric",
        "note": "Only single-stream decode measured.",
        "cmd": "# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      },
      {
        "title": "Eagle3 speculative decode",
        "kind": "perf",
        "note": "Relaunch with --speculative-algorithm EAGLE3 (not EAGLE) + --mem-fraction-static 0.75; measure accept_length and speedup on short vs long context.",
        "cmd": "# add to launch: --speculative-algorithm EAGLE3 --mem-fraction-static 0.75\n# throughput vs concurrency (online)\nfor C in 1 16 64; do\n  python3 -m sglang.bench_serving --backend sglang --dataset-name random \\\n    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \\\n    --num-prompts $((C*2)) --max-concurrency $C --port 30000\ndone"
      }
    ]
  }
];
