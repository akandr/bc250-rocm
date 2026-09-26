# The split throughput campaign with the packed-fp16 prefill GEMM, 2026-09-20

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-pkf16/bin`, 00:14 to 00:42: Fedora 44, kernel 7.2.5 with the bc250
amdgpu module, native gfx1013 rocBLAS 7.1.1, the corrected comgr and the rebuilt ROCr and HIP runtimes
from `/opt/bc250-rocm`, GPU clock policy at 1500 MHz, `ollama` stopped, timers stopped for the window.
Same script, rounds and models as every campaign before it. The build is llama.cpp 7ba604f with the
eight patches: the seven of
[`logs/fedora44-campaign-seven-patches-2026-09-19/`](../fedora44-campaign-seven-patches-2026-09-19/) and
the packed-fp16 prefill GEMM of [`logs/round7-2026-09-20/`](../round7-2026-09-20/). It was the campaign
the front page showed until the tile, q8_0 and expert-path changes of 20 and 21 September replaced it.
Medians of nine samples:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1252.3 | 1850.0 | 197.1 | 212.5 |
| qwen3-8B Q8_0 | 277.4 | 394.7 | 38.5 | 39.0 |
| deepseek-r1-14B Q4_K_M | 145.5 | 199.8 | 32.4 | 35.0 |
| qwen3-14B Q4_K_M | 151.4 | 204.6 | 32.4 | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | 326.5 | 457.2 | 69.9 | 86.6 |
| qwen3.8-27B UD-IQ3_XXS | 72.8 | 105.0 | 14.8 | 17.6 |

## Against the previous campaigns

| model | over seven patches, pp512 | tg64 | over three patches, pp512 | tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B | **1.353** | 1.009 | 1.579 | 1.678 |
| qwen3-8B | 0.992 | 1.033 | 1.139 | 0.989 |
| deepseek-r1-14B | **1.474** | 0.999 | 1.541 | 1.516 |
| qwen3-14B | **1.507** | 0.999 | 1.565 | 1.485 |
| qwen3.6-35B MoE | **1.036** | 0.993 | 1.127 | 2.037 |
| qwen3.8-27B | 1.000 | 0.996 | 1.049 | 1.882 |

The models whose weights the kernel covers gain 35 to 51 percent of prefill; the MoE gains the 3.6
percent its dense q4_K and q6_K tensors are worth, and the 27B, which is IQ3_S, IQ3_XXS and IQ4_XS
throughout, is unchanged to three decimal places. Decode does not move on any model, as it should not:
the kernel only takes batches of 256 tokens and up. Every Vulkan row reproduces the nine earlier
campaigns within 0.5 percent.

Prefill against Vulkan is now 0.68 to 0.74 across the six models, where the seven-patch build was 0.48
to 0.71 and the three-patch build 0.44 to 0.66.

## Spread

Max minus min over the nine samples, as a percentage of the median: ROCm prefill 0.3 to 1.7, ROCm decode
0.6 to 5.1, Vulkan prefill 0.1 to 1.7, Vulkan decode 0.1 to 2.5.

## Files

`log` is the campaign log; `hip_*` and `vk_*` the `llama-bench` JSONL behind each line; `round9-log` the
threshold check that precedes it (pp128 identical to MMQ, pp256 and pp512 ahead) and the perplexity gate.
Both were run by [`scripts/round9.sh`](../../scripts/round9.sh).
