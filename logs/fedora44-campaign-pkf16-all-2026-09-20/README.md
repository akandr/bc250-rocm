# The split throughput campaign with every supported type in the prefill GEMM, 2026-09-20

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-pkf16/bin`, 08:55 to 09:34, the usual configuration: Fedora 44, kernel
7.2.5 with the bc250 amdgpu module, native gfx1013 rocBLAS, corrected comgr, rebuilt ROCr and HIP, 1500
MHz, `ollama` stopped, timers stopped. The build is llama.cpp 7ba604f with the eight patches, the last of
them now covering q4_K, q5_K, q6_K, IQ2_XXS, IQ3_XXS, IQ3_S and IQ4_XS
([`logs/round10-2026-09-20/`](../round10-2026-09-20/)). It was the campaign the front page showed until
the IQ codebooks moved into shared memory
([`logs/fedora44-campaign-iq-shmem-2026-09-20/`](../fedora44-campaign-iq-shmem-2026-09-20/)).
Medians of nine samples:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 | prefill vs Vulkan |
|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1253.3 | 1849.9 | 197.4 | 212.2 | 0.68 |
| qwen3-8B Q8_0 | 279.6 | 394.8 | 37.0 | 39.0 | 0.71 |
| deepseek-r1-14B Q4_K_M | 145.5 | 199.9 | 32.4 | 35.0 | 0.73 |
| qwen3-14B Q4_K_M | 151.5 | 204.6 | 32.4 | 34.7 | 0.74 |
| qwen3.6-35B-A3B MoE IQ2_M | 340.2 | 457.3 | 70.0 | 86.8 | 0.74 |
| qwen3.8-27B UD-IQ3_XXS | 81.2 | 104.9 | 14.8 | 17.6 | **0.77** |

## Against the q4_K-and-q6_K build of the same day, and against three patches

| model | over q4/q6 only, pp512 | over three patches, pp512 | tg64 |
|---|---|---|---|
| qwen2.5-1.5B | 1.001 | 1.581 | 1.680 |
| qwen3-8B | 1.008 | 1.148 | 0.951 |
| deepseek-r1-14B | 1.000 | 1.541 | 1.515 |
| qwen3-14B | 1.000 | 1.565 | 1.487 |
| qwen3.6-35B MoE | **1.042** | 1.174 | 2.041 |
| qwen3.8-27B | **1.115** | 1.170 | 1.884 |

The two codebook models are the ones the new decoders were for and the only ones that move: the 27B by
11.5 percent and the MoE by 4.2. Decode is unchanged everywhere within its spread; the 8B's tg64 reads 4
percent under the previous campaign, which is the session-to-session wander that
[`logs/round3-2026-09-19/`](../round3-2026-09-19/) measured directly on that model, not anything
this build does, since nothing in the change touches q8_0 or decode.

Prefill against Vulkan is 0.68 to 0.77 across the six models, where the three-patch build was 0.44 to
0.66. Every Vulkan row reproduces the ten earlier campaigns within 0.5 percent.
