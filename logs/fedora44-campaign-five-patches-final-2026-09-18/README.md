# The split throughput campaign on the final five-patch build, 2026-09-18

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-f32mv/bin`, 15:02 to 15:45, same boot, clock policy, script, rounds and
models as the five campaigns before it; `hw-watcher.timer` and `crond` stopped. The build is llama.cpp
7ba604f with the five patches exactly as committed, patch 5 in its final form: the type-aware RDNA1
matrix-vector entry with its long-K row for the K-quants, the `v_sad_u8` activation sums, and the
float-activation q4_K matvec ([`logs/rdna1-mmvq-2026-09-18/`](../rdna1-mmvq-2026-09-18/)). It was the
campaign the front page showed until q6_K and q8_0 joined the float matvec
([`logs/fedora44-campaign-float-all-2026-09-18/`](../fedora44-campaign-float-all-2026-09-18/)).
Medians of nine samples:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 925.2 | 1850.5 | **176.8** | 212.6 |
| qwen3-8B Q8_0 | 279.6 | 394.8 | 37.6 | 39.0 |
| deepseek-r1-14B Q4_K_M | 98.7 | 199.8 | **29.2** | 35.0 |
| qwen3-14B Q4_K_M | 100.4 | 204.8 | **29.8** | 34.8 |
| qwen3.6-35B-A3B MoE IQ2_M | 303.7 | 456.9 | **57.0** | 86.9 |
| qwen3.8-27B UD-IQ3_XXS | 71.1 | 105.0 | 11.4 | 17.6 |

## Against the campaigns before it

Ratios of medians, decode (prefill is 0.999 to 1.000 against every earlier ROCm campaign since the
fourth patch, and Vulkan 0.995 to 1.002 throughout):

| model | over three patches | over four | over patch 5 before the float matvec | over the run with the long-K row on IQ types |
|---|---|---|---|---|
| qwen2.5-1.5B | **1.504** | 1.146 | 1.146 | 0.997 |
| qwen3-8B | 0.966 | 0.964 | 0.980 | 1.002 |   <!-- the decode ratio is a session difference, see logs/round3-2026-09-19 -->
| deepseek-r1-14B | **1.366** | 1.145 | 1.145 | 1.001 |
| qwen3-14B | **1.367** | 1.154 | 1.154 | 1.000 |
| qwen3.6-35B MoE | **1.662** | 1.045 | 1.045 | 1.005 |
| qwen3.8-27B | **1.454** | 1.000 | 1.000 | **1.038** |

The last column is the correction this campaign was run for: restricting the long-K row to the K-quants
returns the IQ3_XXS 27B to 11.4 from 11.0. The 8B's decode spread is 7.7 percent in this campaign and 2
to 8 in every campaign here, so its 37.6 against the three-patch 39.0 is inside the noise of that model;
it is the one model whose type none of the decode changes targets, and it tied Vulkan before them.

Spread, max minus min over nine samples as a percentage of the median: ROCm prefill 0.3 to 4.3 (the 8B),
ROCm decode 0.5 to 7.7 (the 8B), Vulkan prefill 0.0 to 1.7, Vulkan decode 0.1 to 1.9. `log` is the
campaign log; `hip_*` and `vk_*` the `llama-bench` JSONL outputs behind each line.
