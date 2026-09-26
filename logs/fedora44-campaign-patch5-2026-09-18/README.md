# The split throughput campaign on the five-patch build, 2026-09-18

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-mmvq2/bin`, 10:24 to 10:52 on the default boot: Fedora 44, kernel 7.2.5
with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the corrected comgr, clock policy 1500 MHz,
`ollama` stopped, `hw-watcher.timer` and `crond` stopped. Same script, rounds and models as the three
campaigns before it. The build is llama.cpp 7ba604f with the five patches as committed; patch 5 in its
final form gives RDNA1 its own matrix-vector table entry, type-aware, plus the `v_sad_u8` activation sums
([`logs/rdna1-mmvq-2026-09-18/`](../rdna1-mmvq-2026-09-18/)). It was the campaign the front page showed
until the float-activation matvec replaced it
([`logs/fedora44-campaign-f32mv-2026-09-18/`](../fedora44-campaign-f32mv-2026-09-18/)).
Medians of nine samples:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 925.2 | 1850.8 | 154.3 | 212.4 |
| qwen3-8B Q8_0 | 279.6 | 394.8 | 38.4 | 39.1 |
| deepseek-r1-14B Q4_K_M | 98.7 | 199.8 | 25.5 | 35.0 |
| qwen3-14B Q4_K_M | 100.5 | 204.8 | 25.8 | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | 303.9 | 457.1 | 54.5 | 86.8 |
| qwen3.8-27B UD-IQ3_XXS | 71.2 | 105.0 | 11.4 | 17.6 |

Ratios of medians against the four-patch campaign
([`logs/fedora44-campaign-final-2026-09-18/`](../fedora44-campaign-final-2026-09-18/)), decode: 1.5B
**1.294**, 8B 0.983, deepseek-14B **1.187**, qwen3-14B **1.182**, MoE **1.570**, 27B **1.448**; prefill
1.000 to 1.002 everywhere; Vulkan 0.998 to 1.001. Against the first form of patch 5
([`logs/fedora44-campaign-five-patches-2026-09-18/`](../fedora44-campaign-five-patches-2026-09-18/)) the
only movement is the 8B, 37.3 to 38.4, which the type-aware entry was made for; the 8B's decode spread
across nine samples is 2 to 3 percent in every campaign here, and 38.4 against the four-patch 39.0 is
inside it.

Spread, max minus min over nine samples as a percentage of the median: ROCm prefill 0.3 to 1.4, ROCm
decode 0.8 to 4.5, Vulkan prefill 0.0 to 1.8, Vulkan decode 0.2 to 2.4. `log` is the campaign log;
`hip_*` and `vk_*` the `llama-bench` JSONL outputs behind each line.
