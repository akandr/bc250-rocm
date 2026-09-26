# The campaign on the ten-patch build, 2026-09-20

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-pkf16/bin`, run twice, 20:05 to 20:45 and 20:46 to 21:12 on the default
boot: Fedora 44, kernel 7.2.5 with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the
corrected comgr from `/opt/bc250-rocm`, GPU clock policy at 1500 MHz with `oberon-governor` active,
`ollama` stopped, the scraper timer and cron stopped for the window. The build is llama.cpp 7ba604f with
the ten patches as committed; the tenth is
[`patches/llamacpp/0010-rdna1-pkf16-halve-column-tile.patch`](../../patches/llamacpp/0010-rdna1-pkf16-halve-column-tile.patch),
which halves the prefill GEMM's column tile ([`logs/rdna1-pkf16-tile-2026-09-20/`](../rdna1-pkf16-tile-2026-09-20/)).

It was the campaign the front page showed until q8_0 joined the GEMM
([`logs/fedora44-campaign-q8-2026-09-20/`](../fedora44-campaign-q8-2026-09-20/)). The two runs are
pooled, so each figure is the median of
eighteen samples; `log` is the first run and `second/log` the second.

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1796.1 | 1849.9 | 196.5 | 212.4 |
| qwen3-8B Q8_0 | 279.5 | 394.7 | 37.8 | 39.1 |
| deepseek-r1-14B Q4_K_M | 191.8 | 199.8 | 32.5 | 35.0 |
| qwen3-14B Q4_K_M | 192.9 | 204.7 | 32.5 | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | 376.9 | 457.1 | 70.4 | 86.7 |
| qwen3.8-27B UD-IQ3_XXS | 102.5 | 105.0 | 15.1 | 17.6 |

## Against the nine-patch build

Ratios of medians over [`logs/fedora44-campaign-iq-shmem-2026-09-20/`](../fedora44-campaign-iq-shmem-2026-09-20/),
the same script and models four hours earlier:

| model | pp512 | tg64 | Vulkan pp512 / tg64 |
|---|---|---|---|
| qwen2.5-1.5B | **1.437** | 0.994 | 1.000 / 1.000 |
| qwen3-8B | 1.007 | 0.980 | 1.000 / 1.000 |
| deepseek-r1-14B | **1.315** | 0.998 | 1.000 / 1.000 |
| qwen3-14B | **1.273** | 0.998 | 1.000 / 0.999 |
| qwen3.6-35B MoE | **1.107** | 0.998 | 1.000 / 1.000 |
| qwen3.8-27B | **1.261** | 1.000 | 1.000 / 1.000 |

Prefill moves everywhere the GEMM runs and decode does not move at all, which is what a change to the
prefill tile should do. The 8B is the exception on prefill because its weights are q8_0, which the GEMM
deliberately leaves to MMQ; its 0.980 on decode sits inside a 6.7 percent spread on that row. ROCm
prefill now runs at 0.94 to 0.98 of Vulkan on the four models the GEMM covers fully, 0.83 on the MoE
whose experts are still MMQ's, and 0.71 on the q8_0 8B. Before the tile change those six read 0.68 to
0.77.

## Spread

Max minus min over the eighteen samples, as a percentage of the median:

| model | ROCm pp | ROCm tg | Vulkan pp | Vulkan tg |
|---|---|---|---|---|
| qwen2.5-1.5B | 2.5 | 2.5 | 0.1 | 0.2 |
| qwen3-8B | 5.6 | 6.7 | 0.1 | 0.5 |
| deepseek-r1-14B | 0.9 | 3.0 | 0.1 | 0.8 |
| qwen3-14B | 1.7 | 4.3 | 0.1 | 0.9 |
| qwen3.6-35B MoE | 2.8 | 16.3 | 1.8 | 2.0 |
| qwen3.8-27B | 11.5 | 2.8 | 0.1 | 0.2 |

Two of those are a single low sample, not a wide distribution. The 27B's prefill reads 100.5 to
102.8 in seventeen of eighteen samples and 90.99 in one; the 8B's is the bimodal row documented in
[`logs/fedora44-campaign-iq-shmem-2026-09-20/`](../fedora44-campaign-iq-shmem-2026-09-20/), at about 279
with occasional excursions below. In both cases it is the first sample of a round that is low. The MoE's
decode spread of 16.3 percent is the widest ROCm figure in the campaign and has been between 7 and 13
in the last three campaigns on that row; its median has not moved. Every Vulkan figure reproduces the
thirteen earlier campaigns within 0.5 percent.

## Files

`log` and `second/log` are the two campaign logs, one line per model, round and backend with the three
samples of each test; `hip_*` and `vk_*` are the `llama-bench` JSONL outputs behind each line, the second
run's under `second/`.
