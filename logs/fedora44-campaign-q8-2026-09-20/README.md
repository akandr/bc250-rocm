# The campaign on the eleven-patch build, 2026-09-20

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-pkf16/bin`, run twice, 21:32 to 22:12 and 22:13 to 22:39 on the default
boot: Fedora 44, kernel 7.2.5 with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the
corrected comgr from `/opt/bc250-rocm`, GPU clock policy at 1500 MHz with `oberon-governor` active,
`ollama` stopped, the scraper timer and cron stopped for the window. The build is llama.cpp 7ba604f with
the eleven patches as committed; the eleventh is
[`patches/llamacpp/0011-rdna1-pkf16-q8_0.patch`](../../patches/llamacpp/0011-rdna1-pkf16-q8_0.patch),
which brings q8_0 onto the prefill GEMM now that the tile has changed
([`logs/rdna1-pkf16-tile-2026-09-20/`](../rdna1-pkf16-tile-2026-09-20/)).

It was the campaign the front page showed until the GEMM took the MoE's experts
([`logs/fedora44-campaign-experts-2026-09-21/`](../fedora44-campaign-experts-2026-09-21/)). The two runs
are pooled, so each figure is the median of
eighteen samples; `log` is the first run and `second/log` the second.

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1796.0 | 1850.2 | 197.7 | 212.4 |
| qwen3-8B Q8_0 | **400.5** | 394.8 | 38.5 | 39.0 |
| deepseek-r1-14B Q4_K_M | 190.6 | 199.8 | 32.4 | 35.0 |
| qwen3-14B Q4_K_M | 192.6 | 204.6 | 32.4 | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | 377.0 | 456.8 | 70.4 | 86.6 |
| qwen3.8-27B UD-IQ3_XXS | 102.4 | 105.0 | 15.0 | 17.6 |

The 8B is the first row on this page where ROCm prefills faster than Vulkan, 400.5 against 394.8. The
other five read 0.83 to 0.98 of Vulkan on prefill and 0.81 to 0.99 on decode.

## Against the ten-patch build

Ratios of medians over [`logs/fedora44-campaign-tile-2026-09-20/`](../fedora44-campaign-tile-2026-09-20/),
the same script and models an hour earlier:

| model | pp512 | tg64 | Vulkan pp512 / tg64 |
|---|---|---|---|
| qwen2.5-1.5B | 1.000 | 1.006 | 1.000 / 1.000 |
| qwen3-8B | **1.433** | 1.018 | 1.000 / 0.999 |
| deepseek-r1-14B | 0.994 | 0.998 | 1.000 / 0.999 |
| qwen3-14B | 0.998 | 0.997 | 1.000 / 0.999 |
| qwen3.6-35B MoE | 1.000 | 0.999 | 0.999 / 0.999 |
| qwen3.8-27B | 0.999 | 0.997 | 1.000 / 1.000 |

One row moves and it is the one the patch is for. The 8B's 1.018 on decode is the same row recovering
from a low sample in the previous campaign, not an effect of a prefill-only change.

## Against the three-patch build

The build the recipe started from, five days earlier
([`logs/fedora44-benchmarks-2026-09-15/clock-corrected/`](../fedora44-benchmarks-2026-09-15/clock-corrected/)):

| model | pp512 | tg64 |
|---|---|---|
| qwen2.5-1.5B | 2.27 | 1.68 |
| qwen3-8B | 1.64 | 0.99 |
| deepseek-r1-14B | 2.02 | 1.52 |
| qwen3-14B | 1.99 | 1.49 |
| qwen3.6-35B MoE | 1.30 | 2.05 |
| qwen3.8-27B | 1.47 | 1.91 |

## Spread, and the first sample of a round

Max minus min over the eighteen samples, as a percentage of the median:

| model | ROCm pp | ROCm tg | Vulkan pp | Vulkan tg |
|---|---|---|---|---|
| qwen2.5-1.5B | 19.4 | 2.4 | 0.2 | 0.4 |
| qwen3-8B | 6.6 | 10.1 | 0.1 | 0.5 |
| deepseek-r1-14B | 1.6 | 4.5 | 0.1 | 0.8 |
| qwen3-14B | 1.6 | 4.7 | 0.1 | 0.9 |
| qwen3.6-35B MoE | 1.3 | 6.6 | 1.8 | 2.4 |
| qwen3.8-27B | 0.8 | 2.6 | 0.1 | 0.2 |

The wide ROCm figures are one artefact, and it is the same one every time: the first of the three samples
`llama-bench` takes after loading a model often reads low, and the second and third do not. The 1.5B's
prefill is the clearest case, at 1767, 1762, 1761, 1629, 1451 and 1660 for the six first samples against
1743 to 1799 for the other twelve; the 8B's decode reads 35.2, 34.7, 37.8, 37.7, 38.2, 37.8 for its
first samples against 37.4 to 38.6 for the rest. The median over eighteen is unaffected, which is why
the campaign is run twice and pooled. Vulkan does not show it: its widest spread here is 2.4 percent and
every Vulkan figure reproduces the fourteen earlier campaigns within 0.5 percent.

## Files

`log` and `second/log` are the two campaign logs, one line per model, round and backend with the three
samples of each test; `hip_*` and `vk_*` are the `llama-bench` JSONL outputs behind each line, the second
run's under `second/`.
