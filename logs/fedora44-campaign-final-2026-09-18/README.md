# The split throughput campaign on the final four-patch build, 2026-09-18

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-fatile/bin`, 05:12 to 05:55 on the default boot: Fedora 44, kernel 7.2.5
with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the corrected comgr from `/opt/bc250-rocm`,
GPU clock policy at 1500 MHz with `oberon-governor` active, `ollama` stopped. Same script, rounds and
models as the three-patch campaign
([`logs/fedora44-benchmarks-2026-09-15/clock-corrected/`](../fedora44-benchmarks-2026-09-15/clock-corrected/))
and as the intermediate four-patch one two hours earlier
([`logs/fedora44-campaign-fa4-2026-09-18/`](../fedora44-campaign-fa4-2026-09-18/)). The build is llama.cpp
7ba604f with the four patches as committed, which is to say the RDNA1 flash-attention patch in its final
form: rows for D=128, 256 and 512, and the dispatch check for models without a power-of-two head ratio
([`logs/rdna1-fattn-spill-2026-09-17/`](../rdna1-fattn-spill-2026-09-17/)). It was the campaign the front
page showed until the first form of patch 5 replaced it
([`logs/fedora44-campaign-five-patches-2026-09-18/`](../fedora44-campaign-five-patches-2026-09-18/)).
Medians of nine samples, from
[`scripts/campaign_medians.py`](../../scripts/campaign_medians.py) on `log`:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 924.3 | 1849.8 | 119.2 | 212.4 |
| qwen3-8B Q8_0 | 279.3 | 394.7 | 39.0 | 39.1 |
| deepseek-r1-14B Q4_K_M | 98.6 | 199.8 | 21.5 | 35.1 |
| qwen3-14B Q4_K_M | 100.4 | 204.8 | 21.9 | 34.8 |
| qwen3.6-35B-A3B MoE IQ2_M | 303.4 | 457.1 | 34.7 | 87.0 |
| qwen3.8-27B UD-IQ3_XXS | 71.1 | 105.0 | 7.9 | 17.6 |

## Against the other two campaigns

Ratios of medians:

| model | over three patches, pp512 | over three patches, tg64 | over the intermediate build, pp512 | Vulkan pp512 / tg64, over three patches |
|---|---|---|---|---|
| qwen2.5-1.5B | **1.166** | 1.015 | 1.005 | 1.000 / 1.001 |
| qwen3-8B | **1.147** | 1.002 | 1.006 | 1.000 / 1.000 |
| deepseek-r1-14B | **1.044** | 1.005 | **1.041** | 1.000 / 1.000 |
| qwen3-14B | **1.038** | 1.002 | **1.035** | 1.000 / 1.000 |
| qwen3.6-35B MoE | **1.047** | 1.013 | 1.000 | 1.000 / 0.996 |
| qwen3.8-27B | **1.025** | 1.004 | 0.999 | 1.001 / 0.999 |

The Vulkan build did not change across the three campaigns and its rows reproduce to within 0.4 percent
in all of them, so the three sessions are comparable and the ROCm differences are the patch. Decode is
unchanged throughout, within 1.5 percent.

The column against the intermediate build isolates the last two changes, the dispatch check and the
wider `nbatch_K` on row 64: the two 14B models, which the intermediate build had not moved at all,
gain 4 percent at pp512 (and 9 to 11 at pp2048, in the spill directory's `ab14b.log`), and the D=128
GQA models gain a further half percent at pp512 (2 to 3 percent at pp2048). The MoE and the 27B are
untouched by those changes, as they should be: their rows did not change.

## Spread

Max minus min over the nine samples, as a percentage of the median: ROCm prefill 0.4 to 3.0 (the 8B
widest this time), ROCm decode 0.5 to 6.2 (the 8B, as usual), Vulkan prefill 0.0 to 1.7, Vulkan decode
0.1 to 2.6.

## Background services

Checked after the fact, because a scraper timer turned out to fire every 50 minutes: `services-audit.txt`
lists every timer and cron job that ran inside a measurement window, with its cumulative CPU time. None
touches the GPU, `ollama` never started, and the busiest of them used 16 CPU-seconds over the board's 16
hours of uptime. The Vulkan rows reproducing to 0.4 percent across three sessions is the empirical side of
the same statement.

## Files

`log` is the campaign log, one line per model, round and backend with the three samples of each test;
`hip_*` and `vk_*` are the `llama-bench` JSONL outputs behind each line.
