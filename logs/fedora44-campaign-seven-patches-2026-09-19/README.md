# The split throughput campaign on the seven-patch build, 2026-09-19

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-final/bin`, 07:56 to 08:23 on the default boot: Fedora 44, kernel 7.2.5 with
the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1, the corrected comgr, and from this campaign on the
rebuilt ROCr and HIP runtimes from `/opt/bc250-rocm` ([`logs/rocr-queue-scratch-2026-09-18/`](../rocr-queue-scratch-2026-09-18/);
both were checked to change neither gates nor throughput before they went in), GPU clock policy at 1500
MHz with `oberon-governor` active, `ollama` stopped, the scraper timer and cron stopped for the window.
Same script, rounds and models as every campaign before it. The build is llama.cpp 7ba604f with the seven
patches as committed: the three small ones, the flash-attention patch with the rows of experiments J and K
and the one-token vector rule of experiment I, the matrix-vector patch, the transposed concat and the
gated delta net's lanes ([`logs/rdna1-fattn-remainder-2026-09-19/`](../rdna1-fattn-remainder-2026-09-19/),
[`logs/rdna1-gdn-concat-2026-09-19/`](../rdna1-gdn-concat-2026-09-19/)). **This is the campaign the front
page shows.** Medians of nine samples, from [`scripts/campaign_medians.py`](../../scripts/campaign_medians.py)
on `log`:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 925.5 | 1849.7 | 195.4 | 212.3 |
| qwen3-8B Q8_0 | 279.5 | 394.8 | 37.3 | 39.0 |
| deepseek-r1-14B Q4_K_M | 98.7 | 199.8 | 32.4 | 34.9 |
| qwen3-14B Q4_K_M | 100.5 | 204.7 | 32.4 | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | 315.0 | 457.8 | 70.3 | 86.6 |
| qwen3.8-27B UD-IQ3_XXS | 72.8 | 105.0 | 14.8 | 17.6 |

## Against the previous campaigns

Ratios of medians, this build over the five-patch build of the day before
([`logs/fedora44-campaign-iq-float-2026-09-18/`](../fedora44-campaign-iq-float-2026-09-18/)) and over the
three-patch build ([`logs/fedora44-benchmarks-2026-09-15/clock-corrected/`](../fedora44-benchmarks-2026-09-15/clock-corrected/)):

| model | over five patches, pp512 | tg64 | over three patches, pp512 | tg64 | Vulkan pp512 / tg64, over three patches |
|---|---|---|---|---|---|
| qwen2.5-1.5B | 1.000 | 1.011 | 1.167 | 1.663 | 1.000 / 1.000 |
| qwen3-8B | 1.000 | 0.989 | 1.148 | 0.958 | 1.000 / 0.999 |
| deepseek-r1-14B | 1.000 | **1.070** | 1.045 | 1.517 | 1.000 / 0.997 |
| qwen3-14B | 1.000 | **1.063** | 1.038 | 1.487 | 1.000 / 0.998 |
| qwen3.6-35B MoE | **1.037** | 1.003 | 1.088 | 2.051 | 1.001 / 0.992 |
| qwen3.8-27B | **1.023** | 0.997 | 1.049 | 1.888 | 1.001 / 0.998 |

The two patches for the hybrid models' prefill show where they should: the MoE's pp512 gains 3.7
percent and the 27B's 2.3, and nothing else's prefill moves. The 14B models' decode gains 6 to 7 percent
at an empty context from the one-token vector rule: their attention tile was the spilling `1 x 1`
instance even over the 256-key padded block a fresh context has, and the vector kernel is cheaper there
too. The 1.5B and the 8B, whose new tile rows matter at depth, read within a percent at depth zero; the
depth table below is where they move. Against the three-patch build the decode of the four models the
matvec patch covers is 1.49 to 2.05 times, the prefill 4 to 17 percent. The 8B's decode reads 4 percent
below the September 15 campaign; that is a difference between the sessions and not between the builds,
as a direct A/B on one boot shows ([`logs/round3-2026-09-19/`](../round3-2026-09-19/)).

## Decode at depth

The same build and the Vulkan build, `tg32` after a 4096-token prefix and after none, one invocation per
depth, three repetitions (`assembly-log`, the `depth` lines):

| model | ROCm at 4096 | Vulkan at 4096 | ratio | ROCm at 0 | Vulkan at 0 (campaign tg64) |
|---|---|---|---|---|---|
| qwen2.5-1.5B | 166.4 | 179.3 | 0.93 | 183.0 | 212.3 |
| qwen3-8B | 34.7 | 35.5 | 0.98 | 37.4 | 39.0 |
| deepseek-r1-14B | 25.7 | 30.5 | 0.84 | 31.3 | 34.9 |
| qwen3-14B | 26.3 | 31.1 | 0.85 | 31.2 | 34.7 |
| qwen3.6-35B MoE | 63.7 | 81.3 | 0.78 | 62.9 | 86.6 |
| qwen3.8-27B | 14.1 | 17.0 | 0.83 | 14.3 | 17.6 |

Before experiments I and K the 8B read 30.5 and deepseek-r1-14B 17.7 at this depth on the same
hardware, against Vulkan's 35.5 and 30.5. The Vulkan depth-zero runs in `assembly-log` came right after
its depth-4096 runs in the same sequence and read low with wide spreads (the 8B 29.1 with a spread of
3.2, the MoE 56.8 with 15.5), the board's heat, not the backend, so the campaign's tg64 stands in
for them in the last column.

## Spread

Max minus min over the nine samples, as a percentage of the median: ROCm prefill 0.4 to 1.5, ROCm decode
0.6 to 2.8 except the 8B at 6.4 and the MoE at 8.8, Vulkan prefill 0.1 to 1.7, Vulkan decode 0.1 to 2.9.
Every Vulkan row reproduces the eight earlier campaigns within 0.5 percent.

## Files

`log` is the campaign log; `hip_*` and `vk_*` the `llama-bench` JSONL outputs behind each line;
`assembly-log` the whole final-assembly run (patch regeneration, the HIP library's gate and install, the
final build's correctness tests and gates, the campaign, the depth table); `tbo-*.log` the correctness
runs.
