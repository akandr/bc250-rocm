# The split throughput campaign on the four-patch build, 2026-09-18

**Superseded the same night** by
[`logs/fedora44-campaign-final-2026-09-18/`](../fedora44-campaign-final-2026-09-18/), the same campaign
on the patch's final form. This one measured the intermediate form, before
the dispatch check for models without a power-of-two head ratio; it is kept because it is what found
that those models had not moved.

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-fatile/bin`, 03:01 to 03:45 on the default boot: Fedora 44, kernel 7.2.5
with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the corrected comgr from `/opt/bc250-rocm`,
GPU clock policy at 1500 MHz with `oberon-governor` active, `ollama` stopped. Same script, rounds and
models as [`logs/fedora44-benchmarks-2026-09-15/clock-corrected/`](../fedora44-benchmarks-2026-09-15/clock-corrected/),
which measured the three-patch build; the only change is llama.cpp patch 4, the RDNA1 flash-attention
tile rows ([`logs/rdna1-fattn-spill-2026-09-17/`](../rdna1-fattn-spill-2026-09-17/)). Medians of nine
samples, from [`scripts/campaign_medians.py`](../../scripts/campaign_medians.py) on `log`:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 919.3 | 1849.7 | 119.6 | 212.4 |
| qwen3-8B Q8_0 | 277.6 | 394.7 | 39.0 | 39.1 |
| deepseek-r1-14B Q4_K_M | 94.7 | 199.8 | 21.6 | 35.1 |
| qwen3-14B Q4_K_M | 97.0 | 204.8 | 21.9 | 34.8 |
| qwen3.6-35B-A3B MoE IQ2_M | 303.3 | 456.9 | 34.8 | 87.0 |
| qwen3.8-27B UD-IQ3_XXS | 71.2 | 105.0 | 7.9 | 17.6 |

## Against the three-patch campaign

Ratio of medians, this campaign over the 15 September one:

| model | ROCm pp512 | ROCm tg64 | Vulkan pp512 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B | **1.159** | 1.018 | 1.000 | 1.000 |
| qwen3-8B | **1.140** | 1.003 | 1.000 | 1.000 |
| deepseek-r1-14B | 1.003 | 1.008 | 1.000 | 1.000 |
| qwen3-14B | 1.003 | 1.006 | 1.001 | 1.000 |
| qwen3.6-35B MoE | **1.047** | 1.014 | 0.999 | 0.996 |
| qwen3.8-27B | **1.025** | 1.004 | 1.000 | 0.999 |

**The Vulkan build did not change between the two campaigns, and its twelve figures reproduce to within
0.4 percent, ten of them to within 0.1.** That is the control: the two sessions are comparable, and the
ROCm prefill differences are the patch. Decode is unchanged everywhere, within 2 percent.

The prefill gains at pp512 are smaller than at pp2048 (16 and 14 percent on the D=128 models against 37
and 38) because attention is a smaller share of a 512-token prompt. And two models gained nothing:
deepseek-r1-14B and qwen3-14B, both at 0.3 percent, inside the spread. Those are the two whose
head-count ratio is not a power of two: 40 query heads over 8 KV heads, a ratio of 5. llama.cpp's
flash-attention dispatch chooses `ncols2`, the number of query heads a tile shares a KV head across,
from that ratio, and 5 has no power-of-two divisor above 1, so these models run the `ncols2 = 1` kernel
variants. The D=256 register sweep showed the `ncols2 = 1` instance still spilling after the new rows
(197 registers, in `kernel-resource-usage.log` of the spill directory), and the D=128 check run after this
campaign says the same: with the committed rows the `<128,128,64,1>` instance still spills 137 registers
and 544 bytes per lane while every GQA-sharing variant of the same row is clean. The 14B models run that
instance. It is the one row the patch tuned for the wrong variant.

## Spread

Max minus min over the nine samples, as a percentage of the median: ROCm prefill 0.4 to 3.6 (the MoE
widest), ROCm decode 0.3 to 8.4 (the MoE and the 8B widest, as in every campaign here), Vulkan prefill
0.1 to 1.7, Vulkan decode 0.1 to 1.9.

## Files

`log` is the campaign log, one line per model, round and backend with the three samples of each test;
`hip_*` and `vk_*` are the `llama-bench` JSONL outputs behind each line.
