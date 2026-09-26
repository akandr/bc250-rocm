# The split throughput campaign with the float matvec on q4_K, q6_K and q8_0, 2026-09-18

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-f32mv/bin`, 17:03 to 17:29 on the default boot: Fedora 44, kernel 7.2.5
with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the corrected comgr from `/opt/bc250-rocm`,
GPU clock policy at 1500 MHz with `oberon-governor` active, `ollama` stopped, the scraper timer and cron
stopped for the window. Same script, rounds and models as every campaign before it. The build is
llama.cpp 7ba604f with the five patches as committed, which is to say patch 5 with the float-activation
matvec extended from q4_K to q6_K and q8_0 ([`logs/rdna1-mmvq-2026-09-18/`](../rdna1-mmvq-2026-09-18/),
experiment F). It was the campaign the front page showed until the IQ types joined the float matvec
([`logs/fedora44-campaign-iq-float-2026-09-18/`](../fedora44-campaign-iq-float-2026-09-18/)).
Medians of nine samples, from
[`scripts/campaign_medians.py`](../../scripts/campaign_medians.py) on `log`:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 925.4 | 1849.8 | 194.6 | 212.5 |
| qwen3-8B Q8_0 | 277.7 | 394.7 | 37.6 | 39.1 |
| deepseek-r1-14B Q4_K_M | 98.7 | 199.9 | 31.0 | 35.0 |
| qwen3-14B Q4_K_M | 100.5 | 204.8 | 31.3 | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | 304.0 | 457.6 | 58.6 | 87.0 |
| qwen3.8-27B UD-IQ3_XXS | 71.2 | 105.0 | 11.4 | 17.6 |

## Against the previous campaigns

Ratios of medians, this build over the q4_K-only float build
([`logs/fedora44-campaign-five-patches-final-2026-09-18/`](../fedora44-campaign-five-patches-final-2026-09-18/))
and over the three-patch build three days earlier
([`logs/fedora44-benchmarks-2026-09-15/clock-corrected/`](../fedora44-benchmarks-2026-09-15/clock-corrected/)):

| model | over q4_K-only float, pp512 | over q4_K-only float, tg64 | over three patches, pp512 | over three patches, tg64 | Vulkan pp512 / tg64, over three patches |
|---|---|---|---|---|---|
| qwen2.5-1.5B | 1.000 | **1.101** | 1.167 | **1.657** | 1.000 / 1.001 |
| qwen3-8B | 0.993 | 0.999 | 1.140 | 0.966 | 1.000 / 1.000 |
| deepseek-r1-14B | 1.000 | **1.062** | 1.045 | **1.451** | 1.001 / 0.999 |
| qwen3-14B | 1.000 | **1.050** | 1.038 | **1.435** | 1.000 / 1.000 |
| qwen3.6-35B MoE | 1.001 | **1.028** | 1.050 | **1.709** | 1.001 / 0.996 |
| qwen3.8-27B | 1.001 | 1.000 | 1.026 | 1.455 | 1.000 / 0.999 |

Prefill does not move, as it should not: the kernel is single-column only. Decode moves where the models
have q6_K tensors: the 1.5B's output head and its `ffn_down` are q6_K and it gains 10 percent on top of the
q4_K kernel's; the two 14Bs, whose Q4_K_M mixes carry q6_K on part of the attention and FFN weights, gain 5
and 6; the MoE's IQ2_M mix keeps a few q6_K tensors and gains 3. The Q8_0 8B is flat to a tenth of a
percent: at 8.2 GiB of weights and 37.6 tokens per second it reads memory at 310 GB/s, and its matvec was
already bandwidth-bound in the int8 form, so the float q8_0 kernel changes the instruction stream and
not the time. Against three patches the 8B's decode reads 3.4 percent lower; a same-session A/B later
attributes that to the sessions, not the builds
([`logs/round3-2026-09-19/`](../round3-2026-09-19/)). The 27B has no covered type.

## Spread

Max minus min over the nine samples, as a percentage of the median: ROCm prefill 0.4 to 1.4 except the
8B at 8.3 (its third round ran 3 to 4 percent below the first two throughout, a warm package; the
decode rows of that round are 3 percent low too), ROCm decode 0.8 to 6.1
(deepseek-r1-14B widest this time), Vulkan prefill 0.1 to 1.7, Vulkan decode 0.2 to 2.6. Every Vulkan row
reproduces the six earlier campaigns within 0.5 percent.

## Files

`log` is the campaign log, one line per model, round and backend with the three samples of each test;
`hip_*` and `vk_*` are the `llama-bench` JSONL outputs behind each line.
