# The split throughput campaign with the float matvec on the IQ types and the experts, 2026-09-18

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-f32iq/bin`, 21:08 to 21:35 on the default boot: Fedora 44, kernel 7.2.5
with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the corrected comgr from `/opt/bc250-rocm`,
GPU clock policy at 1500 MHz with `oberon-governor` active, `ollama` stopped, the scraper timer and cron
stopped for the window. Same script, rounds and models as every campaign before it. The build is
llama.cpp 7ba604f with the five patches as committed: patch 5's float-activation matvec now covers q4_K,
q5_K, q6_K, q8_0, IQ2_XXS, IQ3_XXS and IQ3_S, and the expert-id path with one token
([`logs/rdna1-mmvq-2026-09-18/`](../rdna1-mmvq-2026-09-18/), experiment G). It was the campaign the front
page showed until the seven-patch build replaced it
([`logs/fedora44-campaign-seven-patches-2026-09-19/`](../fedora44-campaign-seven-patches-2026-09-19/)).
Medians of nine samples, from
[`scripts/campaign_medians.py`](../../scripts/campaign_medians.py) on `log`:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 925.7 | 1850.4 | 193.2 | 212.4 |
| qwen3-8B Q8_0 | 279.5 | 394.8 | 37.7 | 39.1 |
| deepseek-r1-14B Q4_K_M | 98.7 | 199.9 | 30.3 | 35.0 |
| qwen3-14B Q4_K_M | 100.5 | 204.7 | 30.5 | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | 303.8 | 457.3 | 70.1 | 86.9 |
| qwen3.8-27B UD-IQ3_XXS | 71.2 | 105.0 | 14.9 | 17.6 |

## Against the previous campaigns

Ratios of medians, this build over the q4/q6/q8 float build four hours earlier
([`logs/fedora44-campaign-float-all-2026-09-18/`](../fedora44-campaign-float-all-2026-09-18/)) and over the
three-patch build three days earlier
([`logs/fedora44-benchmarks-2026-09-15/clock-corrected/`](../fedora44-benchmarks-2026-09-15/clock-corrected/)):

| model | over q4/q6/q8 float, pp512 | over q4/q6/q8 float, tg64 | over three patches, pp512 | over three patches, tg64 | Vulkan pp512 / tg64, over three patches |
|---|---|---|---|---|---|
| qwen2.5-1.5B | 1.000 | 0.993 | 1.167 | 1.645 | 1.001 / 1.001 |
| qwen3-8B | 1.006 | 1.003 | 1.148 | 0.968 | 1.000 / 1.000 |
| deepseek-r1-14B | 1.000 | 0.977 | 1.045 | 1.418 | 1.000 / 0.998 |
| qwen3-14B | 1.000 | 0.975 | 1.038 | 1.400 | 1.000 / 0.999 |
| qwen3.6-35B MoE | 0.999 | **1.197** | 1.049 | **2.046** | 1.000 / 0.995 |
| qwen3.8-27B | 0.999 | **1.302** | 1.025 | **1.894** | 1.001 / 0.998 |

The two models made of the new types move and nothing else does: the MoE, whose experts are IQ2_XXS and
IQ3_XXS and whose attention is Q5_K, gains 20 percent of decode; the 27B, IQ3_S and IQ3_XXS with a Q5_K
output head, gains 30. Prefill is unchanged to a tenth of a percent everywhere, as it should be for a
single-column kernel. The four models with no new type read between 0.3 percent up and 2.5 percent down;
the 14B models' 2.5 percent is inside the 3.6 to 6.1 percent decode spread they show from campaign to
campaign, on a board that had been under continuous GPU load for four hours when this one started, and
the interleaved A/B in the experiment directory has the 1.5B at 180.0 / 180.7 against 180.7 / 180.9
between the two builds. Against three patches the decode of the MoE has doubled and the 27B's is 1.9
times; the 8B's reads 3 percent below that campaign, which later work attributes to a difference between
the sessions, not the builds ([`logs/round3-2026-09-19/`](../round3-2026-09-19/)).

## Spread

Max minus min over the nine samples, as a percentage of the median: ROCm prefill 0.3 to 1.4 with the 8B
at 4.2, ROCm decode 0.6 to 4.3, Vulkan prefill 0.0 to 1.7, Vulkan decode 0.2 to 2.3. Every Vulkan row
reproduces the seven earlier campaigns within 0.5 percent.

## Files

`log` is the campaign log, one line per model, round and backend with the three samples of each test;
`hip_*` and `vk_*` are the `llama-bench` JSONL outputs behind each line.
