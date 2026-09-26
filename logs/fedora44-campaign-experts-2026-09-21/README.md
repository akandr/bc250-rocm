# The campaign on the twelve-patch build, 2026-09-21

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-pkf16/bin`, run twice, 08:32 to 08:58 and 09:00 to 09:25 on the default
boot: Fedora 44, kernel 7.2.5 with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the
corrected comgr from `/opt/bc250-rocm`, GPU clock policy at 1500 MHz with `oberon-governor` active,
`ollama` stopped, the scraper timer and cron stopped for the window. The build is llama.cpp 7ba604f with
the twelve patches as committed; the twelfth is
[`patches/llamacpp/0012-rdna1-pkf16-expert-path.patch`](../../patches/llamacpp/0012-rdna1-pkf16-expert-path.patch),
which takes a mixture-of-experts model's experts off MMQ
([`logs/rdna1-pkf16-experts-2026-09-21/`](../rdna1-pkf16-experts-2026-09-21/)).

**Superseded by [`logs/campaign-graphopt-2026-09-24/`](../campaign-graphopt-2026-09-24/)**, which is
the campaign the front page shows. That one runs the same script with a third arm, ROCm with
`GGML_CUDA_GRAPH_OPT=1`, and its as-shipped arm reproduces everything below; the figures here remain
the reference for the build without that option. The two runs are pooled, so each figure is the median of
eighteen samples; `log` is the first run and `second/log` the second.

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1799.2 | 1850.3 | 197.6 | 212.4 |
| qwen3-8B Q8_0 | **405.3** | 394.7 | 38.6 | 39.1 |
| deepseek-r1-14B Q4_K_M | 193.5 | 199.8 | 32.5 | 35.0 |
| qwen3-14B Q4_K_M | 195.0 | 204.7 | 32.5 | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | **589.9** | 457.1 | 70.7 | 86.7 |
| qwen3.8-27B UD-IQ3_XXS | 102.4 | 105.0 | 15.1 | 17.6 |

Two models prefill faster than Vulkan now, the 8B by 1.03 times and the MoE by 1.29. The other four are
at 0.95 to 0.98. Decode is unchanged, at 0.81 to 0.99.

## Against the eleven-patch build

Ratios of medians over [`logs/fedora44-campaign-q8-2026-09-20/`](../fedora44-campaign-q8-2026-09-20/):

| model | pp512 | tg64 | Vulkan pp512 / tg64 |
|---|---|---|---|
| qwen2.5-1.5B | 1.002 | 1.000 | 1.000 / 1.000 |
| qwen3-8B | 1.012 | 1.002 | 1.000 / 1.001 |
| deepseek-r1-14B | 1.015 | 1.003 | 1.000 / 1.001 |
| qwen3-14B | 1.013 | 1.003 | 1.000 / 1.002 |
| qwen3.6-35B MoE | **1.565** | 1.004 | 1.001 / 1.001 |
| qwen3.8-27B | 1.000 | 1.004 | 1.000 / 1.000 |

The MoE moves and nothing else does by more than 1.5 percent, which is what a patch that only touches
MUL_MAT_ID should do; the MoE is the only one of the six with experts. Decode does not move at all.

## Against the three-patch build

The build the recipe started from, six days earlier
([`logs/fedora44-benchmarks-2026-09-15/clock-corrected/`](../fedora44-benchmarks-2026-09-15/clock-corrected/)):

| model | pp512 | tg64 |
|---|---|---|
| qwen2.5-1.5B | 2.27 | 1.68 |
| qwen3-8B | 1.66 | 0.99 |
| deepseek-r1-14B | 2.05 | 1.52 |
| qwen3-14B | 2.01 | 1.49 |
| qwen3.6-35B MoE | 2.04 | 2.06 |
| qwen3.8-27B | 1.47 | 1.92 |

## Spread

Max minus min over the eighteen samples, as a percentage of the median:

| model | ROCm pp | ROCm tg | Vulkan pp | Vulkan tg |
|---|---|---|---|---|
| qwen2.5-1.5B | 2.3 | 2.3 | 0.3 | 0.2 |
| qwen3-8B | 8.2 | 3.5 | 0.1 | 0.4 |
| deepseek-r1-14B | 1.6 | 2.3 | 0.1 | 1.0 |
| qwen3-14B | 1.5 | 2.9 | 0.1 | 0.8 |
| qwen3.6-35B MoE | 8.2 | 6.9 | 1.8 | 2.7 |
| qwen3.8-27B | 1.3 | 0.8 | 0.1 | 0.2 |

The two wide ROCm prefill rows are the first-sample artefact described in
[`logs/fedora44-campaign-q8-2026-09-20/`](../fedora44-campaign-q8-2026-09-20/): the first of the three
samples `llama-bench` takes after loading a model often reads low and the other two do not. Vulkan does
not show it, and every Vulkan figure reproduces the fifteen earlier campaigns within 0.5 percent.

## Files

`log` and `second/log` are the two campaign logs, one line per model, round and backend with the three
samples of each test; `hip_*` and `vk_*` are the `llama-bench` JSONL outputs behind each line, the second
run's under `second/`. `medians.txt` is
[`scripts/campaign_medians.py`](../../scripts/campaign_medians.py) over the two logs concatenated, which
is where the table above and the front page's figures come from; it carries them to two decimals, which
is the precision [`scripts/make_figures.py`](../../scripts/make_figures.py) plots.
