# The campaign on the nine-patch build, 2026-09-20

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-pkf16/bin`, run twice, 16:16 to 16:42 and 16:46 to 17:13 on the default
boot: Fedora 44, kernel 7.2.5 with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the
corrected comgr from `/opt/bc250-rocm`, GPU clock policy at 1500 MHz with `oberon-governor` active,
`ollama` stopped, the scraper timer and cron stopped for the window. The build is llama.cpp 7ba604f with
the nine patches as committed; the ninth is
[`patches/llamacpp/0009-rdna1-iq-matvec-shmem-codebook.patch`](../../patches/llamacpp/0009-rdna1-iq-matvec-shmem-codebook.patch),
which stages the IQ codebooks in shared memory
([`logs/rdna1-iq-matvec-2026-09-20/`](../rdna1-iq-matvec-2026-09-20/)).

It was the campaign the front page showed until the GEMM's column tile was halved
([`logs/fedora44-campaign-tile-2026-09-20/`](../fedora44-campaign-tile-2026-09-20/)). The two runs are
pooled, so each figure is the median of
eighteen samples, not the usual nine; `log` is the first run and `second/log` the second, and
[`scripts/campaign_medians.py`](../../scripts/campaign_medians.py) on the two concatenated gives:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1249.6 | 1849.8 | 197.7 | 212.4 |
| qwen3-8B Q8_0 | 277.4 | 394.8 | 38.6 | 39.1 |
| deepseek-r1-14B Q4_K_M | 145.9 | 199.8 | 32.5 | 35.0 |
| qwen3-14B Q4_K_M | 151.5 | 204.7 | 32.6 | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | 340.4 | 456.9 | 70.6 | 86.7 |
| qwen3.8-27B UD-IQ3_XXS | 81.3 | 105.0 | 15.1 | 17.6 |

## Against the eight-patch build

Ratios of medians over
[`logs/fedora44-campaign-final-defaults-2026-09-20/`](../fedora44-campaign-final-defaults-2026-09-20/),
the same script and models four hours earlier:

| model | pp512 | tg64 | Vulkan pp512 / tg64 |
|---|---|---|---|
| qwen2.5-1.5B | 0.999 | 0.998 | 1.000 / 1.000 |
| qwen3-8B | 0.993 | 1.003 | 1.000 / 1.002 |
| deepseek-r1-14B | 1.006 | 1.007 | 1.000 / 1.003 |
| qwen3-14B | 1.003 | 1.007 | 1.001 / 1.002 |
| qwen3.6-35B MoE | 1.000 | 0.998 | 0.998 / 1.001 |
| qwen3.8-27B | 1.001 | **1.019** | 1.000 / 1.001 |

One row moves and it is the one the patch is for: the 27B's decode, 14.80 to 15.09 t/s. Everything else
is inside 0.7 percent, which is what the patch predicts, since its kernel only touches the three IQ
types and the q4_K, q5_K, q6_K and q8_0 kernels come out of the compiler instruction for instruction
identical to the eight-patch build. The MoE does not move because the staging is confined to ordinary matrix-vector
products and its experts arrive through MUL_MAT_ID.

## Spread, and why this campaign was run twice

Max minus min over the eighteen samples, as a percentage of the median:

| model | ROCm pp | ROCm tg | Vulkan pp | Vulkan tg |
|---|---|---|---|---|
| qwen2.5-1.5B | 1.6 | 2.4 | 0.3 | 0.2 |
| qwen3-8B | 11.2 | 4.7 | 0.1 | 0.5 |
| deepseek-r1-14B | 0.6 | 4.8 | 0.1 | 0.6 |
| qwen3-14B | 5.9 | 3.3 | 0.1 | 0.9 |
| qwen3.6-35B MoE | 1.2 | 7.1 | 1.9 | 2.6 |
| qwen3.8-27B | 1.2 | 1.4 | 0.1 | 0.2 |

The 8B's ROCm prefill is bimodal and that is what the first run caught: it read 267.2, five percent below
every earlier campaign, on a row whose kernel this patch cannot touch. Fifteen further samples of that
one measurement on an idle board (`recheck-8b-log`, `recheck8b-*.jsonl`) read

```
r1 hip 277.5 279.6 279.7      r2 hip 268.0 267.6 270.7      r3 hip 277.3 279.5 279.5
r4 hip 245.0 279.5 279.5      r5 hip 237.6 276.2 279.4
```

against Vulkan at 394.6 to 394.8 in every one of the same fifteen. So the row sits at about 279 with an
occasional excursion 3 to 15 percent below it, the board was at 59 C throughout, and the low mode is not
the build. Pooling two campaigns puts the median back at 277.4. The cause of the excursion was not
chased; the earlier campaigns show the same row at 4.2 and 4.6 percent spread, so it is not new.

The MoE's decode spread of 7.1 percent is smaller than the 12.7 percent the previous campaign recorded on
that row. Every Vulkan figure reproduces the twelve earlier campaigns within 0.5 percent.

## Files

`log` and `second/log` are the two campaign logs, one line per model, round and backend with the three
samples of each test; `hip_*` and `vk_*` are the `llama-bench` JSONL outputs behind each line, the second
run's under `second/`. `recheck-8b-log` and `recheck8b-*.jsonl` are the fifteen-sample re-measurement of
the 8B's prefill.
