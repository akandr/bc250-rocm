# The campaign with ggml-cuda's multi-stream optimisation turned on, 2026-09-24

[`logs/graph-opt-2026-09-24/`](../graph-opt-2026-09-24/) found that `GGML_CUDA_GRAPH_OPT=1`, which is
off by default, is worth about ten percent of decode on the 1.5B in an A/B. An A/B is not a campaign,
so this is the campaign. **This is the campaign the front page shows.**

[`scripts/campaign_graphopt.sh`](../../scripts/campaign_graphopt.sh) is
[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with a third arm added and
nothing else changed: one `llama-bench` test per invocation, pp512 and tg64 alternated by arm, three
rounds, `-mmp 0 -ngl 99 -fa on`, `HSA_ENABLE_SDMA=0`. Run twice, 13:12 to 13:52 and 13:53 to 14:32,
`HIPBIN=~/llama-master/build-hip-pkf16/bin` against `~/llama-master/build-vk-f44/bin`, `ollama` stopped.
The two runs are pooled, so each figure is the median of eighteen samples; `log` is the first and
`second/log` the second.

**The method is unchanged from the previous campaign, and it shows.** The as-shipped arm reproduces
[`logs/fedora44-campaign-experts-2026-09-21/`](../fedora44-campaign-experts-2026-09-21/) across the
board: the 1.5B at 196.7 against 197.6, the MoE's prefill at 588.8 against 589.9, the 27B at 102.9
against 102.4, and every Vulkan figure within 0.3 percent. That is the control for the third arm.

## The table

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | ROCm tg64, `GRAPH_OPT` | Vulkan tg64 |
|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1798.6 | 1850.0 | 196.7 | **213.1** | 212.3 |
| qwen3-8B Q8_0 | **409.4** | 394.7 | 38.5 | **39.5** | 39.0 |
| deepseek-r1-14B Q4_K_M | 195.8 | 199.8 | 32.6 | **33.6** | 35.1 |
| qwen3-14B Q4_K_M | 197.6 | 204.8 | 32.7 | **33.7** | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | **588.8** | 457.0 | 71.2 | 70.8 | 86.9 |
| qwen3.8-27B UD-IQ3_XXS | 102.9 | 105.0 | 15.2 | 15.1 | 17.6 |

| model | decode vs Vulkan, as shipped | with `GRAPH_OPT` | the option is worth |
|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 0.93 | **1.00** | **1.083** |
| qwen3-8B Q8_0 | 0.99 | **1.01** | **1.026** |
| deepseek-r1-14B Q4_K_M | 0.93 | **0.96** | **1.031** |
| qwen3-14B Q4_K_M | 0.94 | **0.97** | **1.032** |
| qwen3.6-35B-A3B MoE IQ2_M | 0.82 | 0.81 | 0.994 |
| qwen3.8-27B UD-IQ3_XXS | 0.86 | 0.86 | 0.998 |

**Decode against Vulkan moves from 0.81 to 0.99 up to 0.81 to 1.01**, and the two smallest models cross
it: the 1.5B draws level at 213.1 against 212.3, and the 8B goes ahead at 39.5 against 39.0. With
prefill already ahead on those two, **the 8B is now faster than Vulkan on both halves**.

**Prefill does not move at all**, 1798.5 against 1798.6 on the 1.5B and within 0.1 percent on every
other model. That is the expected shape: prefill runs few long kernels that already fill sixteen
compute units, so there is nothing for a second stream to do.

## The two that do not move are the two with nothing to launch

The MoE reads 0.994 and the 27B 0.998, and **both are noise, not small losses**. Their sample
ranges overlap between the arms, 68.23 to 71.60 against 68.41 to 71.17 on the MoE and 15.04 to 15.19
against 15.01 to 15.15 on the 27B, where the 1.5B's ranges do not touch at all, 192.81 to 198.10 against
206.64 to 213.58.

These are exactly the two models where `llama-bench -v` shows the optimiser launching **no streams**:
the MoE identifies 70 fork regions and launches none, because one region whose branches write to
overlapping memory disables every region in the graph
([`logs/graph-opt-2026-09-24/`](../graph-opt-2026-09-24/)). So the option gains where it runs and costs
nothing where it does not, which is the result that makes it safe to recommend unconditionally.

## What this does not change

The correctness gates are unaffected and were checked instead of assumed: 8.9274 and 9.1125 with the
option on, identical to four decimals
([`logs/graph-opt-2026-09-24/`](../graph-opt-2026-09-24/)).

Neither campaign throttled or faulted: the governor logged nothing, the kernel logged no page fault,
preemption failure or reset, and the edge sensor peaked at 75 C against the 93 C at which the governor
drops the clock.

It is still one board, and the option is inactive wherever CUDA graphs are, including under the
`GGML_CUDA_DISABLE_GRAPHS=1` workaround the [known issues](../../README.md#known-issues) table
recommends for the 14B models at very deep context. The depth tables elsewhere in the repository were
measured without the option and have not been redone.

## Files

`log` and `second/log` are the two runs. `medians.txt` is the pooled output of
[`scripts/campaign_medians_graphopt.py`](../../scripts/campaign_medians_graphopt.py), which reads the
three-arm logs that `campaign_medians.py` cannot.
