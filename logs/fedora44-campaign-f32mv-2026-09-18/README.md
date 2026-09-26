# The split throughput campaign with the float-activation matvec, 2026-09-18

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-f32mv/bin`, 14:10 to 14:52, same boot, clock, script, rounds and models as
the four campaigns before it; `hw-watcher.timer` and `crond` stopped. The build adds the float-activation
q4_K matvec and the long-K row to patch 5 ([`logs/rdna1-mmvq-2026-09-18/`](../rdna1-mmvq-2026-09-18/)).
Medians of nine samples:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 925.1 | 1850.3 | **177.4** | 212.4 |
| qwen3-8B Q8_0 | 277.9 | 394.9 | 37.5 | 39.1 |
| deepseek-r1-14B Q4_K_M | 98.7 | 199.9 | **29.2** | 35.0 |
| qwen3-14B Q4_K_M | 100.5 | 204.8 | **29.8** | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | 303.7 | 457.5 | 56.7 | 87.0 |
| qwen3.8-27B UD-IQ3_XXS | 71.2 | 105.1 | 11.0 | 17.6 |

Decode against the previous campaign
([`logs/fedora44-campaign-patch5-2026-09-18/`](../fedora44-campaign-patch5-2026-09-18/)): 1.5B **1.150**,
8B 0.978, deepseek-14B **1.143**, qwen3-14B **1.153**, MoE 1.040, 27B **0.961**; prefill 0.994 to 1.000;
Vulkan 1.000 to 1.002. Against three patches, decode: 1.5B 1.51, 8B 0.96, deepseek-14B 1.36, qwen3-14B
1.37, MoE 1.65, 27B 1.40. The three q4_K models are where the float kernel runs, and they move by what
its A/B predicted.

**The 27B loses 4 percent, and it is the long-K row.** IQ3_XXS never reaches the float kernel; the one
change it sees between the two campaigns is the long-K row, which gave four warps to every non-simple
type with K >= 8192, IQ types included. Their `vec_dot` is long enough that one wave per row was the
better geometry, as experiment A had shown for the MoE. The row is restricted to the K-quants in the next
build, and the campaign after it,
[`logs/fedora44-campaign-five-patches-final-2026-09-18/`](../fedora44-campaign-five-patches-final-2026-09-18/),
superseded this one.
