# The same soak with SDMA enabled, 2026-09-22

Every throughput figure in this repository was taken with `HSA_ENABLE_SDMA=0`, which was forced by a
defect, not chosen. The defect is fixed, and the headline checks have been repeated with SDMA on
without moving: throughput measured ABBA, the correctness gates, Vulkan, the allocation-churn sweep and
decode at depth. The endurance runs had not been, and an hours-long run is where a copy engine that was
quietly wrong would show up.

[`scripts/soak_thirteen_sdma.sh`](../../scripts/soak_thirteen_sdma.sh), three hours, 02:36 to 05:46,
same boot and build as the eight-hour run that preceded it
([`logs/soak-thirteen-2026-09-22/`](../soak-thirteen-2026-09-22/)), the only difference being that
`HSA_ENABLE_SDMA` is left unset.

| | value |
|---|---|
| rounds | 58 |
| gate mismatches | **0** |
| distinct gate values per model | **one each**: 10.2088, 9.4017, 6.2265, 6.2737 |
| kernel fault lines, this boot and the one before | **0** |

Throughput, medians over the rounds of each model:

| model | pp512 | spread | tg64 | spread |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1772.32 | 0.4 % | **183.61** | 1.4 % |
| qwen3-8B Q8_0 | 411.75 | 1.5 % | 38.73 | 0.4 % |
| qwen3.6-35B-A3B MoE IQ2_M | 594.76 | 0.7 % | 71.72 | 1.0 % |
| qwen3.8-27B UD-IQ3_XXS | 103.38 | 2.1 % | 15.18 | 0.5 % |

**The gates are unchanged, so SDMA is not quietly wrong**, which is what this run was for: four
perplexity values over 58 evaluations, each the same figure the SDMA-off soak returned.

Throughput is a different matter and this run is where it surfaced. Three of the four models read the
same as the eight-hour SDMA-off soak to three figures. The 1.5B reads 183.61 against 196.99. That is
two blocked runs instead of an interleaved comparison, so it was measured properly afterwards and it
holds: [`logs/sdma-decode-cost-2026-09-22/`](../sdma-decode-cost-2026-09-22/).

## Files

`log` is the round-by-round record.
