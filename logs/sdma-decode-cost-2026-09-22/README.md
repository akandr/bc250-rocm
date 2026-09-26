# Enabling SDMA costs the small model 5 percent of decode, 2026-09-22

This repository has said since August that fixing SDMA "buys no measurable speed for inference, 0.1
percent on 8B decode measured ABBA and nothing on Vulkan", and [step 2](../../README.md#2-firmware-and-boot-arguments)
offered the microcode substitution and `HSA_ENABLE_SDMA=0` as alternatives on the grounds that
"neither choice changes inference speed measurably". Both statements rest on one model.

They came up again because two soaks run back to back on the same boot disagreed on one of four
models. The eight-hour soak ran with `HSA_ENABLE_SDMA=0` and the three-hour one with SDMA enabled
([`logs/soak-thirteen-2026-09-22/`](../soak-thirteen-2026-09-22/),
[`logs/soak-sdma-2026-09-22/`](../soak-sdma-2026-09-22/)): the 8B, the MoE and the 27B read the same to
three figures, and the 1.5B read 196.99 against 183.61.

Two blocked runs are not a comparison on this board. This is the interleaved version
([`scripts/sdma_ab.sh`](../../scripts/sdma_ab.sh)), ten pairs, both settings inside one session,
alternating, `tg64`, the 8B carried alongside because it is the model the existing claim rests on:

| model | SDMA off | SDMA on | on / off | on slower in |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | **193.87** | 183.56 | **0.947** | **10 of 10 pairs** |
| qwen3-8B Q8_0 | 37.86 | 38.19 | 1.009 | 4 of 10 pairs |

The 1.5B's two ranges do not overlap: 192.67 to 194.93 with SDMA off, 180.05 to 184.03 with it on. Ten
pairs out of ten in the same direction. The 8B shows nothing, which agrees with the August measurement
that the claim was built on.

**So the claim was right about the model it was tested on and wrong as a generalisation.** On this
board, with the navi12 microcode in place, leaving SDMA enabled costs the 1.5B about 5 percent of
decode and costs the 8B nothing.

## All four models, and the obvious explanation fails

The first pass carried two models and suggested the cost might track token rate: the 1.5B decodes five
times as fast as the 8B, so it issues host-device copies five times as often, and this repository has
measured the SDMA engine as faster than the blit path just above 16 KiB and four times slower at 16 MiB
([`logs/sdma-sizes-2026-08-19/`](../sdma-sizes-2026-08-19/)).

Repeating it across all four soak models, six pairs each (`log-four-models`), says that is not it:

| model | decode rate | SDMA off | SDMA on | on / off | on slower in | ranges overlap |
|---|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 194 t/s | **193.53** | 181.91 | **0.940** | **6 of 6** | **no** |
| qwen3.6-35B MoE IQ2_M | 70 t/s | 70.28 | 70.56 | 1.004 | 1 of 6 | yes |
| qwen3-8B Q8_0 | 38 t/s | 38.00 | 37.78 | 0.994 | 4 of 6 | yes |
| qwen3.8-27B UD-IQ3_XXS | 15 t/s | 15.12 | 15.14 | 1.001 | 1 of 6 | yes |

The MoE decodes at 70 tokens a second, nearly twice the 8B, and shows nothing at all. If the cost were
per-copy overhead scaling with how often the model copies, the MoE should sit between the 1.5B
and the 8B. It does not. **Only the fastest model pays, and the ordering in between is flat**, so
whatever this is, "copies more often" does not describe it.

The effect on the 1.5B is not in doubt: two independent interleaved runs, ten pairs and six, 10 of 10
and 6 of 6 in the same direction, ranges not overlapping in either. What causes it is unidentified,
and the obvious candidate has since been ruled out. Tracing every copy a decode issues shows the
largest is 6144 bytes, below the 16384-byte point at which ROCclr uses the SDMA engine at all, and the
extra copy time from enabling SDMA is 1.0 to 1.5 percent of the token on all three models including
the two that show no effect ([`logs/sdma-copy-inventory-2026-09-25/`](../sdma-copy-inventory-2026-09-25/)). No
per-transfer measurement was made, and nothing here distinguishes a threshold in rate from something
particular to that model, its size, or its quantisation.

What this does establish is narrower and enough to change the recipe: the two options step 2 offers are
not equivalent for every model, and the one that is free is `HSA_ENABLE_SDMA=0`.

## Files

`log` is the first run, ten interleaved pairs on two models. `log-four-models` is the second, six
pairs on all four. One line per invocation, with the edge temperature.
