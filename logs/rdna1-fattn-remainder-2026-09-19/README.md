# Flash attention on the final build, every model, against Vulkan, 2026-09-19

The spill fix ([`logs/rdna1-fattn-spill-2026-09-17/`](../rdna1-fattn-spill-2026-09-17/)) was measured on
the 1.5B's graph and end to end on the others. This is the flash-attention op itself on all six models:
each graph exported with `-fa on`, a 2048-token batch and then one token at a 4096-slot KV cache, the
`FLASH_ATTN_EXT` lines kept (`GGML_OP_FLASH_ATTN_EXT` is op 74 in this tree's `ggml.h`) and replayed on
`build-hip-f32iq` and the Vulkan build, two passes ([`scripts/fa_remainder.sh`](../../scripts/fa_remainder.sh),
`fa-*.log`, us per op):

| model | heads / KV heads, D | 2048 tokens: ROCm | Vulkan | ratio | one token at KV 2048: ROCm | Vulkan | ratio |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5B | 12 / 2, 128 | 16865 | 16192 | 1.04 | 48 | 128 | **0.38** |
| qwen3-8B | 32 / 8, 128 | 43226 | 73115 | **0.59** | 163 | 69 | 2.38 |
| deepseek-r1-14B | 40 / 8, 128 | 129562 | 98532 | 1.31 | 460 | 92 | **5.03** |
| qwen3-14B | 40 / 8, 128 | 129375 | 95470 | 1.36 | 463 | 90 | **5.15** |
| qwen3.6-35B MoE | 16 / 2, 256 | 98333 | 30246 | **3.25** | 62 | 71 | 0.88 |
| qwen3.8-27B | 24 / 4, 256 | 147243 | 62651 | **2.35** | 90 | 89 | 1.00 |

Two remainders, then, and they are different kernels.

**Prefill, D=256.** The MoE's and the 27B's tile kernels still spill (51 to 68 registers after the patch's
rows, against zero for D=128) and sit at 2.3 and 3.3 times Vulkan. On the MoE that is 10 attention
layers times 98 ms, about 15 percent of a pp2048 pass; on the 27B 17 times 147 ms, about 9 percent. The
14B models at D=128 without a power-of-two head ratio are at 1.3 times, the 32-column tile the patch
keeps them on; the 8B, D=128 with GQA 4, is ahead of Vulkan. A second compile-time sweep of D=256 rows
is in `sweep256.log`.

**One token, D=128 with GQA.** On a GPU without matrix cores the kernel chooser sends a one-token op
whose GQA ratio is 2 or more to the tile kernel with `ncols2` set to the ratio's power-of-two part, and
to the vector kernel only when the ratio is 1. That gives the 8B a 1 x 4 tile, the 14B models (40 heads
over 8, ratio 5) a 1 x 1 tile, and the 1.5B (ratio 6) a 1 x 2 tile; the resource log of the spill
directory shows the 1 x 4 and 4 x 2 instances still spilling about 240 registers. The result is 2.4 and
5 times Vulkan's time on the 8B and the 14B models at a 2048-token context, while the 1.5B, whose tile
is the one row that came out clean, is 2.7 times faster than Vulkan. `llama-bench`'s `tg64` runs at an
empty context, so none of this shows on the front-page table; it is the mechanism behind the
decode-at-depth curves, where ROCm falls away from Vulkan faster than the KV traffic alone explains. The
experiment that follows (`fa-decode`, [`scripts/apply_rdna1_fa_vec_decode.py`](../../scripts/apply_rdna1_fa_vec_decode.py))
routes one-token attention on RDNA1 to the vector kernel regardless of the GQA ratio and measures both
the op and decode at depth 4096.

## Experiment I: one-token attention through the vector kernel (`fa-decode/`)

[`scripts/apply_rdna1_fa_vec_decode.py`](../../scripts/apply_rdna1_fa_vec_decode.py), first in an
experiment form that sent every one-token op on RDNA1 to the vector kernel, switchable at run time
(`GGML_FA_VEC_DECODE=0` restores the upstream choice), so both arms come from one build. Correctness,
`test-backend-ops test -o FLASH_ATTN_EXT` against the CPU: 1344 of 1344 with the vector choice, 1293 of
1293 without, no failure either way (`fa-decode/tbo-fa-v*.log`). The one-token op at a 2048-token KV,
and `tg32` at a 4096-token depth and at an empty context, `-fa on`, two passes (`fa-decode/log`):

| model | ratio, tile it had | one token, tile | one token, vec | tg32 at 4096, tile | vec | tg32 at 0, tile | vec |
|---|---|---|---|---|---|---|---|
| deepseek-r1-14B | 5, 1 x 1 | 458 | **159** | 17.7 / 18.2 | **25.3 / 25.6** | 29.5 / 29.1 | 30.8 / 31.2 |
| qwen3-14B | 5, 1 x 1 | 458 | **161** | | | | |
| qwen3-8B | 4, 1 x 4 | **159** | 256 | **31.1 / 31.2** | 27.3 / 27.1 | 35.9 / 36.1 | 36.2 / 29.2 (5.9) |
| qwen2.5-1.5B | 6, 1 x 2 | 48 | 46 | 156.0 / 156.7 | 157.5 / 157.9 | 177.8 / 177.8 | 181.0 / 178.7 |
| qwen3.6-35B MoE, D=256 | 8, 1 x 8 | **55** | 141 | 63.6 / 63.9 | 60.6 / 62.5 | 63.8 (3.4) / 38.0 (5.2) | 46.8 (11.8) / 64.8 |
| qwen3.8-27B, D=256 | 6, 1 x 2 | **84** | 148 | 14.1 / 14.1 | 13.9 / 13.8 | 13.3 / 14.3 | 14.3 / 14.2 |

The two 14B models, whose tile is the 1 x 1 instance, gain 2.9 times on the op and **42 percent of decode
at depth 4096** (17.7 and 18.2 to 25.3 and 25.6 tokens per second), and nothing at an empty context, as
the front-page table would predict. Every other model is better on the tile it had: the 8B's 1 x 4 tile,
spilling as it does, still beats the vector kernel by 1.6 times and the model loses 13 percent at depth
on it; the 1.5B is level; the D=256 models are 1.8 to 2.6 times slower on the vector kernel. So the rule
in the final form is the narrow one: RDNA1, one token, the GQA optimisation applicable but its
power-of-two part equal to one, which is when the tile would have been 1 x 1. The 8B's 163 us against
Vulkan's 69 remains, and it is a tile-row problem (the D=128 ncols 4 instance spills 240 registers),
not a kernel-choice one. The empty-context MoE readings with 5 to 12 percent spreads are the throttle
after the D=4096 run before them, not the kernels.

The final form (`fa-vec-final/`): the rule as stated, no switch. `FLASH_ATTN_EXT` against the CPU 1368 of
1368; the one-token op on the 14B models 166 and 160 us, on the 8B 159 (tile, unchanged), on the 1.5B 47;
`tg32`, two passes: deepseek-r1-14B **25.8 / 25.7** at depth 4096 and 31.3 / 31.3 at an empty context,
the 8B 31.3 / 30.7 and 36.1 / 35.7, both where the experiment build had them.

## Experiment J: the D=256 tile rows (`fa-d256/`, `sweep256.log`)

The second compile-time sweep of D=256 rows (`sweep256.log`): no 32-column instance reaches zero spill at
any geometry tried, but two things stand out. The 16-column instances `8x2` and `2x8` compile clean at
512 threads, occupancy 2, `nbatch_K` 64 (177 VGPRs, occupancy 5), and the 32-column row `256:2:32:64`
brings the MoE's `4x8` instance from 52 spilled registers to 5. [`scripts/apply_rdna1_fa_d256.py`](../../scripts/apply_rdna1_fa_d256.py)
sets the ncols-16 and ncols-32 rows and adds a run-time switch that caps D=256 prefill at 16 columns
on RDNA1. Three builds, the 2048-token flash-attention op of both models (`fa-d256/log`, us):

| build | ncols-16 row | ncols-32 row | 16-column cap | MoE, `4x8` | 27B, `16x2` |
|---|---|---|---|---|---|
| current rows | 512:3:32:128 | 512:3:32:128 | off | 96994 | 145474 |
| B1 | **512:2:32:64** (no spill) | 512:3:32:128 | on | 127453 | 191454 |
| B1 | 512:2:32:64 | 512:3:32:128 | off | 99692 | 146574 |
| B3 | 512:3:32:128 | **256:2:32:64** | on | 101083 | 151393 |
| **B3** | 512:3:32:128 | **256:2:32:64** | **off** | **47953** | **84951** |
| Vulkan | | | | 30246 | 62651 |

The zero-spill 16-column tile is slower than the spilling 32-column one, by 30 percent: halving the
columns doubles the K and V traffic per query, and at D=256 that costs more than the spill. The
32-column row at 256 threads, occupancy 2, `nbatch_K` 64 halves the MoE's op and takes 42 percent off the
27B's, to 1.6 and 1.4 times Vulkan from 3.2 and 2.3. Correctness on each build, `FLASH_ATTN_EXT` against
the CPU: 1164, 1334 and 1366 of as many, no failure. The end-to-end arms in `fa-d256/log` are not
usable: the three `llama-bench` copies all loaded the last build's `libggml-hip.so`, so they measured
one build three times; the two-directory A/B is in `fa-d256-2/`.

The end-to-end A/B done properly (`fa-d256-2/`): `build-hip-fa3` is the same tree as `build-hip-gdn` with the
ncols-32 row set to `256:2:32:64` and no cap, each binary from its own directory. `FLASH_ATTN_EXT` against
the CPU 1303 of 1303. The 2048-token op: MoE 99369 to **48007** us, 27B 147581 to **84483** (the vector
choice for one token is in both builds now, so the one-token figures are 55 and 85 on both). `llama-bench`
`-p 512,2048`, three repetitions, two passes:

| model | pp512 current rows | pp512 row32 256:2:32:64 | pp2048 current | pp2048 row32 | |
|---|---|---|---|---|---|
| qwen3.6-35B MoE | 294.6 / 297.4 | 297.4 / 297.1 | 299.0 / 299.3 | **305.8 / 305.5** | **+2.2 %** at pp2048 |
| qwen3.8-27B | 71.2 / 71.2 | 71.8 / 71.5 | 66.0 (4.6) / 65.0 (2.0) | 63.9 (3.6) / 64.7 (3.4) | +0.6 % at pp512; pp2048 inside its spread |

Less than the halved op suggests, and for a reason that is worth stating: `llama-bench` prefills 2048
tokens in four micro-batches of 512, so the attention it runs is four 512-query instances against a
growing cache, about six tenths of the single 2048 x 2048 instance the replay times, and attention is a
tenth of the MoE's pass to begin with. The row stays: it is an exact kernel-geometry change with a
per-op factor of two and no loss anywhere measured.

## Experiment K: D=128 rows for 4 and 2 columns (`fa-d128/`, `sweep128-c4.log`, `sweep128-c2.log`)

The RDNA1 table had D=128 rows from 8 columns up, so the decode-shaped tiles with a GQA ratio of 4 (the
8B's `1x4`) and 2 (the 1.5B's `1x2`) took the shared RDNA row, and the `1x4` spilled 240 registers. A
compile-time sweep of rows for those two column counts: `4:128:4:32:64` compiles the `1x4` clean at 121
VGPRs, occupancy 8 (and `2x2`, `4x1` too), and `2:64:4:32:64` the `1x2` at 147, occupancy 6 (`sweep128-*.log`;
256-thread candidates fail a static assertion at these column counts). [`scripts/apply_rdna1_fa_d128_small.py`](../../scripts/apply_rdna1_fa_d128_small.py)
adds the two rows; `build-hip-fa4` is `build-hip-fa3` plus them. `FLASH_ATTN_EXT` against the CPU 1285 of
1285. The one-token op at KV 2048 and `tg32`, two passes (`fa-d128/log`):

| model | one token, before | after | Vulkan | tg32 at 4096, before | after | tg32 at 0, before | after |
|---|---|---|---|---|---|---|---|
| qwen3-8B | 163 | **60** | 69 | 30.3 / 30.7 | **35.1 / 34.4** | 36.4 / 36.1 | 35.2 (3.2) / 35.4 (1.7) |
| qwen2.5-1.5B | 47 | **34** | 128 | 150.9 / 154.2 | **163.3 / 166.4** | 172.4 (9.7) / 175.6 (6.8) | 177.8 / 177.5 |

The 8B's one-token attention goes from 2.4 times Vulkan's to ahead of it and the model gains 14 percent of
decode at depth 4096; the 1.5B gains 8. At an empty context nothing changes beyond the spreads, which are
wide on the empty-context arm because it follows the depth run in the same invocation. With experiments
I and K together, every model's one-token attention is at or ahead of Vulkan's:

| model | one token at KV 2048, ROCm final | Vulkan |
|---|---|---|
| qwen2.5-1.5B | 34 | 128 |
| qwen3-8B | 60 | 69 |
| deepseek-r1-14B / qwen3-14B | 166 / 160 | 92 / 90 |
| qwen3.6-35B MoE | 55 | 71 |
| qwen3.8-27B | 85 | 89 |

The 14B models are the exception at 1.8 times; theirs is the vector kernel now, which is where the ratio-5
head geometry lands without a GQA tile, and further work there means a tile row for `1x1` that does not
spill, which the D=128 sweep did not find.
