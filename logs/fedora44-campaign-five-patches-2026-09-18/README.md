# The split throughput campaign with the first form of patch 5, 2026-09-18

[`scripts/campaign_split_f44.sh`](../../scripts/campaign_split_f44.sh) with
`HIPBIN=~/llama-master/build-hip-sad/bin`, 09:20 to 10:00 on the default boot: Fedora 44, kernel 7.2.5 with
the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the corrected comgr, clock policy 1500 MHz,
`ollama` stopped, `hw-watcher.timer` and `crond` stopped. Same script, rounds and models as
[`logs/fedora44-campaign-final-2026-09-18/`](../fedora44-campaign-final-2026-09-18/) (four patches) and
the three-patch campaign of 15 September. The build adds patch 5 in its first form: RDNA1 routed to the
RDNA2 matrix-vector table (one wave per row for every type) plus the `v_sad_u8` activation sums
([`logs/rdna1-mmvq-2026-09-18/`](../rdna1-mmvq-2026-09-18/)). Medians of nine samples:

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 924.7 | 1850.3 | **153.2** | 212.6 |
| qwen3-8B Q8_0 | 279.5 | 394.7 | 37.3 | 39.1 |
| deepseek-r1-14B Q4_K_M | 98.7 | 199.8 | **25.3** | 35.0 |
| qwen3-14B Q4_K_M | 100.5 | 204.7 | **25.7** | 34.7 |
| qwen3.6-35B-A3B MoE IQ2_M | 304.0 | 457.7 | **55.6** | 87.0 |
| qwen3.8-27B UD-IQ3_XXS | 71.2 | 105.0 | **11.4** | 17.6 |

Against the four-patch campaign, decode: 1.5B **1.286**, 8B **0.957**, deepseek-14B **1.178**, qwen3-14B
**1.178**, MoE **1.601**, 27B **1.450**; prefill 1.000 to 1.002 on every row, as it should be for a change
that touches only the matrix-vector kernel; every Vulkan row 0.999 to 1.001.

**One regression, and it is informative.** qwen3-8B Q8_0 loses 4 percent of decode, with spreads of 2.6
percent, so it is real. Q8_0's `vec_dot` is short and its matvec is memory-bound; one wave per row
starves it where the K-quants, whose `vec_dot` is long, are kept busy. That is the same distinction
llama.cpp's RDNA4 table already draws with its per-type whitelist. The second form of the patch gives
RDNA1 its own table entry: one wave per row for the K-quants and IQ types, the four-warp form for Q8_0
and the Q4/Q5 legacy types; its campaign is
[`logs/fedora44-campaign-patch5-2026-09-18/`](../fedora44-campaign-patch5-2026-09-18/).
