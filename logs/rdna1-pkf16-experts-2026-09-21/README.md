# The mixture-of-experts prefill, off MMQ at last, 2026-09-21

After the tile change ([`logs/rdna1-pkf16-tile-2026-09-20/`](../rdna1-pkf16-tile-2026-09-20/)) the MoE
was the only model still nowhere near Vulkan on prefill, at 0.83, and a kernel trace said why: **60.8
percent of its prefill was MMQ on the experts**, which the packed-fp16 GEMM never saw because MUL_MAT_ID
reaches `ggml_cuda_mul_mat_id` and the GEMM's hook is in `ggml_cuda_mul_mat`
([`trace-moe-pp512-eleven-patches.txt`](trace-moe-pp512-eleven-patches.txt)):

| dispatches | ms | share | kernel |
|---|---|---|---|
| 160 | 1166.0 | 39.3 % | MMQ, iq2_xxs, expert path |
| 74 | 602.8 | 20.3 % | MMQ, iq3_xxs, expert path |
| 360 | 533.9 | 18.0 % | packed-fp16 GEMM, q5_K |
| 140 | 212.7 | 7.2 % | packed-fp16 GEMM, q6_K |
| 6 | 34.7 | 1.2 % | MMQ, iq4_xs, expert path |

Those expert dispatches run at about 1.18 TFLOP/s, on a chip that does 2.08 Tmac/s in MMQ's emulated
int8 and 14.4 TFLOP/s in packed fp16 ([`logs/alu-rates-2026-09-19/`](../alu-rates-2026-09-19/)).

## What the expert path needs, and what it does not

`ggml_cuda_launch_mm_ids_helper` already builds the whole mapping, and both MMQ and MMF call it: a
compact ordering of the used (token, slot) pairs sorted by expert, the activation column each compact
slot reads (`ids_src1`), the destination column it writes (`ids_dst`), and each expert's range
(`expert_bounds`). So the GEMM needs no gather kernel and no compact activation buffer: it reads the
activation column through the map, one indirection per column per stage, and
[`scripts/apply_rdna1_pkf16_id.py`](../../scripts/apply_rdna1_pkf16_id.py) hooks it in ahead of MMQ. One expert per block in z, and
a block walks its expert's column tiles in a loop, so the grid does not have to be sized for the most
heavily used expert.

## The column tile wants to be narrower here

At a 512-token batch with eight of 256 experts used, an expert sees about sixteen of the columns. A
64-wide tile is then mostly padding, and a 16-wide one amortises the weight dequantisation, which is the
expensive part for the IQ types, over too few outputs. Medians of three rounds of two samples, arms
interleaved inside one build, qwen3.6-35B-A3B:

| batch | MMQ | BN=16 | BN=32 | BN=64 | best over MMQ |
|---|---|---|---|---|---|
| 256 | 265.7 | 449.3 | **464.0** | 412.6 | 1.75 |
| 512 | 379.3 | 555.7 | **588.5** | 549.7 | 1.55 |
| 2048 | 370.8 | 544.0 | **574.9** | 536.3 | 1.55 |

32 wins at every batch, which is what the two arguments above predict between them. pp512 goes from 379
to 588 against Vulkan's 457, so this model now prefills 1.29 times faster than Vulkan, not 0.83
times.

## Accuracy improves

For the same reason it did when the kernel first shipped on the dense path. MMQ quantises the
activations to eight bits; the experts here are IQ2_XXS and IQ3_XXS, far narrower than that, so fp16
activations are the better trade and the loss from accumulating in fp16 does not catch up with it. Six
chunks of `wiki.test.raw` at context 512 ([`gates-log`](gates-log)):

| | expert GEMM | MMQ |
|---|---|---|
| qwen3.6-35B-A3B | **6.2265** | 6.2473 |

Greedy text over 64 tokens is identical, and `test-backend-ops` passes 865/865 MUL_MAT_ID cases at all
three column widths.

## The bug worth recording

The first version failed 24 of the 865 cases, at one shape, for some routings and not others. The A-tile
loader puts one thread on each row, so a block has exactly BM threads, which means BN / TN must equal
TM. All three of the column widths were declared with BN / TN = 16 against TM = 8, so half the threads
had no column to write and the untouched destination columns kept whatever the allocator left there,
which is why it depended on the routing. The kernel now carries a static assertion for that constraint
and for TN being even, so the next tile cannot be declared wrong silently.

## Files

`bn-*.jsonl` are the three column widths against MMQ, `gates-log` the perplexity and text run,
`text-moe-expert-gemm.txt` the generation, and `trace-moe-pp512-eleven-patches.txt` the kernel trace that
sized the opportunity. [`scripts/pkf16_expert_bn_sweep.sh`](../../scripts/pkf16_expert_bn_sweep.sh) is
the sweep.
