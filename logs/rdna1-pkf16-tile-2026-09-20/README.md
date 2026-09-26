# The prefill GEMM's tile was tuned on the wrong thing, 2026-09-20

Patch 8's tile, 128 weight rows by 128 columns of tokens, came from a standalone prototype measured at
2048 tokens ([`logs/pk-gemm-prototype-2026-09-19/`](../pk-gemm-prototype-2026-09-19/)). In the backend
it is the wrong shape, and by a wide margin. Halving the column tile to 64 is worth 1.07 to 1.55 times
prefill at every batch size from 256 to 2048 on every model measured, and changes no results at all.

## Why 128 columns is wrong here

Two reasons, both visible without running anything. A 512-token batch is the one the benchmarks use, and
a 128-wide column tile divides it into four, so the grid for a 512-row matrix is 4 by 4, sixteen
workgroups on a board with sixteen compute units and nothing left over to hide latency with. And the
tile costs registers: at 128 by 128 each thread holds 128 accumulators and the kernel takes 221 of them,
which allows four waves per SIMD, where 128 by 64 takes 125 and allows eight
([`isa-tiles.log`](isa-tiles.log)).

| BM | BN | TM | TN | VGPRs | shared memory | waves per SIMD | column tiles at 512 tokens |
|---|---|---|---|---|---|---|---|
| 128 | 128 | 8 | 16 | 221 | 17408 | 4 | 4 |
| 128 | 64 | 8 | 8 | 125 | 13312 | 8 | 8 |
| 128 | 32 | 8 | 4 | 89 | 11264 | 11 | 16 |
| 64 | 64 | 4 | 16 | 132 | 9216 | 7 | 8 |
| 64 | 128 | 4 | 32 | 245 | 13312 | 4 | 4 |

The A-tile loader puts one thread on each row, so a block has exactly BM threads and the shapes are
constrained to BN / TN == TM. All five above satisfy it.

## The sweep

pp512, medians of three rounds of three samples with the arms interleaved inside one build, selected by
`GGML_RDNA1_PKF16_TILE` ([`scripts/pkf16_tile_sweep.sh`](../../scripts/pkf16_tile_sweep.sh)):

| model | 128x128 | 128x64 | 128x32 | 64x64 | 64x128 | Vulkan |
|---|---|---|---|---|---|---|
| qwen2.5-1.5B | 1245.8 | **1784.0** | 1553.8 | 1258.0 | 757.8 | 1849.8 |
| qwen3-14B | 151.7 | **193.8** | 131.5 | 128.9 | 95.7 | 204.7 |
| qwen3.6-35B MoE | 340.9 | **377.8** | 349.1 | 343.1 | 280.0 | 456.9 |
| qwen3.8-27B | 81.2 | **102.5** | 69.5 | 73.1 | 51.5 | 105.0 |

As ratios of the shipped tile: 1.43, 1.28, 1.11 and 1.26. Spread 0.0 to 3.7 percent except the 1.5B's
128x64 at 6.8. Narrowing the column tile further to 32 helps the 1.5B and the MoE and hurts the other
two, and narrowing the row tile instead is worse everywhere; 64 by 128, which keeps the wide column tile
and halves the rows, is much worse everywhere, which is the same story from the other side.

## It is not a small-batch effect

Two models at four batch sizes, medians of two rounds of two samples:

| model | batch | 128x128 | 128x64 | 128x32 |
|---|---|---|---|---|
| qwen2.5-1.5B | 256 | 875.9 | 1317.5 | **1355.2** |
| | 512 | 1244.8 | **1783.5** | 1551.5 |
| | 1024 | 1230.4 | **1580.3** | 1294.1 |
| | 2048 | 1190.1 | **1506.1** | 1410.3 |
| qwen3.8-27B | 256 | 75.4 | **99.7** | 73.3 |
| | 512 | 82.0 | **102.5** | 69.5 |
| | 1024 | 82.0 | **93.4** | 66.4 |
| | 2048 | 82.0 | **87.8** | 63.3 |

128 by 64 is ahead at every batch on both models. The margin narrows as the batch grows, from 1.32 to
1.55 times at 256 to 1.07 to 1.27 at 2048, but it never reverses, so no batch-dependent rule is needed
and the tile changes. The prototype's preference for 128 columns at 2048 tokens did not survive
the move into the backend, where the kernel shares the machine with everything else in the graph.

## It changes no results

Perplexity is bit-identical between the two tiles on all four models, six chunks of `wiki.test.raw` at
context 512 ([`gates-log`](gates-log)):

| model | 128x64 | 128x128 |
|---|---|---|
| qwen2.5-1.5B | 10.2088 | 10.2088 |
| qwen3-14B | 8.6550 | 8.6550 |
| qwen3.8-27B | 6.2737 | 6.2737 |
| qwen3.6-35B MoE | 6.2473 | 6.2473 |

That is what the arithmetic predicts, not a lucky agreement: with the f32 promotion at every
stage, each output element sums the same 32 products in fp16 in the same order, and the tile only
decides which thread owns which output. `test-backend-ops` passes 1186/1186 MUL_MAT cases and greedy
text is coherent on both models checked ([`text-1.5b.txt`](text-1.5b.txt), [`text-27b.txt`](text-27b.txt)).

## The tile change reverses an earlier exclusion

q8_0 was the one supported type left on MMQ, because at the 128 by 128 tile the GEMM measured 15 to 20
percent behind for it ([`logs/round11-2026-09-20/`](../round11-2026-09-20/)). At 128 by 64 that reverses
and keeps reversing as the batch grows. qwen3-8B, medians of three rounds of three samples with the arms
interleaved (`q8-*.jsonl`):

| batch | q8_0 on MMQ | q8_0 on the GEMM | ratio |
|---|---|---|---|
| 256 | 245.1 | 351.9 | 1.44 |
| 512 | 260.6 | 379.3 | 1.46 |
| 1024 | 244.5 | 387.8 | 1.59 |
| 2048 | 221.5 | 389.3 | 1.76 |

This one does change the arithmetic, where the tile change did not. MMQ quantises the activations to
eight bits and accumulates in int32; the GEMM converts both sides to fp16 and accumulates there. For the
types whose weights are narrower than eight bits that trade is a gain, and the gates improved when the
kernel first shipped. q8_0's weights are already the int8 operands MMQ wants, so there is no weight-side
gain to set against the fp16 accumulation: six chunks of `wiki.test.raw` at context 512 read 9.4017
against MMQ's 9.3944, 0.078 percent worse, with greedy text over 64 tokens identical
([`q8-gates-log`](q8-gates-log)). That is smaller than the 0.2 percent the 14B moved when this kernel
took over its q4_K weights, which is already shipped, so q8_0 joins the others;
`GGML_RDNA1_PKF16_Q8=0` puts it back.

## The other two switches go flat

The f32 promotion interval and the tile prefetch were also chosen at 128 by 128, where prefetching cost
a wave of occupancy and 19 percent on the 14B ([`logs/round14-2026-09-20/`](../round14-2026-09-20/)). At
128 by 64 all six combinations of promotion 1, 2 or 4 with prefetch on or off land within 0.5 percent of
each other on the 1.5B and the 27B (`promote-*.jsonl`), so the shipped values stay where they are and
prefetching no longer hurts: its doubled shared memory is 26624 bytes instead of 34816, which the
smaller tile can afford. Nothing to change, but it closes the question at the new tile instead of
leaving it answered for the old one.

## Files

`sweep-*.jsonl` are the five-tile sweep at pp512, `batch-*.jsonl` the three-tile sweep across batch
sizes, `q8-*.jsonl` and `q8-gates-log` the q8_0 retest, `promote-*.jsonl` the promotion and prefetch retest, `isa-tiles.log` the register and shared-memory
figures per tile, `gates-log` the perplexity and correctness run for the tile change, and `text-*.txt`
the greedy generations.
