# What kernel fusion is already worth here, and where the rest of the floor is, 2026-09-23

[`logs/dispatch-floor-2026-09-22/`](../dispatch-floor-2026-09-22/) measured what a dispatch costs on
this board when the kernel does nothing: 1.78 microseconds by fit, 2.03 for an empty kernel. That makes
a prediction worth testing. If the floor is what holds the mixture-of-experts model back, then anything
that removes dispatches should be worth more here than the same change is worth on hardware with a
smaller floor, and it should be worth most on the models that issue the most dispatches per unit time.

ggml-cuda already fuses, and `GGML_CUDA_DISABLE_FUSION=1` turns all of it off, so the question can be
asked directly. Reading `ggml/src/ggml-cuda/ggml-cuda.cu` in the build these measurements were taken
with, the full list of patterns is `rope`+`view`+`set_rows`, `mul_mat`+`add`+`mul_mat`+`add`+`glu` and
its `mul_mat_id` form, `mul_mat`+`mul_mat`+`glu`, `rms_norm`+`mul`+`add`, **`rms_norm`+`mul`**,
`ssm_conv`+`add`+`unary`, `ssm_conv`+`unary`, `unary`+`mul` for silu, sigmoid and softplus,
`unary`+`sqr`, `scale`+`tanh`+`scale`, `topk_moe`, a gated-delta-net cache fusion, and one that is not
expressed as a pattern at all but as a loop: **runs of up to eight consecutive `ADD` or `MUL` nodes**,
collapsed into one variadic kernel by `ggml_cuda_op_fused_add` / `_fused_mul`.

Those last two matter for what follows, and an earlier version of this page missed both: the list was
assembled by grepping for the named `_ops` arrays, which does not find the patterns passed as inline
initialiser lists, and does not find the multi-add loop at all.

## What it is worth

[`scripts/fusion_ab.sh`](../../scripts/fusion_ab.sh), six interleaved pairs per model, `tg64`:

| model | decode rate | fusion on | fusion off | on / off | on faster in |
|---|---|---|---|---|---|
| qwen3.6-35B-A3B MoE IQ2_M | 70 t/s | **70.43** | 56.98 | **1.236** | 6 of 6 |
| qwen2.5-1.5B Q4_K_M | 194 t/s | **193.53** | 168.17 | **1.151** | 6 of 6 |
| qwen3.8-27B UD-IQ3_XXS | 15 t/s | 15.07 | 15.05 | 1.001 | 4 of 6 |

**Fusion is worth 24 percent of decode on the MoE and 15 on the 1.5B, and nothing on the 27B.** Six
pairs of six on both models that move.

That ordering is the prediction. Taking the dispatch counts from
[`logs/moe-decode-2026-09-20/`](../moe-decode-2026-09-20/) and the floor at 1.78 microseconds:

| model | dispatches per token | mean kernel | token | floor | floor as a share | fusion worth |
|---|---|---|---|---|---|---|
| qwen2.5-1.5B | 375 | 13.0 us | 5.17 ms | 0.67 ms | 12.9 % | 15.1 % |
| qwen3.6-35B MoE | 1298 | 9.3 us | 14.20 ms | 2.31 ms | 16.3 % | 23.6 % |
| qwen3.8-27B | 1745 | 39.0 us | 66.36 ms | 3.11 ms | 4.7 % | 0.1 % |

The 27B issues the most dispatches of the three and cares least, because its kernels are four times
longer, so the floor is 4.7 percent of its token, not 16. The two models where the floor is a
large share are the two where fusion pays, and in the same order.

The measured gain exceeds the floor share on both, which is what should happen: turning fusion off does
not merely stop removing dispatches, it splits fused patterns back into their components and so adds
dispatches above the 1298 the trace counted. The arithmetic is consistent instead of exact, and no
attempt was made to count the dispatches in the fusion-off arm.

## Where the remaining floor is

The trace those counts come from was taken with fusion **on**, so what it shows is what survives it. Of
1298 dispatches a token, 748 are elementwise, normalisation, copy or gather kernels, costing 1332
microseconds of launch overhead, 9.6 percent of the token:

| kernel | per token | mean | the floor is |
|---|---|---|---|
| `k_bin_bcast` | 216.6 | 2.47 us | 72 % of it |
| `rms_norm_f32` | 129.0 | 3.09 us | 58 % |
| `unary_op_kernel` | 68.9 | **1.51 us** | **118 %** |
| `unary_gated_op_kernel` | 68.9 | 2.20 us | 81 % |
| `l2_norm_f32` | 59.1 | 2.96 us | 60 % |
| `cpy_scalar` | 39.4 | 4.95 us | 36 % |
| `k_get_rows_float` | 30.5 | 2.73 us | 65 % |
| `concat_cont` | 29.5 | 3.08 us | 58 % |
| `k_set_rows` | 19.7 | 2.15 us | 83 % |

`unary_op_kernel` runs for less time than it costs to launch: 69 of them a token, each 1.51
microseconds of work behind a 1.78 microsecond door.

## What this does and does not say

It says that on this board the cost of a dispatch is large enough that fusion is worth 15 to 24 percent
on the models that dispatch often, that ggml-cuda's existing coverage is already collecting most of
that, and that 748 dispatches a token of small elementwise work are still outside it on the MoE.

It does not say how much extending that coverage would buy. Fusing *k* operations saves *k-1* launches,
not *k*, and which of these can fuse at all depends on the graph, not on their size: an
elementwise op with two consumers cannot be folded into its producer. 9.6 percent is therefore a
ceiling on what is left in this direction on this model, not an estimate.

## How many are in a foldable position

`GGML_SCHED_DEBUG=2` prints every node of the graph with the number of consumers its output has
(`sched-graph-one-token.txt`, one decode graph of the MoE, 2355 nodes; more than the 1298 dispatches
because view operations do not dispatch and fused patterns collapse several nodes into one).

Counting the elementwise, normalisation, copy and gather nodes by consumer count:

| | nodes |
|---|---|
| elementwise, normalisation, copy, gather | 1424 |
| **of those with exactly one consumer** | **954 (67 %)** |
| with more than one consumer | 470 |

By operation, the single-consumer share is what decides whether a pattern is worth pursuing at all:

| operation | single-consumer | total |
|---|---|---|
| `RMS_NORM` | **131** | 131 |
| `L2_NORM` | **60** | 60 |
| `ADD` | 351 | 430 |
| `GET_ROWS` | 161 | 162 |
| `MUL` | 161 | 281 |
| `CLAMP` | 40 | 40 |
| `DIV` | 40 | 40 |

Every `RMS_NORM` and every `L2_NORM` in the graph has exactly one consumer, so two thirds of this class
of node is in a position where folding is not blocked by having its output needed elsewhere. Structural
possibility is all this counts, and the next section is where that turns out to matter: a node being
foldable says nothing about whether it has *already been folded*, and most of these had been.

## The edges, and the two patterns that turned out not to be missing

Which fused pattern is missing needs the edges, and they are not recoverable from the dump above: the
debug line truncates tensor names to twenty characters and **layers reuse them**, so `norm-0` appears at
nodes 1, 49 and 59, and 117 of the 1979 distinct names are used more than once. An attempt to
reconstruct producer-consumer chains by name was made and discarded for that reason; its output looked
plausible and was wrong, reporting one `RMS_NORM` chain where there are 131.

[`scripts/patch_sched_srcidx.py`](../../scripts/patch_sched_srcidx.py) makes the scheduler print each
source as a node index as well as a name, six lines in `ggml_backend_sched_print_assignments`. It was
applied, used, and reverted; it is not part of the patch set. The dump it produces is
`sched-graph-srcidx.txt`, taken from a **CPU-only build** so that the measurement build was never
touched: the graph comes from llama.cpp and not from the backend, and the two dumps agree exactly,
2355 nodes with every one of the 23 operation counts identical.

This section first reported two patterns as missing, the expert-accumulation `ADD` chain at 4.0 percent
of the token and 111 unfused `RMS_NORM`+`MUL` pairs at 1.4. **Both were wrong.** ggml-cuda fuses both,
and the traces in this repository show both fusions firing in every measurement taken here. What follows
is the correction and what is actually left.

### `RMS_NORM` + `MUL`: already fused, nothing left

`ggml-cuda.cu` fuses this as a two-operation pattern in its own right, independently of any `ROPE` form,
calling `ggml_cuda_op_rms_norm_fused`. The claim that 111 pairs fell through for want of a third
operation was a misreading of the pattern list.

The traces settle it without needing the source. `norm.cu` declares
`rms_norm_f32<block_size, do_multiply, do_add>`, and `do_multiply=true` is reachable **only** through
`rms_norm_mul_f32_cuda`, which only `ggml_cuda_op_rms_norm_fused` calls; the unfused
`ggml_cuda_op_rms_norm` launches `<block_size, false>`. Every `rms_norm` instantiation in every trace in
this repository is `ILb1ELb0E`, that is `<..., true, false>`:

| instantiation | dispatches, MoE decode, 64 tokens | multiply fused |
|---|---|---|
| `rms_norm_f32<1024, true, false>` | 5103 | yes |
| `rms_norm_f32<256, true, false>` | 3150 | yes |
| any `<..., false, ...>` | **0** | n/a |

8253 over 64 tokens is 129 a token, the same 129 counted as surviving dispatches above. They survive
because fusing a `MUL` into an `RMS_NORM` removes the `MUL`'s dispatch, not the `RMS_NORM`'s.
**All 131 pairs were already fused, and the 1.4 percent does not exist.**

`do_add=false` throughout is worth writing down separately: the three-operation `rms_norm`+`mul`+`add`
fusion is present in the build and never fires on this graph.

### The expert-accumulation chain: already fused, nine into two

The chain is as described, nine binary adds deep, forty times, every intermediate with one consumer,
but the multi-add loop already collapses it. `k_bin_bcast` is variadic, and with a pack of *N* addends it
sums `src0` plus *N* sources in one dispatch. The MoE decode trace contains:

| `k_bin_bcast` instantiation | dispatches / 64 tokens | per token | adds folded in |
|---|---|---|---|
| `op_add`, pack 7 | 2520 | **39.4** | 7 |
| `op_add`, pack 2 | 2520 | **39.4** | 2 |
| `op_add`, pack 1 | 3780 | 59.1 | 1, unfused |
| `op_mul`, pack 1 | 5040 | 78.8 | 1, unfused |

39.4 a token of each against 40 layers is one of each per layer: the nine adds dispatch as **one
seven-node add plus one two-node add**. 280 of the 320 dispatches the earlier claim offered to remove
were already gone.

The real-edge dump shows why it splits seven and two rather than eight and one, which is the loop's cap.
Taking layer 3, nodes 359 to 365 are seven consecutive `ADD` nodes chained through `src[0]`, fused as
one; then `ffn_out-3` is node **373** and `l_out-3` is 374. The gap is nodes 366 to 372, the
shared-expert FFN. `ggml_can_fuse` requires consecutive node indices, so the run ends at `ffn_moe_out`
not because of a layout or consumer-count obstacle but because the two halves of the chain, adjacent in
dataflow, are eight nodes apart in graph order.

### What is actually left

| | dispatches / token | worth |
|---|---|---|
| merging the two halves of the adder chain into one | 40 | **0.5 %** |

That is all of it, and it is not a ggml-cuda pattern to add. Emitting the shared-expert FFN before
the expert accumulation would make the nine adds consecutive, and the existing loop would fuse eight of
them unchanged; that is a change in how llama.cpp builds the graph, not in the backend. Half a percent
of one model's token is a poor reason to make it.

## What this still does not say

The 0.5 percent prices a dispatch removed and nothing else. It assumes the nine-wide kernel costs what
the seven-wide one costs, which is likely for a memory-bound elementwise add over 8K elements but was
not measured, and it assumes the reordering is free, which needs checking against whatever ordering
constraint put the shared expert there.

It is one model's graph. The 1.5B and the 27B were measured end to end here but not traced, and the 27B
says plainly that a model with long kernels gets nothing from any of this.

The wider conclusion is unchanged and is the useful one: **fusion is worth 24 percent of the MoE's
decode and 15 of the 1.5B's on this board, and ggml-cuda is already collecting nearly all of it.** The
9.6 percent ceiling above is a ceiling on the class, not a queue of available work; having looked for
the work, there is about half a percent of it.

## Files

`log` is the six interleaved pairs per model.
