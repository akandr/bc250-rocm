# ggml-cuda can overlap independent work too, and the switch is off, 2026-09-24

> **Correction, 2 October 2026.** The speeds here stand, but the option, as shipped, computes wrong
> tokens: the Q branch overwrites `attn_norm` while the K and V projections on the other streams still
> read it, which turns qwen3-8B and qwen3-14B into word salad. The perplexity gates re-run here cannot
> see it, because they evaluate prompts and the option only acts on single-token decode.
> [`logs/graphopt-correctness-2026-10-02/`](../graphopt-correctness-2026-10-02/) has the evidence and a
> fix that keeps the speed.

This came out of trying to close the last of the MoE's decode gap, and it is the only thing in that
line of work that makes the board faster.

## Vulkan overlaps, and by how much

[`logs/floor-vs-vulkan-2026-09-23/`](../floor-vs-vulkan-2026-09-23/) could not say whether ggml-vulkan
overlaps independent dispatches, because the instrument it used for Vulkan's kernel times, the perf
logger, **calls `ggml_vk_sync_buffers` after every node**. Its totals are therefore a fully serialized
GPU span, so they are larger than the token they sit in.

`GGML_VK_PERF_LOGGER_CONCURRENT=1` writes timestamps only where a sync is already required, so it
measures the real execution without adding barriers. The difference between the two is the overlap
(`vulkan-overlap.txt`, MoE decode, stock `build-vk`):

| | GPU span a token |
|---|---|
| logger serial, a sync after every node | 13.00 ms |
| logger concurrent, real execution | **11.60 ms** |

**Overlap is worth 1.40 ms, 10.8 percent, to ggml-vulkan on this model**, and 11.60 ms of GPU span plus
1.58 ms of gap reconciles with that build's 13.18 ms token.

## ggml-cuda has the same machinery, opt-in, and it was off

Reading `ggml-cuda.cu`, `ggml_backend_cuda_graph_optimize` looks for fork/join regions in the graph,
Q/K/V branches and the like, and runs the branches on separate streams with an event to fork and join.
It is gated:

```c
static bool enable_graph_optimization = [] {
    const char * env = getenv("GGML_CUDA_GRAPH_OPT");
    return env != nullptr && atoi(env) == 1;
}();
if (!enable_graph_optimization) { return; }
```

Off unless asked for, and it also requires CUDA graphs to be enabled. With `llama-bench -v`, which
exposes the `GGML_LOG_DEBUG` lines the default log level hides, the default configuration prints
**nothing at all**: no `Adding stream at node`, no `Launching N streams`. The pass never runs. With
`GGML_CUDA_GRAPH_OPT=1` the 1.5B adds 168 stream nodes and launches three streams 56 times a token.

## What it is worth

The headline build, `build-hip-pkf16`, identified by its gates reproducing exactly (below) and by
pp512 landing at 1777 against the front page's 1796. One `llama-bench` process per data point, arms
interleaved, `ab-1.5b-headline-build.txt`:

| qwen2.5-1.5B | n | off | on | ratio | on faster in |
|---|---|---|---|---|---|
| **tg64** | 10 | 182.98 | **200.83** | **1.098** | **10 of 10** |
| pp512 | 10 | 1776.8 | 1779.1 | 1.001 | n/a |

**Decode gains 9.8 percent and the ranges do not touch**: the worst on-reading, 199.09, is above the
best off-reading, 183.74. **Prefill does not move**, which is what should happen; prefill runs few long
kernels that already fill the machine, and there is nothing to gain by overlapping them.

Across the model set (`ab-six-models.txt`, tg32, six interleaved pairs each), the gain tracks whether
the pass finds anything to fork:

| model | concurrent launches | off | on | ratio |
|---|---|---|---|---|
| qwen2.5-1.5B | 56 | 182.22 | 201.12 | **1.104** |
| qwen3-8B q8_0 | 72 | 36.63 | 37.71 | 1.029 |
| qwen3-14B | 80 | 31.01 | 31.86 | 1.027 |
| deepseek-r1-14B | 96 | 31.29 | 32.26 | 1.031 |
| qwen3.6-35B-A3B MoE | **0** | 62.45 | 63.53 | 1.017 |
| qwen3.8-27B | **0** | 14.22 | 14.25 | 1.002 |

That sweep was taken on `build-hip-final`, a neighbouring build whose gates do **not** reproduce the
documented values, so its absolute numbers are not the front page's and the three-percent rows are at
six pairs, not ten. They are reported as a direction, not as figures; only the 1.5B row has been
repeated on the headline build at ten pairs.

The two models that gain nothing are the two where no stream is ever launched. The MoE is the
interesting case: the pass identifies **70** fork points and launches **none**, and the reason is in the
source, not in the graph.

Each candidate region is checked by `ggml_cuda_concurrent_event::is_valid()` (`common.cuh:1305` in the
tree this build comes from), which walks the branches' write ranges and rejects the region if two
branches that would run on different streams write to overlapping memory, which the graph allocator can
easily arrange by reusing a buffer. The launch gate, `ggml-cuda.cu:4250`, then reads:

```c
for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
}
```

**It is all or nothing.** One rejected region anywhere in the graph disables every other region in that
graph. That looked like the explanation for the MoE, and it is not: see below, where changing the gate
turns out not to help, which means the regions are not being lost to one unlucky neighbour.

Since the overlap comes from the graph allocator reusing buffers, the allocation knobs are the obvious
thing to try from outside, and they do not help (`moe-forks-not-launched.txt`):

| condition | phase | forks identified | streams launched |
|---|---|---|---|
| default | decode | 90 | **0** |
| default | prefill | 70 | **0** |
| `GGML_CUDA_POOL_NOREUSE=1` | decode | 90 | **0** |
| `GGML_CUDA_POOL_NOREUSE=1` | prefill | 70 | **0** |
| `GGML_CUDA_NO_POOL=1` | decode | 90 | **0** |
| `GGML_CUDA_NO_POOL=1` | prefill | 70 | **0** |

That is the expected answer instead of a surprising one: the pool those knobs control holds temporary
allocations, while the overlapping writes come from the graph allocator's reuse of **tensor** buffers,
which no environment variable reaches. Prefill is no better than decode. So nothing available from
outside unlocks the MoE.

### The per-region gate was built and tested, and it does not unlock it either

The obvious change is to drop only the unsafe regions instead of all of them, so the same tree was
built with the gate made per-region, ten lines in `ggml_cuda_graph_evaluate_and_capture`
(`per-region-streams.txt`). **The MoE still launches nothing.**

| | base | per-region gate |
|---|---|---|
| qwen3.6-35B-A3B MoE | 0 | **0** |
| qwen3.8-27B | 0 | **0** |
| qwen2.5-1.5B | 56 | 56 |

That measurement rules the explanation out. If the MoE's 70 regions were being lost to one
unlucky neighbour, dropping only that neighbour would have let the other 69 launch. None launches, so
**every region is individually rejected**, and `is_valid()` is failing on all of them and not on
one. Why it fails was measured on 25 September and it is the same collision in every region: the
allocator assigns `Kcur` and `Vcur` the same buffer, so two branches would write the same bytes
([`logs/moe-stream-aliasing-2026-09-25/`](../moe-stream-aliasing-2026-09-25/)).
The 1.5B is untouched at 56 either way, so the change is a no-op where the regions are already valid.

Decode confirms it, six interleaved pairs each with both arms on `GGML_CUDA_GRAPH_OPT=1`: the MoE reads
70.11 against 69.81 and the 1.5B 211.75 against 211.82, ranges overlapping in both. Nothing moves.

The count of invalid regions was not taken directly, which would need a debug print and another build;
what is measured is that per-region gating changes nothing here, and that is enough to rule the change
out for this board. Whatever makes
the MoE's branches overlap is systematic across the graph rather than incidental to one region, which
points at how the allocator packs the expert branches, not at the gate. The patch is therefore **not
shipped**: it builds cleanly and is a
no-op on every model here, so it does not earn a place in a patch set whose premise is that each entry
makes this board work better.

It is worth saying what the change still is, separately from whether it helps here. The all-or-nothing
gate is a real weakness on its own terms: a graph with one unsafe region and ten safe ones loses all
eleven. No model on this board exercises that case, because the MoE's regions are all unsafe and the
1.5B's are all safe, so there is nothing to measure and nothing to claim. The diff is kept in
`per-region-streams.txt`, not in `patches/`.

The lesson for the page above it: the all-or-nothing gate was visible in the source and looked
sufficient, and it was the wrong explanation. Reading a plausible mechanism out of the code is not the
same as measuring that it is the operative one.

## It does not change the arithmetic

The pass **reorders graph nodes** to interleave the branches, which is exactly the kind of change that
moves an accumulation order, so the documented gates were re-run with it on
(`gates-headline-build.txt`):

| gate | documented | off | `GGML_CUDA_GRAPH_OPT=1` |
|---|---|---|---|
| 1.5B, c4096, 8 chunks | 8.9274 | 8.9274 | **8.9274** |
| 8B q8_0, c2048, 2 chunks | 9.1125 | 9.1125 | **9.1125** |

**Identical to four decimals on both.** The branches it interleaves are independent, so nothing that is
summed together is reordered.

## One interaction to watch for

The pass returns early unless CUDA graphs are in use, so **`GGML_CUDA_GRAPH_OPT=1` does nothing
wherever graph capture is off**. That includes `GGML_CUDA_DISABLE_GRAPHS=1`, which this repository's
known-issues table recommends as the workaround for graph instantiation failing at very deep context on
the 14B models. A 14B run deep enough to need that workaround therefore gives up this option's roughly
three percent as well, and the two cannot be combined.

## What this does not say

It is one option on one board, and nothing here explains why it is off by default upstream; that may be
a judgement about other hardware, where filling the machine with one kernel is easier than it is with
sixteen compute units. No claim is made about other GPUs.

The gain has not been traced to overlap **inside** ROCm the way it was for Vulkan. The Vulkan number
above is a GPU-span measurement; the ROCm number is end-to-end throughput. That they point the same way
on a board where small kernels cannot fill the machine is consistent, and it is not proof.

The front page has since been restated from a campaign and not from this A/B
([`logs/campaign-graphopt-2026-09-24/`](../campaign-graphopt-2026-09-24/)), which is the right
instrument for it. Worth noting why the two disagree on the absolute numbers: this A/B's off-arm reads
183 where the campaign reads 196.7 for the same model and build, because the campaign passes `-mmp 0`
and `HSA_ENABLE_SDMA=0` and this A/B does not. The **ratio** is what carried over, 1.098 here against
1.083 pooled over eighteen samples.

Whether this interacts with the thirteen patches, or would be worth more or less on a stock build, is
partly answered: on stock `build-hip-f44` the same A/B gives 112.7 against 120.6, 1.070, so the option
helps there too and helps the patched build slightly more (`ab-1.5b-stock.txt`).

## Files

`ab-1.5b-headline-build.txt` is the ten interleaved pairs on `build-hip-pkf16`, decode and prefill.
`ab-1.5b-stock.txt` is the same on the unpatched `build-hip-f44`. `ab-six-models.txt` is the six-model
sweep on `build-hip-final`. `gates-headline-build.txt` and `gates-other-build.txt` are the perplexity
gates with the option off and on. `vulkan-overlap.txt` is the ggml-vulkan serial-against-concurrent
comparison. `moe-forks-not-launched.txt` is the MoE fork/launch probe under the allocation knobs.
`per-region-streams.txt` is the per-region gate experiment: launch counts, decode pairs, and the diff
that was built, measured and reverted. `graph-capture-check.txt` is `llama-bench -v` showing that all
four prefill and decode cases capture a graph.
Scripts: [`scripts/graph_opt_ab.sh`](../../scripts/graph_opt_ab.sh),
[`scripts/graph_opt_per_region_test.sh`](../../scripts/graph_opt_per_region_test.sh),
[`scripts/graph_opt_moe_probe.sh`](../../scripts/graph_opt_moe_probe.sh),
[`scripts/graph_opt_models.sh`](../../scripts/graph_opt_models.sh),
[`scripts/graph_opt_gates.sh`](../../scripts/graph_opt_gates.sh),
[`scripts/vulkan_overlap.sh`](../../scripts/vulkan_overlap.sh).
