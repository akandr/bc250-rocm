# Why the MoE and the 27B launch no concurrent streams: K and V share one buffer, 2026-09-25

## What was open

`GGML_CUDA_GRAPH_OPT=1` is worth about 3 percent on the 8B and the two 14Bs and nothing at all on the
mixture-of-experts model and the 27B, where the pass launches no streams
([`logs/graph-opt-2026-09-24/`](../graph-opt-2026-09-24/)). That log established that every region is
rejected, not one unlucky one, since making the gate per-region changed nothing, and left the
reason open: "whatever makes the MoE's branches overlap is systematic across the graph".

This is the reason.

## Result

`ggml_cuda_concurrent_event::is_valid()` was instrumented to print a verdict per region and, when it
rejects one, both colliding address ranges (`instrumentation.diff`). Same build, same boot, one
`llama-bench -p 0 -n 8 -r 1 -v` per model (`per_model.txt`):

| model | regions | rejected | launches | what collides |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 56 | 0 | **56** | nothing |
| qwen3-8B Q8_0 | 72 | 0 | **72** | nothing |
| qwen3.8-27B UD-IQ3_XXS | 32 | **32** | 0 | `Kcur` against `Vcur`, 4096 bytes |
| qwen3.6-35B-A3B MoE IQ2_M | 20 | **20** | 0 | `Kcur` against `Vcur`, 2048 bytes |

Every rejection on both affected models is the same collision, and the two ranges are not merely
overlapping, they are **identical**:

    OVERLAP Kcur-35 (view) [139988196967552,139988196969600) size=2048 stream=3
        vs  Vcur-35 (view) [139988196967552,139988196969600) size=2048 stream=2

Across the MoE's twenty rejections there are two distinct addresses in total, one per graph
evaluation, and in each the K tensor and the V tensor of the same layer have the same start and the
same end (`moe_aliasing.txt`).

## Which tensors actually share the address, and a naming trap

Dumping every tensor of a rejected region with its address, size and parent (`region_evidence.txt`)
gives the collision exactly. In region `join=node_3300`:

| stream | tensor | address | bytes |
|---|---|---|---|
| 2 | `Vcur-35`, and its view | `0x7feaf820c480` | 2048 |
| 3 | `norm-35`, `Kcur_normed-35`, `Kcur-35` and its view | `0x7feaf820c480` | 2048 |
| 3 | `Kcur-35` (reshaped), a second tensor of that name | `0x7feaf820cc80` | 2048 |

The V branch and the K branch hold the same 2048 bytes at `0x7feaf820c480`. That is the write-write
hazard the check refuses, and it is ordinary allocator reuse: ggml-alloc's own trace puts `Vcur-35` and
`norm-35` both at chunk offset 50304, which is that address, because under sequential execution their
lifetimes do not overlap.

**The names mislead, and they misled this investigation.** ggml reuses a name along a chain, so the
allocator logs `Kcur-35 -> offset 52352` while the tensor that ends up called `Kcur-35` in the K branch
was allocated earlier as `norm-35` at offset 50304. Reading the allocation log by name alone says K and
V never share a slot, which is how an intermediate revision of this page came to walk the finding back
before the region dump settled it. Two distinct tensors carry the name `Kcur-35` in this one region.

## What it means

The graph allocator gives a K-branch tensor and a V-branch tensor the same scratch buffer. Under
sequential execution that is correct, since their lifetimes do not overlap. The concurrency pass then puts them on different
branches, where they would be written at the same time, and `is_valid()` refuses.

So the check is not being conservative and is not misreading views: it is declining a real
write-write hazard that the allocation created. Three consequences follow.

A per-region gate cannot help, so the September experiment saw no change. Every region in
these graphs contains the same K and V pair, so dropping the invalid ones drops all of them.

The fix is not in the validity check. It would have to be in allocation: either the branches are
decided before buffers are assigned, or the allocator is told which nodes will run concurrently. The
pass already interleaves nodes "to extend lifetimes so that ggml graph doesn't recycle them", and on
these two models that is not enough to separate K from V.

Nothing here is specific to gfx1013 or to this board. The check, the allocator and the pass are all
architecture-independent.

## How the two branches come to share a block

Enabling ggml-alloc's own tracing gives the sequence exactly (`alloc_sequence.txt`). In the MoE's
decode graph, layer 35:

    allocating Vcur-35 (2048 bytes) - offset 50304
    allocating Kcur-35 (2048 bytes) - offset 52352
    freeing    Vcur-35              at offset 50304     <- V consumed into the cache by SET_ROWS
    allocating norm-35  (2048 bytes) - offset 50304     <- takes the block V just freed
    reusing parent norm-35        for Kcur_normed-35
    reusing parent Kcur_normed-35 for Kcur-35           <- so this Kcur sits at 50304

`Vcur` is freed the moment `SET_ROWS` has copied it into the V cache, and `norm-35` is allocated into
that block immediately after. The K branch then reuses `norm-35` in place, through `Kcur_normed-35` to
`Kcur-35`, so the tensor the K branch finally writes occupies the bytes the V branch was using. Run in
order that is correct, because V really is dead by then. Run concurrently it is a write-after-read
hazard, and the validity check is detecting exactly that.

**The models that work are not safe by design, only by luck of the free list.** The 8B does the same
thing in the same order, freeing `Vcur-35` and then allocating `norm-35` a few nodes later. The
allocator's best-fit search hands it a different block, 33792 instead of V's 50176. Nothing
about the MoE's graph causes the collision; the difference is which free block happened to fit best at
that moment. A change in graph shape could produce it on any model.

That also locates the fix. `ggml_backend_graph_optimize`, which identifies the branches, runs inside
`ggml_backend_sched_split_graph` before `ggml_gallocr_alloc_graph`, so the information needed is
available before the buffers are assigned: the allocator would have to be told not to recycle a block
across node ranges that will run on different streams. No fix was attempted here.

## What is not established

Whether the same is true of the 27B in detail. Its collision is the same shape, `Kcur` against `Vcur`
at 4096 bytes, but only the MoE's allocation sequence was traced.

`dependent_srcs`, the other half of `is_valid()`, never fired once on any model.

## A correction, and an instrument that broke what it measured

An earlier note in this work guessed that `dependent_srcs` was the hidden cause. It is not; it never
fires.

A second guess was that the range check was wrong because it resolves a view to its parent,
`t = tensor->view_src ? tensor->view_src : tensor`, which would make every view into one KV cache look
like the whole cache. That was patched to use each tensor's own range, built and run: **20 rejections
before, 20 after**. The hypothesis was wrong and the collision is real. Both builds recompiled 143 HIP
objects, so neither result is a stale-build artefact.

Then, to print the name of the other tensor, `write_ranges` was changed from `pair<int64_t,int64_t>`
to `tuple<int64_t,int64_t,const char *>`. That silently changed the ordering `std::lower_bound` uses:
for two exactly equal ranges the comparison falls through to the name pointer, and because
`tensor->name` sits below the `""` literal in memory, the search stepped past the equal element. The
overlap stopped being detected. The run reported 0 rejections and 40 launches, which looks exactly
like a fix and is a broken check.

The numbers in this page come from a third build whose comparator sorts and searches on the range
only, and which reproduces the original 20 rejections and 0 launches. The detour is recorded because
the failure mode is the one this repository keeps meeting: the instrument changed the thing it was
measuring, and the wrong answer was the encouraging one. It also happens to be evidence, since the
only case that tie-break can hide is an exactly equal range.

## Reproducing

Apply `instrumentation.diff` to `ggml/src/ggml-cuda/common.cuh`, rebuild, and run any model with
`GGML_CUDA_GRAPH_OPT=1` and `-v`. The verdict lines are one per region; the `OVERLAP` lines carry both
ranges. A header change rebuilds about 143 HIP objects, roughly half an hour on this board, so build
first and measure afterwards instead of during.
