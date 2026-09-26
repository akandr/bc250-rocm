# The MoE per-op replay, reweighted, and the artifact that reversed it, 2026-09-23

[`logs/floor-vs-vulkan-2026-09-23/`](../floor-vs-vulkan-2026-09-23/) left a question it could not
answer. The MoE's decode deficit is 2.39 ms a token, it is not the dispatch floor, and 87 percent of
ROCm's token is inside kernels, so the room is mostly there. But
[`logs/moe-decode-2026-09-20/`](../moe-decode-2026-09-20/) said the opposite about the kernels:

> Replaying its one-token graph op by op puts ROCm at or ahead of Vulkan on every shape it contains,
> so no kernel is slow.

Both are in this repository and they cannot both be right. This reconciles them, from files already
here and with no new measurement on the board:
[`scripts/reweight_moe_ops.py`](../../scripts/reweight_moe_ops.py).

**The replay's conclusion is wrong, for two independent reasons, and the kernels are back.**

## One row of the Vulkan replay is an artifact worth more than the whole margin

`test-backend-ops` measures `MUL_MAT shared_expert_gate-0`, a 1x1 output from a 2048-long dot product,
at **131.71 us on Vulkan** against 4.09 on ROCm. That single row is 127.6 microseconds of the
88.4-microsecond margin the published sum rests on.

It is not what the graph does. ggml-vulkan's own perf logger, on the real decode graph, dispatches that
shape as `MUL_MAT_VEC f32 m=1 n=1 k=2048` at **4.258 us**, forty times a token. The replay reading is
**31 times** the real cost: replaying the op standalone routes it somewhere the graph never goes.

Correcting that one row alone reverses the unweighted result:

| | ROCm | Vulkan | ROCm / Vulkan |
|---|---|---|---|
| as published | 1453.9 us | 1542.2 us | **0.943** (ROCm ahead) |
| artifact row corrected | 1453.9 us | 1414.8 us | **1.028** (ROCm behind) |

## "At or ahead on every shape" was never true

That phrase can be checked directly against the same two logs, and it does not hold however the sum
comes out. **ROCm is faster on 30 of the 63 shapes and slower on 33**, with a median per-shape ratio of
**1.024**. The typical shape in this graph is a couple of percent slower on ROCm, not faster. Many of
those margins are small enough to be noise on a 3-microsecond kernel, which is the point: the evidence
never supported "every shape", in either direction.

## Why a weighted figure is not available, which is a finding in itself

The sum's other flaw is that it weights every distinct shape equally, so the output head, which runs
once a token, counts the same as a projection that runs a hundred times. The obvious fix is to weight
each shape by how often it runs, and an earlier version of this page did that, using the occurrence
counts from ggml-vulkan's perf logger, and reported ROCm 6.5 percent slower. **That figure was wrong
and is withdrawn.** It weighted *ROCm's* isolated per-shape times by *Vulkan's* dispatch counts, and
the two backends do not issue the same dispatches for this graph.

They do not, because they do not fuse the same things. ROCm's kernel trace shows **437.7 matvec
dispatches a token** against Vulkan's **511**, and the clearest case is the expert gate/up pair in
iq2_xxs:

| | dispatches per token | each | what one dispatch does |
|---|---|---|---|
| ROCm, `mul_mat_vec_f32_rdna1<iq2_xxs, has_fusion, ...>` | 39.4 | 41.4 us | gate matmul + up matmul + GLU |
| Vulkan, `MUL_MAT_ID_VEC iq2_xxs m=512 n=8 k=2048` | 80.0 | 19.3 us | one matmul |

ROCm runs one kernel where Vulkan runs two, and the ROCm one is doing strictly more: its second template
parameter is `has_fusion`, and `mmvq-rdna1-f32.cu` sets it when a gate or bias is folded in, so that
single dispatch also absorbs the `GLU` that Vulkan issues separately, 80 times a token at 2.96 us.
**The two rows are therefore not comparable as totals and no total is given**, which is exactly the
point: multiplying ROCm's *unfused* replay reading of that shape, 22.38 us, by Vulkan's count of 80
prices 1790 us of ROCm work that the graph never does, and does not account for the GLU either way.

So a cross-backend per-shape weighted total is **not defined** for this graph, and no number of the
form "ROCm's kernels are N percent slower weighted" can be produced from these two instruments. The
instruments do not help either: `kerntrace` reports pure kernel durations while the Vulkan perf logger
reports consecutive timestamp deltas that tile the graph, so its figures carry the gaps as well and the
two are not comparable in aggregate. For the record, the aggregates are ROCm 8.90 ms a token of matvec
kernel time against a Vulkan figure of at most 8.53 ms measured the other way; that comparison is
directional at best and is not used here.

## What the weights are, which needs no weighting at all

The useful pattern does not depend on any of that, because it is the same instrument on both backends,
shape by shape. Splitting the replayed matmuls by what the weights are:

| | shapes | median ROCm / Vulkan | |
|---|---|---|---|
| quantised weights | 10 | **1.078** | ROCm slower on **9 of 10** |
| f32 weights | 3 | **0.820** | ROCm faster on **3 of 3** |

| name | weights | ROCm us | Vulkan us | ratio |
|---|---|---|---|---|
| ffn_moe_down-0 | iq3_xxs | 34.06 | 25.94 | **1.313** |
| z-0 | q5_K | 24.68 | 20.85 | 1.184 |
| attn_output-3 | q5_K | 24.85 | 21.15 | 1.175 |
| node_13 | q5_K | 41.02 | 37.65 | 1.090 |
| ffn_moe_down-34 | iq4_xs | 34.37 | 31.89 | 1.078 |
| ffn_gate-0 | q5_K | 7.53 | 6.99 | 1.077 |
| ffn_moe_gate-0 | iq2_xxs | 22.38 | 20.81 | 1.075 |
| result_output | q4_K | 955.13 | 920.51 | 1.038 |
| linear_attn_out-0 | q6_K | 23.74 | 23.17 | 1.025 |
| ffn_shexp-0 | q6_K | 6.12 | 8.27 | 0.740 |
| shared_expert_gate-0 | f32 | 4.09 | 4.26 | 0.961 |
| node_34 | f32 | 4.23 | 5.16 | 0.820 |
| ffn_moe_logits-0 | f32 | 7.60 | 10.50 | 0.724 |

**ROCm is slower on nine of the ten quantised matmuls and faster on all three f32 ones.** The one
quantised shape it wins is `ffn_shexp-0`, q6_K with k=512. The sign is not scattered, and it is the
same split the rest of this repository already documents: gfx1013 has no integer dot product, so the
quantised matvec emulates dp4a where RADV uses float fma
([`logs/rdna1-iq-matvec-2026-09-20/`](../rdna1-iq-matvec-2026-09-20/), which prices the remaining
directions at about one percent each). So the MoE's kernel deficit is not a new phenomenon needing a
new explanation: it is that gap, now at 1.03 to 1.31, not the 1.7 to 2.0 it started at.

## What that accounts for, and what it does not

How much of the 2390-microsecond deficit that is worth cannot be stated per shape, for the reason above.
What can be stated is the direction and the location: ROCm's kernels **are** slower, systematically and
on the quantised matvecs, and the replay's sum was hiding it behind an artifact.

A lower bound on the size is available, and it needs no cross-instrument comparison at all, only two
numbers that are each measured on their own backend:

| | |
|---|---|
| ROCm, kernel time alone, `kerntrace` over 64 tokens | **12.12 ms** a token |
| Vulkan, the **entire** token, `llama-bench` | **11.54 ms** |

**ROCm spends longer inside kernels than Vulkan takes for the whole token.** So at least 0.58 ms of the
2.39 ms deficit is kernel time, before counting whatever part of Vulkan's own token is not kernel time,
which can only make the kernel gap larger. That is a floor on the kernel contribution, not an estimate
of it, and it answers the question this page opened with: the kernels carry part of the deficit.

The one assumption in it is that a traced run's kernel durations carry over to an untraced one. The
trace is slower end to end, 17.8 ms a token against 13.93, but that is tracing overhead between
kernels instead of inside them, and this repository already leans on the same assumption for the "87
percent
of the token is inside kernels" figure.

The rest is not settled here, and the tempting shortcut does not work. Vulkan's perf logger totals
**13.27 ms** for one decode graph, which looks at first like a contradiction against an 11.54 ms token
and like evidence of something. It is neither. Reading `ggml-vulkan.cpp`, each node's figure is
`timestamps[i] - timestamps[i-1]`, consecutive deltas that **tile** the graph, so the sum is by
construction the GPU span of that graph **in the run that produced it**, and that run is the slow one:
the logger costs Vulkan 86.7 down to 54.5 tokens a second, an 18.35 ms token. 13.27 ms of span inside
an 18.35 ms token is ordinary. It cannot be compared with the 11.54 ms token of a run without
the logger, and **this page does not use the total for anything**.

The per-shape figures survive that, which is why the weighting above is trustworthy: they
agree with the replay to 0.9 percent weighted, on a different run with a different instrument, so
whatever the logger costs the run as a whole it is not distorting individual kernel durations.

That distinction matters because the overlap reading was probed from the other side and did not show
up: given 256 independent nodes instead of a dependent chain, neither backend speeds up and Vulkan slows
down by 1.71 times ([`logs/floor-vs-vulkan-2026-09-23/`](../floor-vs-vulkan-2026-09-23/)). That arm does
not settle it either, since ggml-vulkan's own timings show its kernels got slower in it as well. So
there is evidence both ways and nothing here decides it. Naming the mechanism would be guessing.

Where the remaining 1.8 ms sits is open: ROCm's non-matmul kernels, which are 24 percent of its kernel
time and were not reweighted here because the perf logger prints no shapes for them, and the 1.81 ms of
ROCm's token that is outside any kernel.

## What this does not touch

The prefill result, the fusion result and the floor measurement itself are independent of all of this.
So is the conclusion that the deficit is not the dispatch floor, which rests on Vulkan's floor and
dispatch count, not on any per-op replay.

It is also one model and one graph. Whether other models' replays carry a comparable artifact has not
been checked; the general lesson is that a standalone op replay can route a shape somewhere the graph
does not, and a shape whose replayed cost is dozens of times its cost in the graph will not look
obviously wrong in a sorted table.

That is the **second** way this instrument has misled here, in a different direction.
[`logs/round6-2026-09-19/`](../round6-2026-09-19/) already records the first: a per-op replay runs the
same op thousands of times, so a pool buffer stays warm and dequantised weights are reused across
iterations in a way no real model does, which "flatters any design that caches work across calls".
Taken together: a replay can make an op look faster than the graph will (caching) **and** slower than
the graph will (a different dispatch path). Both were found by checking a replayed number against the
same shape measured inside a real graph, which is the check worth keeping.

## Files

`reweighted.txt` is the script's output. Its inputs are
[`logs/moe-decode-2026-09-20/`](../moe-decode-2026-09-20/)`ops-moe-n1-{hip,vk}.log` and
[`logs/floor-vs-vulkan-2026-09-23/`](../floor-vs-vulkan-2026-09-23/)`vk-perf-last-block.txt`.

**Naming, corrected 25 September 2026.** Two lines above called the instrument `rocprof`. The numbers
came from [`scripts/kerntrace.cpp`](../../scripts/kerntrace.cpp), the LD_PRELOAD roctracer tool, which
is what was on the board; no `rocprof` was installed at the time and none is packaged for Fedora. The
measurements are unaffected, only the name was wrong. A working `rocprof` was built later, in
[`logs/hw-counters-2026-09-25/`](../hw-counters-2026-09-25/).

**The remaining 1.8 ms, answered 25 September 2026.** It is per-dispatch overhead
([`logs/decode-residue-2026-09-25/`](../decode-residue-2026-09-25/)). Measured as the untraced token
minus the kernel time, the 1.5B leaves 0.657 ms over 369 dispatches and the MoE 2.197 ms over 1298:
1.78 and 1.69 microseconds a dispatch, from two models whose dispatch counts differ by 3.5 times. It
is not memory copies, which are one percent of it, and not anything HIP graph capture hides, since
disabling capture moves neither model outside its error bars. The estimate this repository already
carried, 1.78 microseconds a dispatch taken from a separate microbenchmark, is right.

The HIP API timeline this page asked for is still not available, and for a reason worth knowing:
Fedora builds HIP without the profiler handshake as well as the HSA runtime, so `--hip-runtime-trace`
attaches to nothing. It would not have helped anyway. Tracing inflates the gaps between kernels more
than twentyfold on this board, which is exactly the quantity it would have been asked to measure.
