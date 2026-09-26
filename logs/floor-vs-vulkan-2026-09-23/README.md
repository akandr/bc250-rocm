# Vulkan's dispatch floor, and what it does to the explanation of the MoE's decode gap, 2026-09-23

[`logs/dispatch-floor-2026-09-22/`](../dispatch-floor-2026-09-22/) measured ROCm's dispatch floor and
concluded that the mixture-of-experts model's decode deficit **is** that floor. It also said, in its own
last paragraph, what would be needed to test that:

> Vulkan's own dispatch floor was not measured, so this does not say ROCm's is the larger of the two;
> it says ROCm's is large enough to be the whole difference.

This measures it. **The conclusion does not hold up.** Vulkan's floor is the larger of the two, Vulkan
issues more dispatches per token than ROCm, not fewer, and ROCm's entire non-kernel time is
smaller than its own nominal floor. The MoE's decode gap is not ROCm's dispatch floor.

## How both were measured

[`scripts/dispatch_floor_ggml.c`](../../scripts/dispatch_floor_ggml.c) times a graph of 256
`ggml_sqrt` nodes through ggml, so each backend dispatches down its own real path: ggml-cuda onto a
stream or a captured HIP graph, ggml-vulkan recording into command buffers. At the smallest tensor,
1024 floats, a node moves 8 KiB and its own work is negligible, so the per-node time there is the
floor. Two graph shapes are built:

- **chain**, 256 nodes each consuming the one before, so no two can overlap;
- **fan**, 256 nodes on 256 independent inputs, so nothing forbids overlapping them.

Both backends come from the **same stock checkout**, llama.cpp `bfdc321`, ggml 0.24.0, Release,
`gfx1013`, built into `build-hip-f44` and `build-vk`; the working tree has no patches applied. The only
difference between the two arms is the backend. Ten interleaved pairs under `flock` on an otherwise
idle board, `scripts/floor_vs_vulkan.sh`.

## The floors

| arm | n | chain, median | range | fan, median |
|---|---|---|---|---|
| **ROCm**, HIP graphs on | 10 | **2.536 us** | 2.532 to 2.542 | 2.680 us |
| **ROCm**, HIP graphs off | 10 | **2.511 us** | 2.502 to 3.226 | 2.644 us |
| **Vulkan** | 10 | **3.181 us** | 3.175 to 3.202 | 5.452 us |

**ROCm's floor is the smaller of the two, by 25 percent.** ROCm is faster in 10 of 10 pairs and the
ranges do not come close to touching: ROCm's worst is 2.542 microseconds, Vulkan's best is 3.175.

Tuning Vulkan's submission batching does not close it (`vk-submit-sweep.txt`). Its default is already
near its best; `GGML_VK_MAX_NODES_PER_SUBMIT=64` gives 3.043 microseconds, still above every ROCm
reading, and one node per submit costs 6.4.

The graphs-off row's range is worth a second look, because 2.502 to 3.226 is not scatter around a
median. It is two states, and chasing it is
[`logs/dispatch-bimodal-2026-09-23/`](../dispatch-bimodal-2026-09-23/): with graph capture off about
three processes in ten run every dispatch 25 percent slower for the life of the process, which is why
that row is wide and the graphs-on row is not. It does not affect the comparison here, since ROCm's
**slow** mode at 3.148 is still below Vulkan's best reading of 3.175, but it is the reason to quote the
graphs-on row as ROCm's floor.

HIP graphs look again like they make no difference, 2.536 against 2.511 on the medians, which is how
this was first read. Against the **mean** of what a process gets they are worth 6 percent, for the
reason in the paragraph above; the medians agree only because the median of a distribution that is 29
percent slow still sits in the fast mode.

These numbers are larger than the 1.78 to 2.03 microseconds the bare-HIP program measured because they
include ggml's own per-node host work. How that host cost splits between the two backends cannot be
separated from this measurement, so the comparison is of the whole per-node cost in each backend's real
path, which is the quantity that matters for llama.cpp.

## Dispatches per token, both counted

The earlier arithmetic divided **both** backends' token times by **ROCm's** 1298 dispatches. Vulkan's
own count had never been taken. `ggml-vulkan` will report it: `GGML_VK_PERF_LOGGER=1` prints every
dispatch of every graph, and the MoE's steady-state decode graph is 1423 of them
(`vk-dispatch-count.txt`, `vk-perf-last-block.txt`; the first two graphs are 1483, the following seven
all 1423). ROCm's 1297.8 is 83060 dispatches over 64 tokens from
[`logs/moe-decode-2026-09-20/`](../moe-decode-2026-09-20/).

| | dispatches per token | token | per dispatch | floor |
|---|---|---|---|---|
| ROCm | 1298 | 13.93 ms | 10.73 us | **2.54 us** |
| Vulkan | **1423** | 11.54 ms | **8.11 us** | **3.18 us** |

**Vulkan issues 125 more dispatches per token than ROCm and pays more for each one, and still finishes
the token 2.4 ms sooner.** Whatever ROCm is losing, it is not losing it at the dispatch.

## Why the old arithmetic could not have been right

There is a second way to see it, entirely inside the earlier measurements. The kernel trace that
produced the 1298 also recorded 775.832 ms of kernel time over 64 tokens, which is 12.12 ms of a 13.93
ms token. **Only 1.81 ms of ROCm's token is outside a kernel at all.** The nominal floor, 1298
dispatches at 2.54 microseconds, is 3.29 ms. A cost larger than all the non-kernel time there is cannot
be sitting on top of the kernels; most of it is already hidden behind them. The earlier page added
1298 × 1.78 us = 2.3 ms to the account as though it were not.

## The obvious explanation, and why this does not settle it

If Vulkan can dispatch more and pay more per dispatch yet still win, an obvious guess is that it
overlaps independent work while a single HIP stream serializes everything. The **fan** arm was built to
test that:

| | chain | fan | fan / chain |
|---|---|---|---|
| ROCm | 2.536 us | 2.680 us | 1.057 |
| Vulkan | 3.181 us | 5.452 us | 1.714 |

**Neither backend goes faster when the work is independent; both go slower**, and Vulkan much more so.

That is a real result about this shape, but it is **not** a clean test of overlap, and an earlier
version of this page called the guess refuted on the strength of it. It is not. Running the same
program under ggml-vulkan's own perf logger (`vk-logger-calibration.txt`) shows the fan arm's per-node
**GPU** time rising with it, 2.87 to 8.03 microseconds at the smallest tensor, so giving each node its
own input buffer changed what the kernels cost as well as whether they could overlap. An arm that moves
both cannot separate them. What the fan arm shows is that independence alone does not buy wall-clock
here; it does not show that the backends cannot overlap.

The same file is worth keeping for a second reason: it calibrates the perf logger, which the dispatch
counts above depend on. Across five tensor sizes in the chain arm the logger's per-node GPU time is
0.79 to 0.92 of the wall-clock per-node cost, below it at every size and tracking it, which is what a
sound kernel-timer should do when the rest is host work. The one caveat when reading that file is that
the logger reports a **mean over every graph** while this program's own figure is a **best of five**,
so the two are not directly comparable and small gaps between them mean nothing.

## What is now open

The MoE's decode deficit is 2.4 ms a token and this page says where it is **not**. It is not the
dispatch floor, not the dispatch count, and not overlap.

Since 87 percent of ROCm's token is inside kernels, the remaining room is mostly there, which sat
awkwardly against [`logs/moe-decode-2026-09-20/`](../moe-decode-2026-09-20/), where replaying the
graph's shapes one at a time was read as putting ROCm at or ahead of Vulkan on every one of them.
**That is now resolved and it is the replay that gives way**
([`logs/moe-kernels-reweighted-2026-09-23/`](../moe-kernels-reweighted-2026-09-23/)): ROCm is slower on
33 of the 63 shapes, the margin in its sum comes from a single artifact row measured 31 times its cost
in the real graph, and split by weight type ROCm is slower on 9 of 10 quantised matmuls and faster on 3
of 3 f32 ones. A floor on the size needs no cross-instrument comparison at all: ROCm spends 12.12 ms a
token inside kernels where Vulkan's entire token is 11.54 ms, so at least 0.58 ms of the deficit is
kernel time. Still open: the rest of it. One difference not examined here is that
a per-shape replay runs each kernel in isolation, and a decode graph runs 1298 of them back to back
against a cache and a clock that the previous kernel just left in some state; whether that accounts for
anything has not been measured, and it is a direction to look instead of a finding.

Two limits on this page itself. The floor was measured with one elementwise op at one small size, not
across the shapes a real graph contains, so it is a floor and not a model of anything larger. And the
Vulkan dispatch count came from a run with the perf logger on, which serializes the queue and drops the
model from 86.7 to 54.5 tokens a second; the **counts** are unaffected by that, the **timings** in that
file are not, and none of them are used here.

## Files

`floor-interleaved.txt` is the ten pairs. `floor-hip.txt`, `floor-hip-nographs.txt` and
`floor-vulkan.txt` are single full runs with the size sweep and the fits. `vk-submit-sweep.txt` is the
submission-batching sweep. `vk-dispatch-count.txt` is the parsed per-op dispatch count for one MoE
decode graph and `vk-perf-last-block.txt` the raw block it came from. `env.txt` is the board, the
checkout and the temperatures either side.
