# A process either gets the fast dispatch path or it does not, 2026-09-23

Measuring the dispatch floor for
[`logs/floor-vs-vulkan-2026-09-23/`](../floor-vs-vulkan-2026-09-23/) turned up something that was not
what the run was for. With HIP graph capture **on**, the per-node cost is the same number every time.
With it **off**, it is one of two numbers.

| | processes | in the slow mode | fast mode | slow mode | overall mean |
|---|---|---|---|---|---|
| HIP graphs **on** | 24 | **0 of 24** | 2.535 us (2.529 to 2.543) | none seen | **2.535 us** |
| HIP graphs **off** | 72 | **21 of 72, 29 %** | 2.509 us (2.496 to 2.532) | 3.148 us (2.977 to 3.296) | **2.695 us** |

**About three processes in ten land in a state where every dispatch costs 25 percent more, and stay
there for the life of the process.** Each figure above is one process's best of five blocks of forty
graph executions, so the slow state is not a blip that a best-of escapes: a process that starts slow
finishes slow.

Twenty-four consecutive runs with graphs on produced no slow reading at all. If the two arms drew from
the same distribution that would happen about once in sixteen thousand times.

## It does not happen on Vulkan

The control that says this is about the HIP runtime, not the board, the kernel or the hardware:
the same program, the same chain, the same board, through ggml-vulkan instead (`vulkan-control.txt`
plus the Vulkan arm of [`logs/floor-vs-vulkan-2026-09-23/`](../floor-vs-vulkan-2026-09-23/)).

| | processes | in a slow mode | range | spread |
|---|---|---|---|---|
| Vulkan | **30** | **0 of 30** | 3.170 to 3.207 us | 1.2 % |
| ROCm, graphs off | 72 | 21 of 72 | 2.496 to 3.296 us | 32 % |

**Thirty Vulkan processes, no second mode.** At the 29 percent rate the ROCm arm shows, seeing none in
thirty is about a 1 in 18000 event. Whatever picks the mode is on the HIP side and graph capture hides
it.

## It is not the clock

A lower clock would slow the bandwidth-bound end of the sweep as well as the launch-bound end. It does
not (`clock-or-dispatch.txt`, and the same columns in the earlier runs):

| | per-node floor at 1024 elements | GB/s reached at 1048576 elements |
|---|---|---|
| fast mode | 2.509 us | 378.6 (n = 20) |
| slow mode | 3.148 us | 376.4 (n = 8) |

**The throughput is the same to within half a percent while the floor rises by a quarter.** Whatever
changes, it changes the fixed cost of issuing a dispatch and not the rate at which the machine executes
one. It is also not temperature: slow readings appear at 66 C and fast ones at 75, and the two arms are
interleaved so both see the same thermal history.

## It is not the hardware queue count

`GPU_MAX_HW_QUEUES=1` is the documented knob on this board and was the obvious candidate, so it was
tested against the default, twenty interleaved pairs, graphs off in both arms (`hw-queues.txt`). It does
not remove the bimodality: the forced-single-queue arm still produces slow readings, and it adds
intermediate ones the default arm does not show. **Eliminated.**

## It is not the kernel-argument pool

A fixed per-dispatch cost is the kind of thing a kernel-argument placement decision would move, and
`HIP_FORCE_DEV_KERNARG=1` is the documented knob for it, so it was tested against the default, 15
interleaved pairs, graphs off in both (`kernarg.txt`). It does not separate them: **3 of 15 slow in each
arm**, and the forced arm adds two intermediate readings near 2.70 that the default does not produce.
**Eliminated.**

## It is not address-space randomisation

`setarch -R` disables ASLR for the child, so if the mode were decided by where the address space
happened to land, the two arms would separate. They do not (`aslr.txt`, 14 interleaved pairs, graphs off
in both): **7 of 14 slow with ASLR off against 5 of 14 with it on**. **Eliminated**, and at a size that
means something this time; an earlier ASLR question on this board was called at n=3 and should not have
been.

The same file also answers whether the state persists between processes, which the clustering in the
other runs hints at. In 14 pairs of consecutive processes under the same condition, 1 pair was both
slow, 4 both fast and 9 mixed, where independence at this rate predicts about 3, 5 and 7. **If anything
that is less agreement than chance, so there is no evidence the state carries from one process to the
next**, and the apparent clustering elsewhere is what a 29 percent rate looks like.

## It does not reach a real model, and that withdraws the interesting claim

The obvious thing to say next was that graph capture is therefore worth something, since it removes a
penalty that hits a third of processes. **An earlier version of this page said exactly that, 6.0 percent
on the mean, and it is wrong.** The synthetic result does not transfer.

`scripts/dispatch_mode_end_to_end.sh` runs the same comparison through `llama-bench` instead, one
process per data point on the 1.5B, 16 processes an arm at `-r 3` and 12 more at `-r 8` in case graph
capture was being charged to the first repetition (`end-to-end-r3.txt`, `end-to-end-r8.txt`):

| | processes | mean | range | spread |
|---|---|---|---|---|
| HIP graphs **on** | 28 | 112.60 t/s | 110.87 to 113.09 | 2.0 % |
| HIP graphs **off** | 28 | **113.59 t/s** | 113.17 to 113.97 | **0.7 %** |

**No second mode.** The floor is 12.9 percent of this model's token, so a process in the slow mode
should read about 3 percent low. **Zero of 28 graphs-off processes do**, and that arm is the *tighter*
of the two, spanning 0.7 percent against 2.0.

And graphs-off is **faster in 28 of 28**, by 0.88 percent. So on a real model the ordering is the one
this repository already had: **graph capture buys nothing here and costs about one percent**, which is
what [`logs/dispatch-floor-2026-09-22/`](../dispatch-floor-2026-09-22/) measured with a bare HIP program
and what the campaign with graphs disabled found within 0.4 percent. That claim stands. The correction
offered above it does not, and is withdrawn.

What survives is narrower and is a caveat about instruments, not about the backend: **a
synthetic dispatch benchmark on this board can land in either of two states with graph capture off, and
a microbenchmark that does not check for it will report whichever one it drew.** It is highly
reproducible where it occurs, 21 of 72 processes across four separate runs with six candidates
eliminated and a clean Vulkan control, and it does not occur in llama.cpp. Both halves of that are
measured.

Why a 256-node chain of trivial elementwise kernels shows it and a transformer decode graph does not is
not established. The two differ in almost everything that could matter, including the number of
distinct kernels, the allocation pattern and what the runtime does before the first dispatch, and
nothing here separates them.

## What this does not say

The mechanism is not identified and nothing here names one, though the Vulkan control places it in the
HIP runtime and not in the board or the kernel driver. Eliminated so far: the clock, the
temperature, the hardware queue count, the kernel-argument pool, address-space randomisation, and
persistence between processes.
What has not been looked at includes which queue or doorbell a process is given and anything else the
runtime decides once at initialisation.

The effect is measured on one synthetic program, a chain of 256 trivial elementwise nodes, not on a
model. Whether a 25 percent higher dispatch floor would be visible end to end depends on what share of
a token that floor is, which is 13 to 16 percent on the models that dispatch often
([`logs/fusion-value-2026-09-23/`](../fusion-value-2026-09-23/)).

Two further things were checked instead of assumed. `llama-bench -v` exposes the `GGML_LOG_DEBUG` line
saying whether a graph was captured, which the default log level hides: on the 1.5B and the MoE, in both
prefill and decode, every run reports `CUDA graph warmup complete` and none reports `disabling CUDA
graphs` (`graph-capture-check.txt`). So the benchmarks here do capture graphs, and capture is not
restricted to decode in this build; the compatibility check only refuses `MUL_MAT_ID` nodes that take
the synchronising fallback path. And the end-to-end arm above shows that even with capture off a real
model does not enter the slow mode, so nothing in this repository's numbers is exposed to it either
way.

It is worth holding next to the other unexplained variance on this board, the one where the first one or
two repetitions after a model load read differently from the rest
([`logs/decode-variance-clock-2026-09-22/`](../decode-variance-clock-2026-09-22/)). That one is
within a process and this one is between processes, so they are not the same observation, and no
attempt is made here to join them.

## Files

`graphs-on-vs-off.txt` is 24 interleaved pairs, graphs on against graphs off.
`hw-queues.txt` is 20 interleaved pairs, graphs off in both arms, default against `GPU_MAX_HW_QUEUES=1`.
`clock-or-dispatch.txt` is two runs of 14 reporting the floor and the large-tensor throughput together.
`aslr.txt` is 14 interleaved pairs, graphs off in both arms, default against `setarch -R`.
`vulkan-control.txt` is 20 consecutive Vulkan runs of the same program.
`kernarg.txt` is 15 interleaved pairs, graphs off in both arms, default against `HIP_FORCE_DEV_KERNARG=1`.
`graph-capture-check.txt` is `llama-bench -v` on two models in prefill and decode, showing that all four
capture a graph.
The scripts are [`scripts/dispatch_mode_graphs.sh`](../../scripts/dispatch_mode_graphs.sh),
[`scripts/dispatch_mode_queues.sh`](../../scripts/dispatch_mode_queues.sh),
[`scripts/dispatch_mode_aslr.sh`](../../scripts/dispatch_mode_aslr.sh),
[`scripts/dispatch_mode_vulkan.sh`](../../scripts/dispatch_mode_vulkan.sh),
[`scripts/dispatch_mode_kernarg.sh`](../../scripts/dispatch_mode_kernarg.sh) and
[`scripts/dispatch_mode_probe.sh`](../../scripts/dispatch_mode_probe.sh), all driving
[`scripts/dispatch_floor_ggml.c`](../../scripts/dispatch_floor_ggml.c).
