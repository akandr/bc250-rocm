# Where a decoded token's time goes: kernel traces of real runs, 2026-09-19

`test-backend-ops` replays one op at a time and its Vulkan timings carry per-op submission overhead, so
summing replays does not say how a real token's time divides. [`scripts/kerntrace.cpp`](../../scripts/kerntrace.cpp)
is a small `LD_PRELOAD` tool on Fedora's `roctracer`: it interposes the first HIP calls llama.cpp makes
(`hipGetDeviceCount`, `hipSetDevice`, `hipMalloc`) to switch HIP_OPS activity tracing on once the runtime is
loaded, and at exit sums device time per kernel. `llama-bench -p 0 -n 64 -r 1` and `-p 512 -n 0 -r 1` on
the front-page build (`trace-*.txt`, `trace-*.bench`).

| run | dispatches | kernel time | first dispatch to last completion | kernel time over that span |
|---|---|---|---|---|
| qwen2.5-1.5B tg64 | 23985 | 310.9 ms | 439.4 ms | 71 % |
| qwen3.8-27B tg64 | 111667 | 4354 ms | 5324 ms | 82 % |

**The last column is not the idle time of an ordinary run, and an earlier version of this page read it
that way.** The tracer slows the run it measures: `trace-*.bench` records what `llama-bench` reported
while being traced, 162.38 t/s on the 1.5B and 12.61 t/s on the 27B, against about 198 and 14.8 for the
same build untraced. That 15 to 18 percent is the same size as the shortfall in the table, so most of
it is the cost of interposing every dispatch, not a gap the GPU would see on its own. Taking the
kernel totals at face value, the 27B's 4354 ms of kernel time for 64 tokens is already 68.0 ms a token
against an untraced token of 67.6 ms, so an untraced decode is close to fully occupied; the tracer
inflates the kernel figures as well, so no exact occupancy can be read off these runs. What the traces
do support is the split by class below, which is a ratio within one run and so survives the overhead.

Kernel time by class, decode:

| class | 1.5B | | 27B | |
|---|---|---|---|---|
| float matvec (patch 5) | 260.6 ms | 83.8 % | 3687 ms | 84.7 % |
| int8 matvec (iq4_xs, iq1_m, q2_K) | | | 321 ms | 7.4 % |
| flash attention (tile + combine) | 23.3 ms | 7.5 % | 16.4 ms | 0.4 % |
| rms_norm | 13.7 ms | 4.4 % | 66.9 ms | 1.5 % |
| rope | 9.6 ms | 3.1 % | 7.2 ms | 0.2 % |
| set_rows / get_rows | 3.6 ms | 1.2 % | 68.7 ms | 1.6 % |
| gated delta net | | | 48.7 ms | 1.1 % |
| everything else | 0.2 ms | | 138 ms | 3.2 % |

So on both models the matrix-vector kernels are 84 to 92 percent of the kernel time, and no other class
reaches 8 percent. Read together with the note above, that places the decode gap against Vulkan in the
matvec kernels themselves, not in the spaces between dispatches: there is little room between
dispatches to recover. Four attempts to close it on the two codebook models are in
[`logs/rdna1-iq-matvec-2026-09-20/`](../rdna1-iq-matvec-2026-09-20/), and none of them did.

The prefill traces (`trace-*-pp.txt`) list every kernel twice, once under its demangled name and once
under the mangled one with identical counts; the MMQ GEMM is the run at 78 percent of the 1.5B's
(doubled) kernel time and flash attention 3 percent, which matches the op-replay picture of prefill. The
double counting was not chased.
