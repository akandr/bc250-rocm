Streaming-read memory bandwidth, measured from `patches/membw.cpp` as shipped.

This exists because of a gap the figure audit could not see. The 432 GB/s ceiling
is used throughout the write-up to compute what share of available bandwidth each
model's decode uses, and it appeared in no measurement log: only as a reference
inside another log directory's README. The audit script matches figures with two
to four decimal places, so a whole number like 432 is invisible to it, which is
the blind spot its own docstring warns about.

Three runs on the current stack:

| run | best read bandwidth |
|---|---|
| 1 | 432.2 GB/s (402.5 GiB/s) |
| 2 | 431.7 GB/s (402.0 GiB/s) |
| 3 | 431.7 GB/s (402.1 GiB/s) |

The quoted 432 GB/s and 402 GiB/s are confirmed, and the spread across runs is
0.12 percent, so this is one of the steadier measurements here. An earlier set of
three runs in the same session gave 431.6, 431.5 and 432.1, which is the same
answer; the table matches `membw.out` in this directory rather than that first
set, because the shipped log is what a reader can check.

Built with `clang++ -x hip --offload-arch=gfx1013 -O3` from the shipped source,
run with `HSA_ENABLE_SDMA=0`, 2048 MiB buffer.
