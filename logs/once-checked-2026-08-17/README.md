# Re-measuring the once-checked capability claims, 2026-08-17

The "Other things that work" and "ROCm-only capabilities" sections of the
top-level README each rest on a single observation. This is those observations
taken again, on the production configuration, and captured rather than left in
a transcript.

| file | claim | measured |
|---|---|---|
| `dgemm.log` | FP64 DGEMM at N=2048 | 456.3 GFLOP/s, no wrong results, about 95 percent of the 480 GFLOP/s FP64 rate peak |
| `torch_sweep.log` | PyTorch matmul agrees with a CPU reference | relative error 1.5e-06 at N=1024 to 5.9e-06 at N=8192 |
| `retrieval.log` | cosine-similarity search over a million 384-dimensional embeddings | 928 queries per second, agreeing with the CPU reference to 2.1e-07 |
| `server.log`, `completion.json` | `llama-server` over HTTP | `/health` answered in 8 seconds, a coherent completion at 114.6 tokens/s by the server's own timings |
| `mandelbrot.log` | FP64 Mandelbrot render | writes its 2048x1536 image; the image itself is not kept here |

Two corrections came out of this.

The README previously gave the PyTorch matmul relative error as 4.3e-07 "over 30
iterations". No probe in this repository reproduces that: the CPU-reference
sweep gives figures an order of magnitude larger, and the sustained same-size
run compares against its own first result and so reports exactly zero. The
figures above replace it.

The retrieval example was quoted at "about 960 queries per second" and measures
928 here. The difference is small and the older figure may have come from a
different batch size, but only the measured one is now stated.

## The FP64 stencil, and why its numbers need care

`jacobi_builds.txt` and `jacobi_convergence.txt` cover the Jacobi stencil claim,
which needed three corrections.

**Build flags dominate it.** The build line documented in the probe's own source
carries no optimisation flag. That build runs 6.85 ms per sweep at 2.4 GFLOP/s.
The same source with `-O2` runs 0.19 ms per sweep at 87.7 GFLOP/s, a 36-fold
difference, with identical results. Neither is wrong, but a reader taking the
first as an FP64 capability figure would be badly misled.

**Its "effective GB/s" is not memory throughput.** The optimised build reports
877 GB/s, above the board's measured 432 GB/s DRAM ceiling. The figure counts
logical accesses, and a five-point stencil re-reads its neighbours mostly from
cache, so the two are not comparable.

**Its convergence check does not pass, and that is expected.** At two thousand
sweeps the error against the analytic solution is 0.95. That is the method, not
the hardware: the error falls monotonically with sweep count (0.95, 0.87, 0.65
at 2 thousand, 20 thousand, 200 thousand), and Jacobi on a 2048-wide grid needs
on the order of the grid width squared to converge, since information moves one
cell per sweep.

A note on testing this: the shipped binary hardcodes 2000 iterations and ignores
any command-line argument, so a first attempt to vary the count produced three
identical results and looked like a defect. The convergence figures above come
from a rebuild that reads the argument.

## The server figures

An earlier revision of the top-level README recorded `/health` in 12 seconds and
97.9 tokens/s from a single unlogged run. Re-measured here: 8 seconds and 114.6
tokens/s, the latter consistent with the 113 to 119 the command-line tools give
for the same model on the same configuration. The completion text is the same
one quoted in the README, so the earlier run was real; only its numbers were
never captured.
