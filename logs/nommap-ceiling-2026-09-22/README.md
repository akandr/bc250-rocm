# What `--no-mmap` costs in usable context, 2026-09-22

[Limits](../../README.md#limits) has said "loading with `--no-mmap` costs usable context" since
September without a number beside it. This puts one on it, for one model, because the absence of a
number cost a measurement.

The qwen3-8B Q8_0 aborted with `ggml-cuda.cu:111: ROCm error` at a primed depth of 16128 while a
measurement was being reproduced from
[`logs/decode-variance-state-2026-08-21/`](../decode-variance-state-2026-08-21/), which ran that exact
model at that exact depth in August and got 17.67 t/s. That looks like a regression, and ten patches
have landed since, so it was worth bisecting instead of assuming.

## Where it starts failing, and what it is not

[`scripts/nommap_ceiling.sh`](../../scripts/nommap_ceiling.sh), `-p 0 -n 8`, one invocation per point,
the board otherwise idle and the lock held:

| depth | ROCm, `-mmp 0` | |
|---|---|---|
| 4096 | 34.27 | |
| 8192 | 32.79 | |
| 12288 | 30.91 | |
| 14336 | 30.32 | |
| 15360 | 29.76 | |
| 16128 | **aborts** | `ROCm error` |
| 16384 | **aborts** | `ROCm error` |

Three controls say what it is not:

| control | depth | result |
|---|---|---|
| `GGML_RDNA1_PKF16=0`, the prefill GEMM disabled | 15360 | 26.30, runs |
| `GGML_RDNA1_PKF16=0` | 16128 | aborts, so not that patch |
| `-fa off` | 16128 | aborts, so not flash attention |
| **Vulkan**, same model, same flags | 16128 | **28.30, runs** |
| **ROCm with mmap at llama-bench's default** | 16128 | **27.83, runs** |
| **ROCm with mmap at llama-bench's default** | 16384 | **27.30, runs** |
| ROCm, `-mmp 0`, repeated under the lock | 16384 | aborts |

No kernel fault line was logged at any point, before or after. What the kernel does log, thousands of
times across the failing runs, is the line the ceilings page names:

    amdgpu: SVM mapping failed, exceeds resident system memory limit

## It is the documented limit, reached sooner

Not a patch, and not the board: Vulkan runs the same point on the same flags, and ROCm runs it too once
mmap is left alone. The kernel line names the mechanism, and it is the one
[`logs/fedora44-ceilings-2026-09-16/`](../fedora44-ceilings-2026-09-16/) already describes: the KFD
resident-memory limit, 63/64 of RAM minus 1.5 GiB, counting everything KFD has registered. `-mmp 0`
reads the whole 8.24 GiB model into memory that cannot be reclaimed instead of mapping it from the
file, and on a board where the GPU allocates from the same pool that comes out of the context budget.
**For this model it moves the ceiling from at least 16128 down to between 15360 and 16128.** Vulkan is
unaffected because it does not go through that limit at all.

One thing notable in passing, since it is a fix from this repository working. That page records
the stock runtimes segfaulting on this failure instead of reporting it, and three null checks across
ROCr and HIP being needed to turn it into an error ([step 6](../../README.md#6-install-both-ahead-of-the-system-libraries),
[`logs/rocr-queue-scratch-2026-09-18/`](../rocr-queue-scratch-2026-09-18/)). Here it arrives as a
reported `ROCm error` that llama.cpp aborts on cleanly, which is what those checks were for.

The ceilings in [`logs/fedora44-ceilings-2026-09-16/`](../fedora44-ceilings-2026-09-16/) were measured
with mmap on, as that page says, and they stand: this model generates at 16384 there. Every throughput
campaign in this repository passes `-mmp 0`, for the good reason that it takes the page cache out of
the comparison, so the two are describing different usable depths. The front page now says so rather
than leaving a reader to find out the way this did.

## Why it is written up at all

The failure was mine twice before it was anything. The first attempt ran two `llama-bench` invocations
at once without taking the lock, which gave `failed to load model`, a different error with a different
cause. The second hid stderr, so a model that never loaded read as a measurement that returned nothing.
Both are discipline this repository already documents. What survived the tidying up is the number, and
the number is worth having.

## Files

`log` is the bisection and the three controls. The mmap comparison at 16128 and 16384 is `mmap-grid.txt`.
