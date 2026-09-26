# What PyTorch and rocBLAS actually deliver here, against the board's own ceilings, 2026-09-24


**Ceiling correction, 25 September.** The ceilings this page divides by, 4.74 TFLOP/s fp32 and 14.40
packed fp16, were measured below the clock cap. Re-measured with the clock verified at 1500 MHz they
are 6.52 and 13.02, and packed fp16 is twice fp32, not three times
([`logs/alu-rates-recheck-2026-09-25/`](../alu-rates-recheck-2026-09-25/)). Every percentage of peak on
this page is therefore against the wrong denominator; the measured rates themselves are unaffected.

PyTorch is the reason to run ROCm on this board instead of Vulkan, which has no equivalent. Until now
this repository only showed that PyTorch is **correct** here
([`logs/torch-train-2026-08-19/`](../torch-train-2026-08-19/),
[`logs/fedora44-validation-2026-09-15/`](../fedora44-validation-2026-09-15/)) and never how fast it is.
This measures it, and rocBLAS underneath it.

Absolute throughput on a part nobody has a reference for says little, so every figure is also given as
a fraction of a ceiling measured on this same silicon, not a vendor number
([`logs/alu-rates-2026-09-19/`](../alu-rates-2026-09-19/)): `v_fma_f32` **4.74 TFLOP/s**,
`v_pk_fma_f16` **14.40 TFLOP/s**, streaming read **432 GB/s**.

Method: buffers allocated once and reused so the caching allocator is outside the measurement, three
untimed warm-up iterations, seven timed ones each bracketed by `torch.cuda.synchronize()`, median
reported with the full spread. Every GEMM is checked against a CPU reference. `HSA_ENABLE_SDMA=0`,
edge 55 C at the start and 64 C at the end, no throttling.

**Run under the source-built PyTorch, not Fedora's package.** `python3-torch-2.9.1-10.fc44` has no
gfx1013 kernels and dumps core on the first matmul; the wheel from
[`scripts/build_pytorch_gfx1013.sh`](../../scripts/build_pytorch_gfx1013.sh) was installed into a venv
that shadows it.

## GEMM: fp32 is at the ceiling, fp16 is not

| dtype | N=2048 | N=4096 | N=8192 | ceiling | best % of peak |
|---|---|---|---|---|---|
| fp32 | 2.99 | 4.55 | **4.59 TFLOP/s** | 4.74 | **96.9 %** |
| fp16 | 4.16 | 4.25 | **4.25 TFLOP/s** | 14.40 | **29.5 %** |
| bf16 | 3.72 | 3.81 | **3.78 TFLOP/s** | 14.40 | 26.2 % |

**fp32 matmul reaches 96.9 percent of what the arithmetic units can do.** There is nothing left in it,
and the figure agrees with this repository's rocBLAS SGEMM measurement to three digits.

**fp16 is not faster than fp32 here. It is slower.** 4.25 against 4.59 TFLOP/s at N=8192, and bf16
slower still. On a part whose packed-fp16 rate is three times its fp32 rate, that is the opposite of
what the hardware allows.

## That is rocBLAS, not PyTorch

[`scripts/rocblas_bench.cpp`](../../scripts/rocblas_bench.cpp) asks the library directly
(`rocblas_bench.out`):

| | N=4096 | N=8192 | ceiling | % of peak |
|---|---|---|---|---|
| SGEMM fp32 | 4527 | **4612 GFLOP/s** | 4740 | **97.3 %** |
| HGEMM fp16 | 4611 | **4689 GFLOP/s** | 14400 | **32.6 %** |
| DGEMM fp64 | 457 |, | none measured |, |

**HGEMM is 1.7 percent faster than SGEMM**, where the instruction rate says it could be three times
faster. PyTorch is passing the call through: 4.25 TFLOP/s in torch against 4.70 in the library,
the same plateau. The fp16 gap therefore belongs to rocBLAS, not to the framework.

**Correction, 25 September.** The rocBLAS table above and the figure built from it first read
4541 and 4593 for SGEMM and 4615 and 4698 for HGEMM, giving a 2.3 percent gap. Those values match
no run: `rocblas_bench.out` in this directory, and the same file still on the board, give 4526.9
and 4612.4 for SGEMM and 4611.4 and 4689.3 for HGEMM, a 1.7 percent gap. The table, README.md and
`scripts/make_gpgpu_figures.py` now carry the logged values. Nothing else in this directory
changes, and the conclusion does not: fp16 is not meaningfully faster than fp32 here.


**Correction, later the same day.** This paragraph originally went one step further and said Tensile
does not emit `v_pk_fma_f16` for this target. That was inferred from the two timings above, not
observed. Disassembling the code objects refutes it: the object HGEMM opens carries 720
`v_pk_fma_f16` ([`logs/hgemm-isa-2026-09-24/`](../hgemm-isa-2026-09-24/)). The gap is real and it is
in the library, but the reason is open.

That is the same hole this repository already exploited from the other side. llama.cpp's prefill got
1.4 to 1.75 times faster by not using rocBLAS for fp16 and writing a packed-fp16 GEMM instead
([`logs/rdna1-pkf16-tile-2026-09-20/`](../rdna1-pkf16-tile-2026-09-20/)). This is the measurement that
says why that was available: **roughly 3x sits unclaimed in rocBLAS's fp16 path on gfx1013.**

One caveat on HGEMM, visible in the log: its relative error grows to 2.5e-2 at N=8192 against SGEMM's
1.5e-6, because `rocblas_hgemm` accumulates in fp16. That is expected rather than a defect, and it is
a reason to prefer a fp32-accumulating path for anything numerically sensitive.

fp64 runs at 457 GFLOP/s, almost exactly a tenth of fp32. No fp64 ceiling was measured here, so no
percentage is claimed.

## Convolution and attention agree with the GEMM result

| | fp32 | fp16 | ceiling fp32 / fp16 |
|---|---|---|---|
| conv2d 3x3, 8x256x64x64 | **13.4** | 3.7 | 4.74 / 14.40 |
| conv2d 3x3, 16x128x64x64 | **12.8** | 3.3 | 4.74 / 14.40 |
| attention, seq 2048 | **1.64** | 1.47 | 4.74 / 14.40 |

**fp16 convolution is about four times slower than fp32**, and fp16 attention slower than fp32
attention, so the precision is the pattern and not any one operator.

The conv2d fp32 rates exceed the fp32 ceiling, 13.4 against 4.74, and that is a counting convention
and not a broken timer: the rate is effective flops by the direct-convolution formula, and a
library using Winograd does algebraically fewer multiplies than that formula counts. Exceeding the
ceiling is therefore evidence that MIOpen is **not** doing direct convolution in fp32, and, reading the
fp16 row the same way, that it is not doing the cheaper thing in fp16.

Attention reaches only 1.6 TFLOP/s, a third of the fp32 ceiling. This build reports no
memory-efficient attention kernel compiled in, which the log records, so the fallback path is what is
being measured.

## Memory-bound work

| elements | moved | GB/s | % of 432 |
|---|---|---|---|
| 1 M | 12 MiB | 282 | 65 % |
| 16 M | 192 MiB | 381 | 88 % |
| 64 M | 768 MiB | **388** | **90 %** |

`c = a + b` in fp32 reaches 90 percent of the streaming-read ceiling once the arrays are large enough
to hide launch cost. Elementwise PyTorch work is not leaving anything on the table either.

## Training

Four pre-norm transformer blocks, d_model 512, 8 heads, seq 256, Adam, one optimiser step timed:

| precision | batch 1 | batch 4 | batch 16 |
|---|---|---|---|
| fp32 | 66.6 steps/s (17.0 ktok/s) | 27.9 (28.6) | 8.25 (**33.8 ktok/s**) |
| fp16 autocast | 49.6 (12.7) | 6.16 (6.3) | 1.08 (**4.4 ktok/s**) |

fp32 scales the way it should, 17.0 to 33.8 ktok/s as the batch grows.

**`torch.autocast` with fp16 is a 7.7x regression at batch 16, not a speedup.** Given the GEMM table
that is unsurprising in direction, since fp16 matmul is already slower than fp32 here, but the size of
it is not explained by the GEMM numbers alone and is not chased further on this page. The practical
advice is plain: **do not use fp16 autocast on this board.** The log also records that this build has no
memory-efficient attention compiled in, so the attention path falls back.

## Against the CPU

The same fp32 GEMM at N=2048, twelve threads:

| | TFLOP/s | ms |
|---|---|---|
| torch CPU | 0.007 | 2341 |
| numpy, scipy-openblas | **0.284** | 60.5 |
| GPU | **2.986** | 5.8 |

**The GPU is 10.5 times the CPU at its best**, and that is the figure to quote. The torch-CPU row is
forty times slower than numpy on the same machine because this PyTorch is a source build with neither
MKL nor MKLDNN, so its CPU GEMM falls back to a generic kernel. Quoting the GPU against *that* would
have given a flattering and meaningless 420x, so both rows are here.

## What this does not say

One board, one PyTorch build, one shape of training model. The training figures are a small transformer
chosen to be representative, not a benchmark anyone else runs, so they are useful for comparing
precisions and batch sizes on this board and not for comparing this board with another machine. The
GEMM and bandwidth figures, being fractions of directly measured ceilings, travel better.

Nothing here was measured with `GGML_CUDA_GRAPH_OPT=1`, which is a llama.cpp setting and does not touch
PyTorch or rocBLAS.

## Files

`torch_bench.out` and `torch_bench.json` are the PyTorch run, `rocblas_bench.out` the library sweep.
Scripts: [`scripts/torch_bench.py`](../../scripts/torch_bench.py),
[`scripts/rocblas_bench.cpp`](../../scripts/rocblas_bench.cpp). The figures on the front page are drawn
from `torch_bench.json` directly by
[`scripts/make_gpgpu_figures.py`](../../scripts/make_gpgpu_figures.py), so re-running the benchmark
updates them instead of requiring the numbers to be copied across by hand.
