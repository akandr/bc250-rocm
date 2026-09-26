# Porting a CUDA program to this board, and what it costs, 2026-09-24

[The backend section](../../README.md#which-backend-to-use-and-why-rocm-at-all) claims that hipified
CUDA is one of the things ROCm gives this board and Vulkan cannot. That is worth demonstrating rather
than asserting.

[`scripts/cuda_demo.cu`](../../scripts/cuda_demo.cu) is ordinary CUDA, written as if for an NVIDIA
card and never edited for AMD: it includes `cuda_runtime.h` and `cublas_v2.h`, calls `cudaMalloc`,
`cudaMemcpy` and `cudaDeviceSynchronize`, launches with `<<< >>>`, times with `cudaEvent_t`, and calls
`cublasSgemm`. It exercises the two things that translate by different mechanisms: a hand-written
kernel, and a library call. Both are checked against a CPU reference, because a port that compiles and
returns wrong numbers is the failure worth catching.

## The translation

`hipify-perl` rewrites it mechanically:

| | |
|---|---|
| lines | 142 → 143 |
| lines changed | 41, of which 38 are code |
| hand edits afterwards | **0** |
| kernel body | **untouched** |

`__global__`, `threadIdx`, `blockIdx` and the `<<< >>>` launch syntax are shared between the two
languages, so **the device code does not change at all**. What changes is the host side:
`cudaMalloc` → `hipMalloc`, `cudaEvent_t` → `hipEvent_t`, `cublasSgemm` → `hipblasSgemm`,
`CUBLAS_STATUS_SUCCESS` → `HIPBLAS_STATUS_SUCCESS`. The translated file is kept here as
`cuda_demo.hip.cpp`.

Two things hipify does not know about this machine, and both are build flags, not code:

- it emits `#include <hipblas.h>`, and Fedora packages that header as `hipblas/hipblas.h`, so the
  compile needs `-I/usr/include/hipblas`;
- `hipcc` did not pull the HIP runtime into this link by itself, so it needs `-lamdhip64`.

## It runs at native speed

| | this port | native |
|---|---|---|
| hand-written kernel, 4.2 M elements, 64 Newton steps | **3307 GFLOP/s** (median of five, range 3306 to 3309) | n/a |
| `cublasSgemm` → `hipblasSgemm`, N=4096 | median **4225**, best **4528 GFLOP/s** | rocBLAS SGEMM **4541 GFLOP/s** |

The library call lands in the same rocBLAS measured in
[`logs/torch-rocblas-bench-2026-09-24/`](../torch-rocblas-bench-2026-09-24/), and best against best it
is the same kernel taking the same time, 30.35 ms here against 30.27 ms there. **There is no runtime
penalty for having come from CUDA source**, which is the expected answer and the one worth confirming:
hipify is a source translation, not a shim, and the binary it produces is an ordinary HIP binary.

N=4096 is a noisy size on this board in both harnesses, 4155 to 4528 over five runs here and a 44.9
percent spread in the rocBLAS sweep's own five, so the median and the range are both given
and why the best-to-best comparison is the meaningful one. The hand-written kernel, which is not a
Tensile-dispatched shape, is steady to 0.1 percent.

Correctness: the kernel agrees with a CPU `1/sqrt` to 6.6e-08, and the GEMM's first element with a
double-precision CPU dot product to 3.8e-07.

## What this does not say

One small program. It uses the parts of the CUDA runtime that hipify handles best; a real port pulls in
things that do not translate, and this repository's own PyTorch build needed a patch set instead of a
script. What it does show is the shape of the path: for CUDA that sticks to the runtime API and a BLAS
call, the translation is mechanical, the device code is untouched, and the result runs at the speed a
native HIP program would.

Nothing equivalent exists on the Vulkan side: there the kernels would be rewritten in GLSL or SPIR-V
and the BLAS call would have no home at all.

## Files

`hipify_demo.out` is the entire run, translation through five timed repetitions.
`cuda_demo.hip.cpp` is hipify's output, unedited. The source is
[`scripts/cuda_demo.cu`](../../scripts/cuda_demo.cu) and the driver
[`scripts/hipify_demo.sh`](../../scripts/hipify_demo.sh).

One practical note recorded because it cost nothing here and could cost a lot: **`dnf install hipify`
on this machine wants 153 packages and removes 10, including both installed kernels.** The script is a
standalone Perl file, so it was extracted from the RPM with `rpm2cpio` and run from `/tmp` instead.
