// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// rocblas_bench.cpp: SGEMM, HGEMM and DGEMM on native gfx1013 rocBLAS, swept over size,
// reported against the arithmetic ceilings this repository measured on the same silicon.
//
// The GPGPU section already gives SGEMM and DGEMM at two sizes each. This adds the size curve and,
// more to the point, HGEMM, because PyTorch's fp16 matmul plateaus at 29 percent of the packed-fp16
// ceiling (logs/torch-pytorch-bench-2026-09-24/) and the question is whether that is PyTorch's doing
// or the library's. Same buffers reused, best-of and median over iterations, two elements checked
// against a double-precision CPU reference at every size.
//
// Ceilings from logs/alu-rates-2026-09-19/, measured with dependent-free accumulator chains:
//   v_fma_f32      4.74 TFLOP/s
//   v_pk_fma_f16  14.40 TFLOP/s
// fp64 has no measured ceiling here, so DGEMM is reported without one.
//
// Build:
//   /usr/lib64/rocm/llvm/bin/clang++ -std=c++17 -O2 -x hip --offload-arch=gfx1013 \
//     -I/opt/bc250-rocm/include rocblas_bench.cpp -o rocblas_bench \
//     -L/opt/bc250-rocm/lib64 -lrocblas -L/usr/lib64 -lamdhip64
// Run:
//   LD_LIBRARY_PATH=/opt/bc250-rocm/lib64:/usr/lib64 ./rocblas_bench [iters]

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define HIP_OK(x)  do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "hip error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)
#define RB_OK(x)   do { rocblas_status s = (x); if (s != rocblas_status_success) { \
    fprintf(stderr, "rocblas error %d at %d\n", (int) s, __LINE__); exit(1);} } while (0)

static const double CEIL_F32 = 6.52;   // TFLOP/s, logs/alu-rates-recheck-2026-09-25
static const double CEIL_F16 = 13.02;  // TFLOP/s, packed, exactly 2x fp32

// rocblas_half carries no conversions, so give every element type an explicit pair.
template <typename T> static inline T from_d(double v) { return (T) v; }
template <> inline rocblas_half from_d<rocblas_half>(double v) {
    _Float16 x = (_Float16) v;
    rocblas_half r;
    *reinterpret_cast<uint16_t *>(&r) = *reinterpret_cast<uint16_t *>(&x);
    return r;
}
static inline double to_d(float v)  { return (double) v; }
static inline double to_d(double v) { return v; }
static inline double to_d(const rocblas_half & v) {
    _Float16 x;
    *reinterpret_cast<uint16_t *>(&x) = *reinterpret_cast<const uint16_t *>(&v);
    return (double) x;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// One element of C checked against a double-precision CPU dot product.
template <typename T>
static double check_one(const std::vector<T> & A, const std::vector<T> & B,
                        const std::vector<T> & C, int N, int row, int col) {
    double ref = 0.0;
    for (int k = 0; k < N; k++) {
        ref += to_d(A[row + (size_t) k * N]) * to_d(B[k + (size_t) col * N]);  // column-major
    }
    double got = to_d(C[row + (size_t) col * N]);
    return std::fabs(got - ref) / std::max(1.0, std::fabs(ref));
}

template <typename T, typename F>
static void sweep(const char * name, rocblas_handle h, const std::vector<int> & sizes, int iters,
                  double ceiling, F gemm) {
    printf("\n## %s\n", name);
    if (ceiling > 0) {
        printf("%6s %10s %12s %10s %8s %9s %10s\n",
               "N", "ms", "GFLOP/s", "ceiling", "% peak", "spread", "rel err");
    } else {
        printf("%6s %10s %12s %10s %8s %9s %10s\n",
               "N", "ms", "GFLOP/s", "", "", "spread", "rel err");
    }
    for (int N : sizes) {
        size_t n2 = (size_t) N * N;
        std::vector<T> hA(n2), hB(n2), hC(n2);
        for (size_t i = 0; i < n2; i++) {
            hA[i] = from_d<T>(((i * 1103515245u + 12345u) % 1000) / 1000.0 - 0.5);
            hB[i] = from_d<T>(((i * 22695477u + 1u) % 1000) / 1000.0 - 0.5);
        }
        T *dA, *dB, *dC;
        if (hipMalloc(&dA, n2 * sizeof(T)) != hipSuccess) { printf("%6d   out of memory\n", N); continue; }
        HIP_OK(hipMalloc(&dB, n2 * sizeof(T)));
        HIP_OK(hipMalloc(&dC, n2 * sizeof(T)));
        HIP_OK(hipMemcpy(dA, hA.data(), n2 * sizeof(T), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(dB, hB.data(), n2 * sizeof(T), hipMemcpyHostToDevice));

        gemm(h, N, dA, dB, dC);            // warm up, and let Tensile pick its kernel
        HIP_OK(hipDeviceSynchronize());

        std::vector<double> ts;
        for (int it = 0; it < iters; it++) {
            auto t0 = std::chrono::steady_clock::now();
            gemm(h, N, dA, dB, dC);
            HIP_OK(hipDeviceSynchronize());
            ts.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        HIP_OK(hipMemcpy(hC.data(), dC, n2 * sizeof(T), hipMemcpyDeviceToHost));

        double med = median(ts);
        double lo = *std::min_element(ts.begin(), ts.end());
        double hi = *std::max_element(ts.begin(), ts.end());
        double gflops = 2.0 * (double) N * N * N / med / 1e9;
        double err = std::max(check_one(hA, hB, hC, N, 0, 0),
                              check_one(hA, hB, hC, N, N / 2, N / 3));

        if (ceiling > 0) {
            printf("%6d %10.3f %12.1f %10.2f %7.1f%% %8.1f%% %10.2e\n",
                   N, med * 1e3, gflops, ceiling * 1000, gflops / (ceiling * 1000) * 100,
                   (hi - lo) / med * 100, err);
        } else {
            printf("%6d %10.3f %12.1f %10s %8s %8.1f%% %10.2e\n",
                   N, med * 1e3, gflops, "-", "-", (hi - lo) / med * 100, err);
        }
        HIP_OK(hipFree(dA)); HIP_OK(hipFree(dB)); HIP_OK(hipFree(dC));
    }
}

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 5;

    hipDeviceProp_t prop;
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    printf("device %s  arch %s  CUs %d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    printf("ceilings measured on this board: fp32 %.2f TFLOP/s, packed fp16 %.2f TFLOP/s; "
           "fp64 has none measured\n", CEIL_F32, CEIL_F16);
    printf("median of %d timed iterations after one warm-up; two elements CPU-checked per size\n", iters);

    rocblas_handle h;
    RB_OK(rocblas_create_handle(&h));

    const float  f1 = 1.0f, f0 = 0.0f;
    const double d1 = 1.0,  d0 = 0.0;
    // rocblas_half is a 16-bit struct; set the IEEE half bit patterns for 1.0 and 0.0 directly.
    // HIP declares a __device__ memcpy that shadows the host one here, so no memcpy.
    rocblas_half h1, h0;
    static_assert(sizeof(rocblas_half) == 2, "rocblas_half is not 16 bits");
    *reinterpret_cast<uint16_t *>(&h1) = 0x3C00;  // 1.0
    *reinterpret_cast<uint16_t *>(&h0) = 0x0000;  // 0.0

    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192};

    sweep<float>("SGEMM, fp32", h, sizes, iters, CEIL_F32,
        [&](rocblas_handle hh, int N, float * A, float * B, float * C) {
            RB_OK(rocblas_sgemm(hh, rocblas_operation_none, rocblas_operation_none,
                                N, N, N, &f1, A, N, B, N, &f0, C, N));
        });

    sweep<rocblas_half>("HGEMM, fp16", h, sizes, iters, CEIL_F16,
        [&](rocblas_handle hh, int N, rocblas_half * A, rocblas_half * B, rocblas_half * C) {
            RB_OK(rocblas_hgemm(hh, rocblas_operation_none, rocblas_operation_none,
                                N, N, N, &h1, A, N, B, N, &h0, C, N));
        });

    std::vector<int> dsizes = {512, 1024, 2048, 4096};
    sweep<double>("DGEMM, fp64", h, dsizes, iters, 0.0,
        [&](rocblas_handle hh, int N, double * A, double * B, double * C) {
            RB_OK(rocblas_dgemm(hh, rocblas_operation_none, rocblas_operation_none,
                                N, N, N, &d1, A, N, B, N, &d0, C, N));
        });

    RB_OK(rocblas_destroy_handle(h));
    printf("\ndone\n");
    return 0;
}
