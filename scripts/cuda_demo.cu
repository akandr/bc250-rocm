// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// cuda_demo.cu: ordinary CUDA, written as if for an NVIDIA card, never edited for AMD.
//
// The point of this file is that it is *not* HIP. It includes cuda_runtime.h and cublas_v2.h, calls
// cudaMalloc/cudaMemcpy/cudaDeviceSynchronize, launches with <<< >>>, times with cudaEvent_t and calls
// cublasSgemm. hipify-perl rewrites it mechanically and hipcc compiles the result for gfx1013.
//
// It exercises the two things that translate by different mechanisms:
//   1. a hand-written kernel, which becomes a HIP kernel with the same body;
//   2. a cuBLAS call, which becomes hipBLAS and lands in the same rocBLAS measured elsewhere here.
// Both are checked against a CPU reference, because a port that compiles and returns wrong numbers is
// the failure worth catching.
//
// On an NVIDIA machine:  nvcc -O2 cuda_demo.cu -o cuda_demo -lcublas
// Here:                  hipify-perl cuda_demo.cu > cuda_demo.hip.cpp
//                        hipcc -O2 --offload-arch=gfx1013 cuda_demo.hip.cpp -o cuda_demo -lhipblas

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CU_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {                      \
    fprintf(stderr, "cuda error %s at line %d\n", cudaGetErrorString(e), __LINE__);     \
    exit(1); } } while (0)

#define BLAS_OK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) {       \
    fprintf(stderr, "cublas error %d at line %d\n", (int) s, __LINE__);                 \
    exit(1); } } while (0)

// A hand-written kernel with enough arithmetic to be worth timing: a few Newton steps of an inverse
// square root, per element, so it is not purely memory bound.
__global__ void rsqrt_iterate(const float * __restrict__ in, float * __restrict__ out, int n, int steps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    float y = 1.0f / sqrtf(x);
    for (int s = 0; s < steps; s++) {
        y = y * (1.5f - 0.5f * x * y * y);   // Newton, 4 flops plus a multiply
    }
    out[i] = y;
}

int main(int argc, char ** argv) {
    const int n     = argc > 1 ? atoi(argv[1]) : (1 << 22);
    const int steps = argc > 2 ? atoi(argv[2]) : 64;
    const int N     = argc > 3 ? atoi(argv[3]) : 4096;

    int dev = 0;
    cudaDeviceProp prop;
    CU_OK(cudaGetDevice(&dev));
    CU_OK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s\n", prop.name);

    // ---- 1. the hand-written kernel ----
    std::vector<float> h_in(n), h_out(n);
    for (int i = 0; i < n; i++) h_in[i] = 1.0f + (float) (i % 1000) / 1000.0f;

    float *d_in, *d_out;
    CU_OK(cudaMalloc(&d_in, n * sizeof(float)));
    CU_OK(cudaMalloc(&d_out, n * sizeof(float)));
    CU_OK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    rsqrt_iterate<<<blocks, threads>>>(d_in, d_out, n, steps);   // warm up
    CU_OK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CU_OK(cudaEventCreate(&t0));
    CU_OK(cudaEventCreate(&t1));
    CU_OK(cudaEventRecord(t0));
    const int iters = 20;
    for (int it = 0; it < iters; it++) rsqrt_iterate<<<blocks, threads>>>(d_in, d_out, n, steps);
    CU_OK(cudaEventRecord(t1));
    CU_OK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CU_OK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    CU_OK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    double worst = 0.0;
    for (int i = 0; i < n; i += 4096) {
        double ref = 1.0 / std::sqrt((double) h_in[i]);
        worst = std::max(worst, std::fabs(h_out[i] - ref) / ref);
    }
    // 5 flops per Newton step, plus the initial rsqrt, counted as one
    double kflops = (double) n * (5.0 * steps + 1.0);
    printf("kernel   %d elements, %d Newton steps: %8.3f ms  %7.1f GFLOP/s  rel err %.2e\n",
           n, steps, ms, kflops / (ms * 1e-3) / 1e9, worst);

    // ---- 2. the library call ----
    std::vector<float> A((size_t) N * N), B((size_t) N * N), C((size_t) N * N);
    for (size_t i = 0; i < A.size(); i++) {
        A[i] = (float) (((i * 1103515245u + 12345u) % 1000) / 1000.0 - 0.5);
        B[i] = (float) (((i * 22695477u + 1u) % 1000) / 1000.0 - 0.5);
    }
    float *dA, *dB, *dC;
    CU_OK(cudaMalloc(&dA, A.size() * sizeof(float)));
    CU_OK(cudaMalloc(&dB, B.size() * sizeof(float)));
    CU_OK(cudaMalloc(&dC, C.size() * sizeof(float)));
    CU_OK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CU_OK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t blas;
    BLAS_OK(cublasCreate(&blas));
    const float alpha = 1.0f, beta = 0.0f;

    BLAS_OK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                        &alpha, dA, N, dB, N, &beta, dC, N));       // warm up
    CU_OK(cudaDeviceSynchronize());

    CU_OK(cudaEventRecord(t0));
    const int giters = 5;
    for (int it = 0; it < giters; it++) {
        BLAS_OK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                            &alpha, dA, N, dB, N, &beta, dC, N));
    }
    CU_OK(cudaEventRecord(t1));
    CU_OK(cudaEventSynchronize(t1));
    CU_OK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= giters;

    CU_OK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // one element against a double-precision CPU dot product, column-major
    double ref = 0.0;
    for (int k = 0; k < N; k++) ref += (double) A[(size_t) k * N] * (double) B[k];
    double rel = std::fabs((double) C[0] - ref) / std::max(1.0, std::fabs(ref));
    printf("cublas   SGEMM N=%d: %8.3f ms  %7.1f GFLOP/s  rel err %.2e\n",
           N, ms, 2.0 * N * N * N / (ms * 1e-3) / 1e9, rel);

    BLAS_OK(cublasDestroy(blas));
    CU_OK(cudaFree(dA)); CU_OK(cudaFree(dB)); CU_OK(cudaFree(dC));
    CU_OK(cudaFree(d_in)); CU_OK(cudaFree(d_out));
    printf("ok\n");
    return 0;
}
