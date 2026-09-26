// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// How fast can a plain packed-fp16 tile GEMM be on gfx1013? The board's ceilings are 14.4 TFLOP/s for
// v_pk_fma_f16, 4.7 for fp32; ROCm's prefill runs at 2.5-2.9 (MMQ, emulated int8) and RADV's shaders at
// 4.9-5.4. This is a straightforward LDS-staged tile GEMM with packed-fp16 accumulation, to find out
// what the format is worth here before anything is written into llama.cpp.
//   C[M,N] = A[M,K] * B[K,N], all f16, row-major A, column-major B (B[n][k]), as a weight x activation
//   product is laid out: A is the weight matrix, B the activations.
// Build: hipcc -O3 --offload-arch=gfx1013 pk_gemm.cpp -o pk_gemm -lamdhip64
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#ifndef BM
#define BM 128      // rows of A per block
#endif
#ifndef BN
#define BN 128      // columns of B per block
#endif
#ifndef BK
#define BK 16       // depth per stage
#endif
#ifndef TM
#define TM 8        // rows per thread
#endif
#ifndef TN
#define TN 16       // columns per thread (TN/2 half2)
#endif
#define NTHREADS ((BM / TM) * (BN / TN))     // 16 x 8 = 128 threads

__global__ __launch_bounds__(NTHREADS) void pk_gemm(const __half * __restrict__ A, const __half * __restrict__ B,
                                                    __half * __restrict__ C, int M, int N, int K) {
    __shared__ __half As[BK][BM + 8];         // A tile transposed: As[k][m], padded against bank conflicts
    __shared__ __half Bs[BK][BN + 8];         // Bs[k][n]

    const int tid = threadIdx.x;
    const int tm = (tid % (BM / TM)) * TM;    // this thread's first row inside the tile
    const int tn = (tid / (BM / TM)) * TN;    // and first column
    const int row0 = blockIdx.x * BM, col0 = blockIdx.y * BN;

    __half2 acc[TM][TN / 2];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) acc[i][j] = __float2half2_rn(0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // stage: NTHREADS threads move BM*BK and BN*BK halves
#pragma unroll
        for (int idx = tid; idx < BM * BK; idx += NTHREADS) {
            const int m = idx / BK, k = idx % BK;
            As[k][m] = A[(size_t) (row0 + m) * K + (k0 + k)];
        }
#pragma unroll
        for (int idx = tid; idx < BN * BK; idx += NTHREADS) {
            const int n = idx / BK, k = idx % BK;
            Bs[k][n] = B[(size_t) (col0 + n) * K + (k0 + k)];
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            __half2 a[TM], b[TN / 2];
            // A: read pairs and split, so the broadcasts cost half as many LDS accesses
#pragma unroll
            for (int i = 0; i < TM; i += 2) {
                const __half2 pair = *(const __half2 *) &As[k][tm + i];
                a[i]     = __low2half2(pair);
                a[i + 1] = __high2half2(pair);
            }
#pragma unroll
            for (int j = 0; j < TN / 2; ++j) b[j] = *(const __half2 *) &Bs[k][tn + 2 * j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN / 2; ++j) acc[i][j] = __hfma2(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
            const int m = row0 + tm + i, n = col0 + tn + 2 * j;
            if (m < M && n + 1 < N) *(__half2 *) &C[(size_t) m * N + n] = acc[i][j];
        }
}

static void reference(const std::vector<__half> & A, const std::vector<__half> & B, std::vector<float> & C,
                      int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += __half2float(A[(size_t) m * K + k]) * __half2float(B[(size_t) n * K + k]);
            C[(size_t) m * N + n] = s;
        }
}

int main(int argc, char ** argv) {
    struct Shape { int M, N, K; const char * name; };
    std::vector<Shape> shapes = {
        {1536, 512, 1536, "Qcur      [1536,1536] x 512"},
        {8960, 512, 1536, "ffn_gate  [1536,8960] x 512"},
        {1536, 512, 8960, "ffn_down  [8960,1536] x 512"},
        {1536, 2048, 1536, "Qcur      [1536,1536] x 2048"},
        {8960, 2048, 1536, "ffn_gate  [1536,8960] x 2048"},
        {1536, 2048, 8960, "ffn_down  [8960,1536] x 2048"},
        {2048, 2048, 2048, "square 2048"},
    };
    // correctness on a small case first
    {
        const int M = 256, N = 128, K = 256;
        std::vector<__half> A((size_t) M * K), B((size_t) N * K);
        for (auto & x : A) x = __float2half((rand() % 17 - 8) * 0.125f);
        for (auto & x : B) x = __float2half((rand() % 17 - 8) * 0.125f);
        std::vector<float> ref((size_t) M * N); reference(A, B, ref, M, N, K);
        __half *dA, *dB, *dC; hipMalloc(&dA, A.size() * 2); hipMalloc(&dB, B.size() * 2); hipMalloc(&dC, (size_t) M * N * 2);
        hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice);
        hipMemcpy(dB, B.data(), B.size() * 2, hipMemcpyHostToDevice);
        pk_gemm<<<dim3(M / BM, N / BN), NTHREADS>>>(dA, dB, dC, M, N, K);
        std::vector<__half> hC((size_t) M * N); hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost);
        double worst = 0;
        for (size_t i = 0; i < hC.size(); ++i) worst = fmax(worst, fabs(__half2float(hC[i]) - ref[i]) / fmax(1.0f, fabs(ref[i])));
        printf("correctness on 256x128x256: worst relative error %.4f (fp16 accumulation)\n\n", worst);
        hipFree(dA); hipFree(dB); hipFree(dC);
    }
    printf("%-34s %10s %10s %12s\n", "shape", "ms", "TFLOP/s", "of 14.4 peak");
    for (auto & s : shapes) {
        const int M = s.M, N = s.N, K = s.K;
        if (M % BM || N % BN || K % BK) { printf("%-34s  skipped (tile)\n", s.name); continue; }
        __half *dA, *dB, *dC;
        hipMalloc(&dA, (size_t) M * K * 2); hipMalloc(&dB, (size_t) N * K * 2); hipMalloc(&dC, (size_t) M * N * 2);
        hipMemset(dA, 0x11, (size_t) M * K * 2); hipMemset(dB, 0x11, (size_t) N * K * 2);
        const dim3 grid(M / BM, N / BN);
        for (int w = 0; w < 8; ++w) pk_gemm<<<grid, NTHREADS>>>(dA, dB, dC, M, N, K);
        hipDeviceSynchronize();
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0);
        const int reps = 30;
        for (int r = 0; r < reps; ++r) pk_gemm<<<grid, NTHREADS>>>(dA, dB, dC, M, N, K);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= reps;
        double best = 1e9;
        for (int pass = 0; pass < 3; ++pass) {                       // three timed passes, keep the best
            hipEventRecord(e0);
            for (int r = 0; r < reps; ++r) pk_gemm<<<grid, NTHREADS>>>(dA, dB, dC, M, N, K);
            hipEventRecord(e1); hipEventSynchronize(e1);
            float t = 0; hipEventElapsedTime(&t, e0, e1); best = fmin(best, (double) t / reps);
        }
        ms = best;
        const double tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
        printf("%-34s %10.3f %10.2f %11.0f%%\n", s.name, ms, tf, 100 * tf / 14.4);
        hipFree(dA); hipFree(dB); hipFree(dC);
    }
    return 0;
}
