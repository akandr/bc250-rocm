// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// What arithmetic rates gfx1013 actually reaches: fp32 FMA, packed fp16 FMA (v_pk_fma_f16), fp16
// scalar, and the emulated int8 dot product MMQ depends on. Each kernel runs an unrolled chain of
// independent operations on registers, enough of them that the loop overhead is negligible; the rate is
// operations x 2 flop / time. Build: hipcc -O3 --offload-arch=gfx1013 alu_peak.cpp -o alu_peak
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>
#include <unistd.h>

#define ITERS 4096
#define UNROLL 16

__global__ void k_f32(float * out, float a, float b, int guard) {
    float acc[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) acc[i] = a + i;
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) acc[i] = fmaf(acc[i], b, a);
    }
    float s = 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) s += acc[i];
    if ((int) threadIdx.x == guard) out[blockIdx.x] = s;
}

__global__ void k_f16x2(float * out, float a, float b, int guard) {
    __half2 acc[UNROLL]; const __half2 va = __float2half2_rn(a), vb = __float2half2_rn(b);
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) acc[i] = __float2half2_rn(a + i);
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) acc[i] = __hfma2(acc[i], vb, va);
    }
    float s = 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) s += __low2float(acc[i]) + __high2float(acc[i]);
    if ((int) threadIdx.x == guard) out[blockIdx.x] = s;
}

__global__ void k_f16(float * out, float a, float b, int guard) {
    __half acc[UNROLL]; const __half va = __float2half(a), vb = __float2half(b);
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) acc[i] = __float2half(a + i);
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) acc[i] = __hfma(acc[i], vb, va);
    }
    float s = 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) s += __half2float(acc[i]);
    if ((int) threadIdx.x == guard) out[blockIdx.x] = s;
}

// the int8 dot product MMQ depends on. gfx1013 has no dot1-insts, so __builtin_amdgcn_sdot4 does not
// compile at all for it; this is the emulation ggml falls back to, four products and four adds.
static __device__ __forceinline__ int dp4a_emul(const int a, const int b, int c) {
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
}

__global__ void k_dp4a(float * out, int a, int b, int guard) {
    int acc[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) acc[i] = a + i;
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) acc[i] = dp4a_emul(acc[i], b, acc[i]);
    }
    int s = 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) s += acc[i];
    if ((int) threadIdx.x == guard) out[blockIdx.x] = (float) s;
}

static const char * clock_now() {                 // the clock the governor is actually holding, sampled inside the run
    static char buf[64]; buf[0] = 0;
    FILE * f = popen("cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '\\*' | tr -d ' \\n' | head -c 40", "r");
    if (f) { if (!fgets(buf, sizeof(buf), f)) buf[0] = 0; pclose(f); }
    return buf;
}

template <typename K> double run(K kernel, const char * name, double ops_per_lane_iter, int blocks, int threads) {
    float * d; hipMalloc(&d, blocks * sizeof(float));
    // warm the clock up before timing: the first kernel of a process otherwise runs at the idle clock
    for (int r = 0; r < 40; ++r) kernel<<<blocks, threads>>>(d, 1, 1, 100000);
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int r = 0; r < 4; ++r) kernel<<<blocks, threads>>>(d, 1, 1, 100000);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1);
    // and again, long enough to sample the clock in the middle of the work
    hipEventRecord(e0);
    for (int r = 0; r < 40; ++r) kernel<<<blocks, threads>>>(d, 1, 1, 100000);
    usleep(150000); const char * clk = clock_now();
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms_long = 0; hipEventElapsedTime(&ms_long, e0, e1); ms = ms_long / 10.0f;
    const double lanes = (double) blocks * threads;
    const double ops = lanes * ITERS * UNROLL * ops_per_lane_iter * 4;
    const double tops = ops / (ms * 1e-3) / 1e12;
    printf("%-26s %8.2f ms   %6.2f TFLOP/s   clock %s\n", name, ms, tops, clk);
    hipFree(d); return tops;
}

int main() {
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("%s, %d CUs, %d MHz\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    const int blocks = p.multiProcessorCount * 8, threads = 256;
    printf("blocks %d threads %d\n\n", blocks, threads);
    for (int round = 0; round < 2; ++round) {
        printf("round %d, forward order\n", round);
        run(k_f32,   "  fp32 fma",             2.0, blocks, threads);
        run(k_f16x2, "  packed fp16 fma",      4.0, blocks, threads);
        run(k_f16,   "  scalar fp16 fma",      2.0, blocks, threads);
        run(k_dp4a,  "  int8 dot4 (emulated)", 8.0, blocks, threads);
        printf("round %d, reverse order\n", round);
        run(k_dp4a,  "  int8 dot4 (emulated)", 8.0, blocks, threads);
        run(k_f16,   "  scalar fp16 fma",      2.0, blocks, threads);
        run(k_f16x2, "  packed fp16 fma",      4.0, blocks, threads);
        run(k_f32,   "  fp32 fma",             2.0, blocks, threads);
    }
    const double peak = (double) p.multiProcessorCount * 64 * 2 * (p.clockRate / 1e6);
    printf("\nfp32 FMA peak at %d CU and %.2f GHz: %.2f TFLOP/s\n", p.multiProcessorCount, p.clockRate / 1e6, peak);
    return 0;
}
