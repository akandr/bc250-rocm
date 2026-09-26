// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// Cycles per VALU instruction on gfx1013, measured with the shader clock inside the kernel, one wave per
// SIMD so there is no contention and no dependence on the lane count or the governor's clock.
//
// Every chain is seeded with the lane index, not from the kernel argument alone. Seeded from
// the argument the accumulators are uniform across the wave, and while floating point survives that
// (RDNA1 has no scalar float ALU) the integer chain is moved onto the scalar unit entirely: the code
// emitted for the int8 dot product was s_mul_i32, s_bfe_i32 and s_sext_i32_i8, and hardware counters
// read 5 VALU instructions per wave against the 32768 the loop contains. Until this was corrected on
// 25 September 2026 the int8 row measured the scalar unit, not the vector emulation it named.
// 16 independent accumulator chains hide the ALU latency. Build:
//   hipcc -O3 --offload-arch=gfx1013 alu_cycles.cpp -o alu_cycles -lamdhip64
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

#define ITERS 2048
#define UNROLL 16

#define BODY(init, op, sum)                                                       \
    const int lane = (int) (threadIdx.x & 1u);                                     \
    T acc[UNROLL];                                                                \
    _Pragma("unroll") for (int i = 0; i < UNROLL; ++i) acc[i] = init;             \
    const long long t0 = clock64();                                               \
    for (int it = 0; it < ITERS; ++it) {                                          \
        _Pragma("unroll") for (int i = 0; i < UNROLL; ++i) op;                    \
    }                                                                             \
    const long long t1 = clock64();                                               \
    float s = 0; _Pragma("unroll") for (int i = 0; i < UNROLL; ++i) s += sum;     \
    if ((int) threadIdx.x == guard) out[0] = s;                                   \
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;

__global__ void c_f32(float * out, long long * cyc, int guard) {
    using T = float; const T a = (T) guard, b = (T) (guard + 1);
    BODY(a + i + (float) lane, acc[i] = fmaf(acc[i], b, a), acc[i])
}
__global__ void c_f16(float * out, long long * cyc, int guard) {
    using T = __half; const T a = __float2half((float) guard), b = __float2half((float) guard + 1);
    BODY(__float2half((float) (i + lane)), acc[i] = __hfma(acc[i], b, a), __half2float(acc[i]))
}
__global__ void c_f16x2(float * out, long long * cyc, int guard) {
    using T = __half2; const T a = __float2half2_rn((float) guard), b = __float2half2_rn((float) guard + 1);
    BODY(__float2half2_rn((float) (i + lane)), acc[i] = __hfma2(acc[i], b, a), __low2float(acc[i]))
}
__global__ void c_mul_f32(float * out, long long * cyc, int guard) {
    using T = float; const T b = (T) (guard + 1);
    BODY((T) (guard + i + lane), acc[i] = acc[i] * b, acc[i])
}
__global__ void c_dp4a(float * out, long long * cyc, int guard) {
    using T = int; const T b = guard + 1;
    BODY(guard + i + lane, ({ const int8_t * x = (const int8_t *) &acc[i]; const int8_t * y = (const int8_t *) &b;
                       acc[i] = acc[i] + x[0]*y[0] + x[1]*y[1] + x[2]*y[2] + x[3]*y[3]; }), (float) acc[i])
}

template <typename K> void run(K kernel, const char * name, double instr_per_iter, double flop_per_instr, int cus,
                                int blocks = 0, int threads = 32) {
    if (blocks == 0) blocks = cus;
    float * d; long long * c; hipMalloc(&d, 4); hipMalloc(&c, blocks * sizeof(long long));
    for (int w = 0; w < 20; ++w) kernel<<<blocks, threads>>>(d, c, 100000);   // warm the clock
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0);
    for (int r = 0; r < 4; ++r) kernel<<<blocks, threads>>>(d, c, 100000);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= 4;
    const int cus_ = blocks;
    std::vector<long long> h(cus_); hipMemcpy(h.data(), c, cus_ * sizeof(long long), hipMemcpyDeviceToHost);
    long long med = 0; { std::vector<long long> v = h; std::sort(v.begin(), v.end()); med = v[cus_ / 2]; }
    const double instr = (double) ITERS * UNROLL * instr_per_iter;
    const double cyc_per_instr = med / instr;
    const double waves = (double) blocks * threads / 32.0;
    const double lanes = 2560.0;                       // 40 CU x 64
    const double tflops = waves * 32 * instr * flop_per_instr / (ms * 1e-3) / 1e12;
    const double eff_ghz = med / (ms * 1e6) * (waves > 20 ? 1.0 : 1.0);
    printf("%-15s %5.0f waves  %6.3f cyc/instr  %6.2f TFLOP/s  shader clock during run %.2f GHz\n",
           name, waves, cyc_per_instr, tflops, eff_ghz);
    (void) lanes;
    hipFree(d); hipFree(c);
}
#include <vector>
#include <algorithm>
int main() {
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("%s, %d WGP/CU units reported, wave %d\n\n", p.name, p.multiProcessorCount, p.warpSize);
    const int cus = p.multiProcessorCount;
    for (int occ : {1, 4, 16}) {
        const int blocks = cus * 2 * occ;             // 2 SIMD per WGP-unit reported, occ waves each
        printf("\n-- %d wave(s) per SIMD --\n", occ);
        run(c_f32,    "v_fma_f32",      1.0, 2.0, cus, blocks, 32);
        run(c_f16,    "v_fma_f16",      1.0, 2.0, cus, blocks, 32);
        run(c_f16x2,  "v_pk_fma_f16",   1.0, 4.0, cus, blocks, 32);
        run(c_dp4a,   "int8 dot4 emul", 8.0, 1.0, cus, blocks, 32);
    }
    return 0;
}
