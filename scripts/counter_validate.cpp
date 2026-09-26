// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// counter_validate: a kernel whose VALU instruction count and wave count are known exactly, so that
// hardware performance counters read on gfx1013 can be checked against arithmetic, not trusted.
//
// This exists because gfx1013 is absent from rocprofiler-sdk's counter_defs.yaml, which lists
// gfx1010/gfx1030/gfx1031/gfx1032 and matches the agent name exactly. Adding gfx1013 to the gfx1010
// lists makes counters appear, but that is an assumption about the hardware block layout, not a
// measurement. Nothing read through those counters means anything until this kernel's predicted
// numbers come back.
//
// The kernel body is UNROLL independent v_fma_f32 chains repeated ITERS times, fully unrolled, with
// no memory traffic in the loop. Predicted per wave:
//   SQ_INSTS_VALU  >= ITERS * UNROLL        (the FMAs; loop and setup add a little)
//   SQ_WAVES        = blocks * threads / 32
// The check is on SQ_INSTS_VALU: it must exceed the FMA count and miss it by only the loop overhead,
// a few instructions per iteration at most. A counter that is off by a factor, or reads zero, or
// scales with the wrong quantity, says the gfx1010 block layout does not transfer.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1013 counter_validate.cpp -o counter_validate -lamdhip64
// Run under the profiler:
//   rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE -- ./counter_validate

#include <hip/hip_runtime.h>
#include <cstdio>

#define ITERS  1024
#define UNROLL 8

__global__ void fma_chain(float * out, int guard) {
    float acc[UNROLL];
    const float a = (float) guard, b = (float) (guard + 1);
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) acc[i] = a + i;
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) acc[i] = fmaf(acc[i], b, a);
    }
    float s = 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) s += acc[i];
    if ((int) threadIdx.x == guard) out[0] = s;   // guard is -1 at run time, so the store never fires
}

int main(int argc, char ** argv) {
    const int blocks  = argc > 1 ? atoi(argv[1]) : 256;
    const int threads = 32;                      // one wave per block, wave32

    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);

    const double waves = (double) blocks * threads / 32.0;
    const double fmas  = (double) ITERS * UNROLL;

    printf("device %s  arch %s\n", p.name, p.gcnArchName);
    printf("launch: %d blocks x %d threads, %.0f waves\n", blocks, threads, waves);
    printf("predicted SQ_WAVES      = %.0f\n", waves);
    printf("predicted SQ_INSTS_VALU >= %.0f   (%.0f FMAs per wave x %.0f waves, plus loop overhead)\n",
           fmas * waves, fmas, waves);

    float * d;
    hipMalloc(&d, sizeof(float));
    fma_chain<<<blocks, threads>>>(d, -1);       // warm up; counted runs are the profiler's business
    hipDeviceSynchronize();
    fma_chain<<<blocks, threads>>>(d, -1);
    hipDeviceSynchronize();
    hipFree(d);

    printf("two launches issued; the profiler should report each separately\n");
    return 0;
}
