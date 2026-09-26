// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// What one kernel dispatch costs on gfx1013 when the kernel itself costs nothing.
//
// The mixture-of-experts model decodes at 0.81 of Vulkan's rate with no single kernel to blame: replaying
// its one-token graph puts ROCm at or ahead of Vulkan on every shape it contains, and what distinguishes
// it is that it issues about 1300 dispatches a token with a mean kernel of 9.3 microseconds. Kernel traces
// of that model show 2 to 5 microsecond gaps between kernels, and HIP graph replay leaves the gap
// distribution unchanged, so the gap is not the host-side launch cost that graph replay removes
// (logs/moe-decode-2026-09-20/).
//
// That leaves a floor to measure, not a theory to argue. This launches a kernel that does nothing
// and a kernel that spins for a known number of shader cycles, back to back on one stream, and reports
// the wall time per dispatch. The difference between the spinning kernel's wall time and its own duration
// is what a dispatch costs when the queue is never empty. If that floor is 2 to 5 microseconds, the MoE's
// gaps are the machine and no arrangement of llama.cpp's kernels removes them; if it is far below,
// something in the sequence is adding them and the gaps are worth chasing.
//
// Three arms, because they isolate different costs:
//   stream     hipLaunchKernelGGL on one stream, the way ggml dispatches
//   graph      the same sequence captured and replayed with hipGraphLaunch
//   nullkern   an empty kernel, which gives the dispatch cost with no work at all to overlap it
//
// Build on the board:
//   hipcc -O3 --offload-arch=gfx1013 dispatch_floor.cpp -o dispatch_floor -lamdhip64
// Run with the GPU otherwise idle; the first measurement of any arm runs at the idle clock, so each arm
// is warmed before it is timed.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>

#define CHECK(x)                                                                      \
    do {                                                                              \
        hipError_t e_ = (x);                                                          \
        if (e_ != hipSuccess) {                                                        \
            std::printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_));      \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

// Does nothing observable but cannot be removed: the store is behind a predicate no lane satisfies.
__global__ void nop_kernel(float * out) {
    if (threadIdx.x == 1024) out[0] = 1.0f;
}

// Spins on the shader clock for a fixed number of cycles, so the kernel has a duration that does not
// depend on memory or on how many CUs it lands on. One wave, one block: the point is the dispatch, not
// the occupancy.
__global__ void spin_kernel(float * out, long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
    if (threadIdx.x == 1024) out[0] = 1.0f;
}

// A created stream instead of the null one: the null stream carries implicit synchronisation with
// every other stream, which is part of what this is trying to measure the absence of, and ggml
// dispatches on its own stream anyway.
static double time_stream(float * out, long long cycles, int n, int block, int grid) {
    hipStream_t s;
    CHECK(hipStreamCreate(&s));
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(a, s));
    for (int i = 0; i < n; ++i) {
        if (cycles == 0) hipLaunchKernelGGL(nop_kernel,  dim3(grid), dim3(block), 0, s, out);
        else             hipLaunchKernelGGL(spin_kernel, dim3(grid), dim3(block), 0, s, out, cycles);
    }
    CHECK(hipEventRecord(b, s));
    CHECK(hipEventSynchronize(b));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, a, b));
    CHECK(hipEventDestroy(a));
    CHECK(hipEventDestroy(b));
    CHECK(hipStreamDestroy(s));
    return ms * 1000.0 / n;  // microseconds per dispatch
}

static double time_graph(float * out, long long cycles, int n, int block, int grid) {
    hipStream_t s;
    CHECK(hipStreamCreate(&s));
    hipGraph_t g;
    hipGraphExec_t ge;
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    for (int i = 0; i < n; ++i) {
        if (cycles == 0) hipLaunchKernelGGL(nop_kernel,  dim3(grid), dim3(block), 0, s, out);
        else             hipLaunchKernelGGL(spin_kernel, dim3(grid), dim3(block), 0, s, out, cycles);
    }
    CHECK(hipStreamEndCapture(s, &g));
    CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CHECK(hipGraphLaunch(ge, s));           // warm the replay
    CHECK(hipStreamSynchronize(s));

    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    CHECK(hipEventRecord(a, s));
    CHECK(hipGraphLaunch(ge, s));
    CHECK(hipEventRecord(b, s));
    CHECK(hipEventSynchronize(b));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, a, b));
    CHECK(hipEventDestroy(a));
    CHECK(hipEventDestroy(b));
    CHECK(hipGraphExecDestroy(ge));
    CHECK(hipGraphDestroy(g));
    CHECK(hipStreamDestroy(s));
    return ms * 1000.0 / n;
}

// Best of several passes: one slow pass contaminates a mean, and the quantity wanted is the floor.
static double best(double (*fn)(float *, long long, int, int, int), float * out,
                   long long cycles, int n, int block, int grid, int passes) {
    double b = 1e30;
    for (int p = 0; p < passes; ++p) b = std::min(b, fn(out, cycles, n, block, grid));
    return b;
}

int main(int argc, char ** argv) {
    const int n      = argc > 1 ? std::atoi(argv[1]) : 20000;
    const int passes = argc > 2 ? std::atoi(argv[2]) : 5;
    const int block  = argc > 3 ? std::atoi(argv[3]) : 64;

    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s, %d CUs, clock %d kHz, %d dispatches per measurement, best of %d\n",
                prop.gcnArchName, prop.multiProcessorCount, prop.clockRate, n, passes);

    float * out = nullptr;
    CHECK(hipMalloc(&out, sizeof(float)));

    // Do NOT calibrate the shader clock with a single kernel on an idle board. The first version of
    // this program did, and it read 1002 cycles per microsecond because the board was at its 1000 MHz
    // step when that one kernel ran, while the 20000-dispatch loop below drives the clock to the
    // pinned 1500. Every derived duration was then 50 percent too long and the gap column came out
    // negative. The clock is recovered from the measurement instead: per-dispatch time is linear in
    // the requested cycle count, with slope 1/f and intercept the dispatch floor, so a fit over
    // several cycle counts gives both, measured in exactly the regime being reported.
    const double CYC_NOMINAL = 1500.0;  // only to choose the sweep points; the fit does not use it

    // Cycle counts spanning the MoE's 9.3 microsecond mean kernel, plus an empty kernel.
    const long long sweep[] = { 0, 1500, 7500, 14000, 30000 };
    const int nsweep = (int) (sizeof(sweep) / sizeof(sweep[0]));

    std::printf("%-8s %10s %14s %14s\n", "cycles", "approx us", "stream us/disp", "graph us/disp");
    std::vector<double> xs, ys_s, ys_g;
    for (int i = 0; i < nsweep; ++i) {
        const long long cycles = sweep[i];
        (void) time_stream(out, cycles, 200, block, 1);          // warm this shape
        const double st = best(time_stream, out, cycles, n, block, 1, passes);
        // The graph arm captures one node per dispatch, so its node count is capped: instantiation of
        // a very large graph is itself slow here, and on this board it is the operation that fails at
        // deep context on the 14B models.
        const int ng = std::min(n, 2000);
        const double gr = best(time_graph, out, cycles, ng, block, 1, passes);
        std::printf("%-8lld %10.2f %14.3f %14.3f\n", cycles, cycles / CYC_NOMINAL, st, gr);
        if (cycles > 0) { xs.push_back((double) cycles); ys_s.push_back(st); ys_g.push_back(gr); }
    }

    // Least squares on the non-empty points. The derived clock is a check on the whole measurement:
    // if it does not come back at the clock the board is pinned to, something moved during the run.
    auto fit = [&](const std::vector<double> & y, const char * name) {
        const int m = (int) xs.size();
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < m; ++i) { sx += xs[i]; sy += y[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * y[i]; }
        const double slope = (m * sxy - sx * sy) / (m * sxx - sx * sx);
        const double inter = (sy - slope * sx) / m;
        double worst = 0;
        for (int i = 0; i < m; ++i) worst = std::max(worst, std::fabs(y[i] - (slope * xs[i] + inter)));
        std::printf("%-8s derived clock %7.1f MHz   dispatch floor %6.3f us   max residual %4.0f ns\n",
                    name, 1.0 / slope, inter, worst * 1000.0);
    };
    std::printf("\n");
    fit(ys_s, "stream");
    fit(ys_g, "graph");

    CHECK(hipFree(out));
    return 0;
}
