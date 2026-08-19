// membw.cpp - achievable device memory read bandwidth on the BC-250.
// Written to check whether decode rates in llama.cpp are bandwidth-bound: at
// Q8_0 the 8B model implies reading about 300 GiB/s of weights per second,
// which would only be possible if the board can actually sustain that.
//
//   /usr/lib64/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1013 -O2 \
//       membw.cpp -o membw
//   HSA_ENABLE_SDMA=0 ./membw [buffer_MiB] [iters]
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

__global__ void stream_read(const float4 * __restrict__ p, size_t n4, float * __restrict__ sink) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (; i < n4; i += stride) {
        float4 v = p[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1e30f) sink[0] = acc.x + acc.y + acc.z + acc.w;  // keep the loads
}

int main(int argc, char ** argv) {
    const size_t mib = (argc > 1) ? atoll(argv[1]) : 2048;
    const int iters = (argc > 2) ? atoi(argv[2]) : 20;
    const size_t bytes = mib << 20;
    const size_t n4 = bytes / sizeof(float4);

    void * buf; float * sink;
    CK(hipMalloc(&buf, bytes));
    CK(hipMalloc(&sink, sizeof(float)));
    CK(hipMemset(buf, 0x3f, bytes));

    hipEvent_t a, b;
    CK(hipEventCreate(&a)); CK(hipEventCreate(&b));

    const int blocks = 512, threads = 256;
    hipLaunchKernelGGL(stream_read, dim3(blocks), dim3(threads), 0, 0, (const float4 *) buf, n4, sink);
    CK(hipDeviceSynchronize());

    double best = 0.0;
    for (int it = 0; it < iters; it++) {
        CK(hipEventRecord(a));
        hipLaunchKernelGGL(stream_read, dim3(blocks), dim3(threads), 0, 0, (const float4 *) buf, n4, sink);
        CK(hipEventRecord(b));
        CK(hipEventSynchronize(b));
        float ms = 0.f; CK(hipEventElapsedTime(&ms, a, b));
        double gbs = (double) bytes / (ms * 1e-3) / 1e9;
        if (gbs > best) best = gbs;
    }
    printf("buffer %zu MiB, best read bandwidth %.1f GB/s (%.1f GiB/s)\n",
           mib, best, best / 1.073741824);
    return 0;
}
