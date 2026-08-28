// Fault from inside a process that is itself holding the GPU busy.
//
// A deliberate out-of-bounds write reproduces the fault signature of the events
// that kill this board, but not the escalation: on an idle GPU, and even
// alongside a separate busy process, the faulting process dies and the driver
// never needs to reset anything. In the real events the faulting process was
// the one running the heavy workload, so recovering from its fault means
// preempting its own queues, and it is that preemption which times out.
//
// This puts the fault inside such a process: a long stream of work in flight on
// several streams, and then the bad write, so the queues that must be preempted
// belong to the faulting context.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define NSTREAM 4

__global__ void burn(float *p, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = p[i];
    for (int k = 0; k < iters; k++) v = v * 1.0000001f + 0.0000001f;
    p[i] = v;
}

__global__ void stomp(float *p, long long off) {
    p[off + threadIdx.x] = 1.0f;
}

int main(int argc, char **argv) {
    long long off = (argc > 1) ? atoll(argv[1]) : (1LL << 30);
    int iters     = (argc > 2) ? atoi(argv[2]) : 200000;
    const int n   = 1 << 22;               // 16 MiB of floats

    float *d[NSTREAM];
    hipStream_t s[NSTREAM];
    for (int i = 0; i < NSTREAM; i++) {
        if (hipMalloc(&d[i], (size_t)n * sizeof(float)) != hipSuccess) {
            printf("alloc %d failed\n", i); return 2;
        }
        hipMemset(d[i], 0, (size_t)n * sizeof(float));
        hipStreamCreate(&s[i]);
    }
    printf("allocated %d buffers, queueing work\n", NSTREAM); fflush(stdout);

    // fill the queues with work that will still be running when the fault lands
    for (int round = 0; round < 8; round++)
        for (int i = 0; i < NSTREAM; i++)
            hipLaunchKernelGGL(burn, dim3(n/256), dim3(256), 0, s[i], d[i], n, iters);

    printf("queued, now faulting at element offset %lld\n", off); fflush(stdout);
    hipLaunchKernelGGL(stomp, dim3(1), dim3(64), 0, s[0], d[0], off);

    hipError_t e = hipDeviceSynchronize();
    printf("hipDeviceSynchronize returned %d (%s)\n", (int)e, hipGetErrorString(e));
    return e == hipSuccess ? 0 : 1;
}
