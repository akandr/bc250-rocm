// Provoke a GPU page fault deliberately, to test whether amdgpu.gpu_recovery=0
// keeps the board alive when one happens.
//
// The faults that have killed this board are gfxhub writes on vmid 8 from an
// address well outside anything mapped. This does the same thing on purpose: a
// kernel writes far past the end of its allocation, at an offset chosen to land
// outside any mapping the process owns.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void stomp(float *p, long long off) {
    p[off + threadIdx.x] = 1.0f;
}

int main(int argc, char **argv) {
    long long off = (argc > 1) ? atoll(argv[1]) : (1LL << 30);
    float *d = nullptr;
    if (hipMalloc(&d, 4096) != hipSuccess) { printf("alloc failed\n"); return 2; }
    printf("allocated %p, writing at element offset %lld\n", (void*)d, off);
    fflush(stdout);
    hipLaunchKernelGGL(stomp, dim3(1), dim3(64), 0, 0, d, off);
    hipError_t e = hipDeviceSynchronize();
    printf("hipDeviceSynchronize returned %d (%s)\n", (int)e, hipGetErrorString(e));
    return e == hipSuccess ? 0 : 1;
}
