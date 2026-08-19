// A single elementwise kernel compiled for gfx1013, exposed with a plain C
// entry point so it can be called on device pointers owned by the stock
// PyTorch wheel without going through torch's extension build system.
//
// Build (verified 2026-08-17). Without --offload-arch=gfx1013 a HIP source
// still compiles and links, but its kernels never run on this board, which
// reads as a clean pass rather than a failure:
//   /usr/lib64/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1013 -O2 -fPIC \
//       -shared bc250_ext.hip -o bc250_ext.so
#include <hip/hip_runtime.h>

__global__ void add_scale_kernel(const float* a, const float* b, float* out, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (a[i] + b[i]) * s;
}

extern "C" int bc250_add_scale(const void* a, const void* b, void* out, float s, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(add_scale_kernel, dim3(blocks), dim3(threads), 0, 0,
                       (const float*) a, (const float*) b, (float*) out, s, n);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return (int) err;
    return (int) hipDeviceSynchronize();
}
