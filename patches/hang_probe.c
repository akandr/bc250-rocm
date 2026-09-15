// hang_probe.c - a compute queue that never yields, for testing KFD queue recovery.
//
// Launches one GPU thread that loops until a value it reads changes, which it never
// does, so the queue cannot finish its dispatch. When the hardware scheduler next
// rebuilds the runlist (any other process creating or destroying a queue does it),
// the preemption of this queue should time out and KFD should either reset the
// queue, reset the GPU, or give up, which is what the test observes.
//
// Usage: hang_probe [limit]   loop iterations (default 1e15, effectively forever);
//                             a small limit checks that the kernel really runs
//
// Build:
//   /usr/lib64/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1013 -O2 \
//       hang_probe.c -o hang_probe
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
__global__ void spin(int *flag, long *counter, long limit){
    /* A counting loop gets folded into a closed form and returns at once (seen on the
     * first two versions of this probe). A linear-congruential recurrence cannot be, and
     * is the same loop seq_probe uses to occupy the GPU. */
    unsigned int x = 1;
    for (long i = 0; i < limit; i++)
        x = x * 1103515245u + 12345u;
    counter[0] = (long)x + flag[0];
}
int main(int argc, char **argv){
    setbuf(stdout, NULL);
    int *flag; long *counter;
    if (hipMalloc(&flag, sizeof(int)) != hipSuccess) { printf("malloc failed\n"); return 2; }
    if (hipMalloc(&counter, sizeof(long)) != hipSuccess) { printf("malloc failed\n"); return 2; }
    hipMemset(flag, 0, sizeof(int));
    printf("hang_probe: pid %d launching a kernel that never returns\n", getpid());
    hipMemset(counter, 0, sizeof(long));
    long limit = argc > 1 ? atol(argv[1]) : 1000000000000000L;
    hipLaunchKernelGGL(spin, dim3(1), dim3(1), 0, 0, flag, counter, limit);
    hipError_t le = hipGetLastError();
    printf("hang_probe: launch status %d (%s)\n", (int)le, hipGetErrorString(le));
    time_t t0 = time(NULL);
    hipError_t e = hipDeviceSynchronize();
    printf("hang_probe: synchronize returned %d (%s) after %ld s\n", (int)e, hipGetErrorString(e),
           (long)(time(NULL) - t0));
    long c = -1;
    hipMemcpy(&c, counter, sizeof(long), hipMemcpyDeviceToHost);
    printf("hang_probe: counter reached %ld (limit %ld)\n", c, limit);
    return e == hipSuccess ? 0 : 1;
}
