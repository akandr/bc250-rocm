// compute_probe.c - minimal reproducer for the gfx1013 (BC-250) compute-queue defect.
//
// Dispatches a compute kernel of a chosen size, verifies the result against a CPU
// reference, and times each hipDeviceSynchronize. No rocBLAS, no llama.cpp, no
// HSA_OVERRIDE. A native gfx1013 build is enough because the kernel is compiled here.
//
// Build (native gfx1013):
//   /usr/lib64/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1013 -I/usr/include \
//       compute_probe.c -o compute_probe -L/usr/lib64 -lamdhip64
// Run:
//   HSA_ENABLE_SDMA=0 ./compute_probe [nblocks] [inner] [iters]
//     nblocks: grid size in blocks of 256 threads (default 65536 -> ~16.7M threads)
//     inner:   arithmetic iterations per thread (default 6000)
//     iters:   number of dispatches (default 8)
//
// Observed on the BC-250 (native gfx1013, no override, oberon 1500 MHz):
//   Small kernels (e.g. 4096 x 1000, ~1M threads): always CORRECT, fast, deterministic.
//   Large kernels (e.g. 65536 x 6000-8000, ~16.7M threads): intermittently produce WRONG
//   results (a small fraction of elements keep their pre-kernel value, i.e. those
//   wavefronts' outputs are dropped) AND degrade the compute queue so that following
//   dispatches slow down and eventually hang.
//
// This is the same compute-queue defect that Mesa/RADV works around by disabling the
// compute queue on gfx1013 (ac_gpu_info.c: "GFX1013 is known to have broken compute
// queue", num_queues = 0). Simple apps work; heavy/complex compute fails. LLM inference
// via ROCm/HIP hits it because it is all large GEMM and attention kernels.

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CK(x) do{ hipError_t e_=(x); if(e_!=hipSuccess){ \
    printf("HIP FAIL %s: %s\n",#x,hipGetErrorString(e_)); return 2; } }while(0)

__global__ void work(int *d, long n, int inner){
    long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=n) return;
    int x = d[i];
    for(int k=0;k<inner;k++) x = x*1103515245 + 12345;
    d[i] = x + 1;
}

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e3 + t.tv_nsec/1e6; }

int main(int argc,char**argv){
    setbuf(stdout,NULL);
    long nblocks = argc>1 ? atol(argv[1]) : 65536;
    int  inner   = argc>2 ? atoi(argv[2]) : 6000;
    int  iters   = argc>3 ? atoi(argv[3]) : 8;
    long n = nblocks*256;

    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p,0));
    printf("=== compute_probe %s nblocks=%ld (%ld threads) inner=%d iters=%d ===\n",
           p.gcnArchName, nblocks, n, inner, iters);
    printf("HSA_ENABLE_SDMA=%s\n", getenv("HSA_ENABLE_SDMA")?getenv("HSA_ENABLE_SDMA"):"(unset)");

    int *d; CK(hipMalloc(&d, n*sizeof(int)));

    // CPU reference: one pass of the kernel starting from 0.
    int ref=0; for(int k=0;k<inner;k++) ref = ref*1103515245 + 12345; ref += 1;

    long total_wrong=0;
    for(int it=0; it<iters; it++){
        CK(hipMemset(d, 0, n*sizeof(int)));
        double a = now_ms();
        hipLaunchKernelGGL(work, dim3(nblocks), dim3(256), 0, 0, d, n, inner);
        CK(hipDeviceSynchronize());
        double w = now_ms()-a;

        int *h=(int*)malloc(n*sizeof(int));
        CK(hipMemcpy(h, d, n*sizeof(int), hipMemcpyDeviceToHost));
        long wrong=0, firstbad=-1;
        for(long i=0;i<n;i++) if(h[i]!=ref){ wrong++; if(firstbad<0) firstbad=i; }
        total_wrong += wrong;
        printf("iter %d: sync=%.1f ms  wrong=%ld/%ld  %s\n",
               it, w, wrong, n, wrong ? "MISMATCH" : "ok");
        if(wrong) printf("   first mismatch at index %ld: got %d, want %d\n",
                         firstbad, h[firstbad], ref);
        free(h);
    }
    printf("RESULT: total_wrong=%ld -> %s\n",
           total_wrong, total_wrong ? "COMPUTE PRODUCES WRONG RESULTS" : "ALL CORRECT");
    return total_wrong ? 3 : 0;
}
