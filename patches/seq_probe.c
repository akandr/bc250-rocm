// seq_probe.c - sequence reproducer: alloc/dispatch/free cycles with varying
// sizes in one process; verifies each generation. Distinguishes VA-reuse
// stale-translation corruption from tail/partial-threadgroup effects.
// Usage: seq_probe <inner> <n1> <n2> [n3...]
//   inner  arithmetic iterations per element (6000 gives a few seconds at 8.4M)
//   nN     element count for each generation, e.g. 8388608 repeated
// Prints "RESULT bad_gens=N"; zero means every generation verified.
//
// Build:
//   /usr/lib64/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1013 -O2 \
//       seq_probe.c -o seq_probe
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(x) do{ hipError_t e_=(x); if(e_!=hipSuccess){ \
    printf("HIP FAIL %s: %s\n",#x,hipGetErrorString(e_)); return 2; } }while(0)
__global__ void work(int *d, long n, int inner){
    long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=n) return;
    int x = d[i];
    for(int k=0;k<inner;k++) x = x*1103515245 + 12345;
    d[i] = x + 1;
}
int main(int argc,char**argv){
    setbuf(stdout,NULL);
    int inner = atoi(argv[1]);
    int ref=0; for(int k=0;k<inner;k++) ref=ref*1103515245+12345; ref+=1;
    int bad=0;
    for(int a=2;a<argc;a++){
        long n = atol(argv[a]);
        int *d; CK(hipMalloc(&d, n*4));
        CK(hipMemset(d, 0, n*4));
        long nblk=(n+255)/256;
        hipLaunchKernelGGL(work, dim3((unsigned)nblk), dim3(256), 0,0, d, n, inner);
        CK(hipDeviceSynchronize());
        int *h=(int*)malloc(n*4);
        CK(hipMemcpy(h,d,n*4,hipMemcpyDeviceToHost));
        long wrong=0, zeros=0, first=-1;
        for(long i=0;i<n;i++) if(h[i]!=ref){ wrong++; if(h[i]==0) zeros++; if(first<0) first=i; }
        printf("gen%d n=%ld tail=%ld va=%p wrong=%ld zeros=%ld first=%ld %s\n",
               a-1, n, n%256, (void*)d, wrong, zeros, first, wrong?"WRONG":"ok");
        if(wrong) bad++;
        free(h); CK(hipFree(d));
    }
    printf("RESULT bad_gens=%d\n", bad);
    return bad?3:0;
}
