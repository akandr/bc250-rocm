// rocblas_probe.c: minimal rocBLAS SGEMM to test gfx1013 kernel availability in isolation.
// Returns cleanly on error (no abort) so it won't wedge the board.
//
// Build: hipcc/g++ with -lrocblas -lamdhip64
// Run:   ./rocblas_probe [N]   (square NxN sgemm, default 512)

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>

#define HIPCK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIP FAIL %s: %s\n",#x,hipGetErrorString(e));return 2;} }while(0)
#define RBCK(x)  do{ rocblas_status s=(x); if(s!=rocblas_status_success){printf("ROCBLAS FAIL %s: status=%d (%s)\n",#x,s,rocblas_status_to_string(s));return 3;} }while(0)

int main(int argc,char**argv){
    int N = argc>1?atoi(argv[1]):512;
    printf("=== rocblas_probe N=%d ===\n", N);
    int dev=0; HIPCK(hipSetDevice(dev));
    hipDeviceProp_t p; HIPCK(hipGetDeviceProperties(&p,dev));
    printf("device: %s  gcnArch=%s\n", p.name, p.gcnArchName);

    size_t sz = (size_t)N*N;
    float *dA,*dB,*dC;
    HIPCK(hipMalloc(&dA,sz*4)); HIPCK(hipMalloc(&dB,sz*4)); HIPCK(hipMalloc(&dC,sz*4));
    float *hA=(float*)malloc(sz*4),*hB=(float*)malloc(sz*4);
    for(size_t i=0;i<sz;i++){hA[i]=1.0f; hB[i]=2.0f;}
    HIPCK(hipMemcpy(dA,hA,sz*4,hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dB,hB,sz*4,hipMemcpyHostToDevice));

    rocblas_handle h; RBCK(rocblas_create_handle(&h));
    float alpha=1.0f, beta=0.0f;
    printf("launching sgemm...\n"); fflush(stdout);
    RBCK(rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
                       N,N,N, &alpha, dA,N, dB,N, &beta, dC,N));
    HIPCK(hipDeviceSynchronize());

    float *hC=(float*)malloc(sz*4);
    HIPCK(hipMemcpy(hC,dC,sz*4,hipMemcpyDeviceToHost));
    // expected C[i] = sum_k 1*2 = 2*N
    printf("C[0]=%.1f expected=%.1f  %s\n", hC[0], 2.0f*N, (hC[0]==2.0f*N)?"CORRECT":"WRONG");
    printf("ROCBLAS_PROBE_OK\n");
    rocblas_destroy_handle(h);
    return 0;
}
