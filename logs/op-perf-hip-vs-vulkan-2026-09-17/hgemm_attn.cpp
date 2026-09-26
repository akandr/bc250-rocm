// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// hgemm_attn: rocBLAS fp16 GEMM alone, at the two attention shapes the -fa off prefill graph of
// qwen2.5-1.5B issues (logs/op-perf-hip-vs-vulkan-2026-09-17), so the per-op MUL_MAT times can be
// split between the library call and llama.cpp's surrounding work (f32->f16 conversion of the
// activations, the f16->f32 result copy). The op-level numbers those compare against, HIP:
//   KQ  f16[4096,128,2] x f32[4096,2048,12] -> per head m=4096 n=2048 k=128, 12 heads: 8087 us
//   KQV f16[128,4096,2] x f32[4096,2048,12] -> per head m=128  n=2048 k=4096, 12 heads: 8208 us
// (Vulkan: 4399 and 5805 us.) Both are strided-batched over the 12 heads; K and V are shared by
// pairs of heads (GQA 6) which the strided batch reproduces with stride 0 on A within a pair only
// approximately, so plain batch 12 with distinct A is used: that is an upper bound on the work.
//
// usage: hgemm_attn [comp:16|32] [reps]
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
struct Shape { int m,n,k,batch; const char* tag; };
int main(int argc,char**argv){
    setbuf(stdout,NULL);
    int comp=argc>1?atoi(argv[1]):32;
    int reps=argc>2?atoi(argv[2]):20;
    rocblas_datatype ct=(comp==32)?rocblas_datatype_f32_r:rocblas_datatype_f16_r;
    Shape shapes[]={ {4096,2048,128,12,"KQ"}, {128,2048,4096,12,"KQV"},
                     {4096,2048,128,1,"KQ/head"}, {128,2048,4096,1,"KQV/head"} };
    rocblas_handle h; rocblas_create_handle(&h);
    float a32=1.f,b32=0.f; __half a16=__float2half(1.f),b16=__float2half(0.f);
    const void*ap=(comp==32)?(void*)&a32:(void*)&a16,*bp=(comp==32)?(void*)&b32:(void*)&b16;
    hipDeviceProp_t p; hipGetDeviceProperties(&p,0); printf("=== hgemm_attn %s comp=f%d reps=%d\n",p.gcnArchName,comp,reps);
    for(auto&s:shapes){
        size_t sa=(size_t)s.m*s.k, sb=(size_t)s.k*s.n, sc=(size_t)s.m*s.n;
        __half *dA,*dB,*dC,*dD;
        if(hipMalloc(&dA,sa*2*s.batch)||hipMalloc(&dB,sb*2*s.batch)||hipMalloc(&dC,sc*2*s.batch)||hipMalloc(&dD,sc*2*s.batch)){printf("%s malloc fail\n",s.tag);return 2;}
        __half* hA=(__half*)malloc(sa*2*s.batch); srand(3);
        for(size_t i=0;i<sa*s.batch;i++)hA[i]=__float2half((rand()%100-50)/100.f);
        hipMemcpy(dA,hA,sa*2*s.batch,hipMemcpyHostToDevice); free(hA);
        hA=(__half*)malloc(sb*2*s.batch); for(size_t i=0;i<sb*s.batch;i++)hA[i]=__float2half((rand()%100-50)/100.f);
        hipMemcpy(dB,hA,sb*2*s.batch,hipMemcpyHostToDevice); free(hA);
        auto gemm=[&](){ return rocblas_gemm_strided_batched_ex(h,rocblas_operation_none,rocblas_operation_none,
            s.m,s.n,s.k, ap, dA,rocblas_datatype_f16_r,s.m,sa, dB,rocblas_datatype_f16_r,s.k,sb, bp,
            dC,rocblas_datatype_f16_r,s.m,sc, dD,rocblas_datatype_f16_r,s.m,sc, s.batch,
            ct, rocblas_gemm_algo_standard,0,0); };
        rocblas_status st=gemm(); hipDeviceSynchronize();
        if(st!=rocblas_status_success){printf("%-9s status=%s FAIL\n",s.tag,rocblas_status_to_string(st));continue;}
        auto t0=std::chrono::steady_clock::now();
        for(int r=0;r<reps;r++) gemm();
        hipDeviceSynchronize();
        double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()/reps;
        double gf=2.0*s.m*s.n*s.k*s.batch/sec/1e9;
        printf("%-9s m=%4d n=%4d k=%4d batch=%2d  %8.1f us  %7.1f GFLOP/s\n",s.tag,s.m,s.n,s.k,s.batch,sec*1e6,gf);
        hipFree(dA);hipFree(dB);hipFree(dC);hipFree(dD);
    }
    return 0;
}
