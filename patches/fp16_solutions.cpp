// Which Tensile solution services the failing fp16 GEMM, and is any of them wrong?
//
// The zeroed fp16 GEMM vanishes when the board is driven as gfx1010 with a
// different rocBLAS, which points at the gfx1013 Tensile kernels rather than at
// the fp16 path. rocBLAS can enumerate the solutions it would consider for a
// problem and run a chosen one, so the shape llama.cpp fails on can be tried
// through each in turn and checked against a CPU reference.
//
// Shape taken from the dispatch trace, not guessed:
//   gemm_ex, transA=T transB=N, M=1024 N=512 K=4096,
//   a/b/c/d all f16_r, compute f16_r, alpha=1 beta=0, lda=4096 ldb=4096 ldc=1024
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define CK(x) do { auto s_=(x); if(s_!=rocblas_status_success){printf("rocblas error %d at line %d\n",(int)s_,__LINE__); exit(1);} } while(0)
#define HK(x) do { auto e_=(x); if(e_!=hipSuccess){printf("hip error at line %d\n",__LINE__); exit(1);} } while(0)

int main(int argc, char** argv) {
    const int M=1024, N=512, K=4096;
    rocblas_handle h; CK(rocblas_create_handle(&h));

    std::vector<_Float16> A((size_t)K*M), B((size_t)K*N), C((size_t)M*N);
    srand(1234);
    for (auto &v : A) v = (_Float16)((rand()%200-100)/400.0f);
    for (auto &v : B) v = (_Float16)((rand()%200-100)/400.0f);

    void *dA,*dB,*dC;
    HK(hipMalloc(&dA, A.size()*2)); HK(hipMalloc(&dB, B.size()*2)); HK(hipMalloc(&dC, C.size()*2));
    HK(hipMemcpy(dA, A.data(), A.size()*2, hipMemcpyHostToDevice));
    HK(hipMemcpy(dB, B.data(), B.size()*2, hipMemcpyHostToDevice));

    // CPU reference for a few sampled output elements (full reference is too slow)
    const int nsample = 16;
    std::vector<int> si(nsample), sj(nsample); std::vector<float> ref(nsample);
    for (int s=0;s<nsample;s++){
        si[s]=rand()%M; sj[s]=rand()%N;
        float acc=0; for(int k=0;k<K;k++) acc += (float)A[(size_t)si[s]*K+k]*(float)B[(size_t)sj[s]*K+k];
        ref[s]=acc;
    }

    const _Float16 alpha=(_Float16)1.0f, beta=(_Float16)0.0f;
    rocblas_int nsol=0;
    CK(rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none,
        M,N,K,&alpha,dA,rocblas_datatype_f16_r,K,dB,rocblas_datatype_f16_r,K,&beta,
        dC,rocblas_datatype_f16_r,M,dC,rocblas_datatype_f16_r,M,
        rocblas_datatype_f16_r, rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none,
        nullptr,&nsol));
    printf("solutions available for this shape: %d\n", nsol);
    std::vector<rocblas_int> sols(nsol>0?nsol:1);
    if(nsol>0) CK(rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none,
        M,N,K,&alpha,dA,rocblas_datatype_f16_r,K,dB,rocblas_datatype_f16_r,K,&beta,
        dC,rocblas_datatype_f16_r,M,dC,rocblas_datatype_f16_r,M,
        rocblas_datatype_f16_r, rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none,
        sols.data(),&nsol));

    int reps = (argc>1)?atoi(argv[1]):1;
    int bad=0;
    for (int s=0; s<nsol; s++) {
        int worst_zero=0; float worst_rel=0;
        for (int r=0;r<reps;r++) {
            HK(hipMemset(dC, 0xFF, C.size()*2));
            auto st = rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none,
                M,N,K,&alpha,dA,rocblas_datatype_f16_r,K,dB,rocblas_datatype_f16_r,K,&beta,
                dC,rocblas_datatype_f16_r,M,dC,rocblas_datatype_f16_r,M,
                rocblas_datatype_f16_r, rocblas_gemm_algo_solution_index, sols[s], rocblas_gemm_flags_none);
            if (st!=rocblas_status_success) { printf("  solution %d: status %d\n", sols[s], (int)st); goto next; }
            HK(hipDeviceSynchronize());
            HK(hipMemcpy(C.data(), dC, C.size()*2, hipMemcpyDeviceToHost));
            int zeros=0; for (size_t i=0;i<C.size();i++) if ((float)C[i]==0.0f) zeros++;
            if (zeros>worst_zero) worst_zero=zeros;
            for (int t=0;t<nsample;t++){
                float got=(float)C[(size_t)sj[t]*M+si[t]];
                float rel=fabsf(got-ref[t])/(fabsf(ref[t])+1e-6f);
                if (rel>worst_rel) worst_rel=rel;
            }
        }
        if (worst_zero > (int)C.size()/2 || worst_rel > 0.05f) { bad++;
            printf("  solution %-6d SUSPECT zeros=%d/%zu worst_rel=%.4f\n", sols[s], worst_zero, C.size(), worst_rel);
        }
        next:;
    }
    printf("checked %d solutions, %d suspect\n", nsol, bad);
    return 0;
}
