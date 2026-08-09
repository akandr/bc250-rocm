// dgemm_iter.cpp - FP64 GEMM via native gfx1013 rocBLAS on the BC-250.
// Double precision is a ROCm-only capability on this board: the Vulkan
// path has no practical FP64 compute story.
// Build: /usr/lib64/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1013 \
//   dgemm_iter.cpp -o dgemm_iter -L/usr/lib64 -lamdhip64 -lrocblas
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
static double now_ms(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec*1e3+t.tv_nsec/1e6; }
#define CK(x) if((x)!=hipSuccess){printf("hip fail %s\n",#x);return 2;}
int main(int argc,char**argv){
  int N=argc>1?atoi(argv[1]):2048, iters=argc>2?atoi(argv[2]):10;
  size_t sz=(size_t)N*N;
  double *A,*B,*C; CK(hipMalloc(&A,sz*8)); CK(hipMalloc(&B,sz*8)); CK(hipMalloc(&C,sz*8));
  double *hA=(double*)malloc(sz*8), *hC=(double*)malloc(sz*8);
  srand(7); for(size_t i=0;i<sz;i++) hA[i]=(rand()%1000)/1000.0-0.5;
  CK(hipMemcpy(A,hA,sz*8,hipMemcpyHostToDevice));
  CK(hipMemcpy(B,hA,sz*8,hipMemcpyHostToDevice));
  rocblas_handle h; rocblas_create_handle(&h);
  const double one=1.0, zero=0.0;
  printf("DGEMM N=%d iters=%d (FP64)\n",N,iters);
  int nwrong=0;
  for(int it=0;it<iters;it++){
    double t0=now_ms();
    rocblas_dgemm(h,rocblas_operation_none,rocblas_operation_none,
                  N,N,N,&one,A,N,B,N,&zero,C,N);
    CK(hipDeviceSynchronize());
    double dt=now_ms()-t0;
    CK(hipMemcpy(hC,C,sz*8,hipMemcpyDeviceToHost));
    // spot-check one row against CPU
    int r=it%N; double ref=0; int bad=0;
    for(int k=0;k<N;k++) ref+=hA[r+ (size_t)k*N]*hA[k];   // C[r,0]
    if(std::abs(hC[r]-ref)>1e-9*N*std::abs(ref)+1e-9) bad=1;
    nwrong+=bad;
    printf("iter %2d %7.1f ms %6.1f GFLOP/s %s\n",it,dt,2.0*N*N*(double)N/dt/1e6,
           bad?"WRONG":"ok");
  }
  printf("DONE nwrong=%d\n",nwrong);
  return nwrong?3:0;
}
