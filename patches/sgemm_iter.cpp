#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
  int N=argc>1?atoi(argv[1]):8192;
  int iters=argc>2?atoi(argv[2]):100;
  size_t sz=(size_t)N*N;
  std::vector<float> A(sz),B(sz);
  for(size_t i=0;i<sz;i++){A[i]=(float)((i%7)+1);B[i]=(float)((i%5)+1);}
  float *dA,*dB,*dC;
  if(hipMalloc(&dA,sz*4)||hipMalloc(&dB,sz*4)||hipMalloc(&dC,sz*4)){printf("hipMalloc fail\n");return 2;}
  hipMemcpy(dA,A.data(),sz*4,hipMemcpyHostToDevice);
  hipMemcpy(dB,B.data(),sz*4,hipMemcpyHostToDevice);
  rocblas_handle h; rocblas_create_handle(&h);
  float alpha=1.0f,beta=0.0f;
  int i2=N/3,j2=N/2;
  auto ref=[&](int i,int j){double s=0;for(int k=0;k<N;k++)s+=(double)A[i+(size_t)k*N]*B[k+(size_t)j*N];return (float)s;};
  float r0=ref(0,0), r1=ref(i2,j2);
  int nwrong=0;
  printf("N=%d iters=%d start\n",N,iters); fflush(stdout);
  for(int it=0; it<iters; it++){
    auto t0=std::chrono::high_resolution_clock::now();
    rocblas_sgemm(h,rocblas_operation_none,rocblas_operation_none,N,N,N,&alpha,dA,N,dB,N,&beta,dC,N);
    hipError_t he=hipDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    float g0,g1; hipMemcpy(&g0,dC,4,hipMemcpyDeviceToHost); hipMemcpy(&g1,dC+(i2+(size_t)j2*N),4,hipMemcpyDeviceToHost);
    bool ok=fabs(g0-r0)<1e-2*fabs(r0)+1 && fabs(g1-r1)<1e-2*fabs(r1)+1;
    if(!ok) nwrong++;
    printf("iter %3d sync=%d %6.1fms %s g0=%.0f/%.0f nwrong=%d\n",it,(int)he,ms,ok?"OK":"WRONG",g0,r0,nwrong); fflush(stdout);
  }
  printf("DONE iters=%d nwrong=%d\n",iters,nwrong); fflush(stdout);
  return nwrong>0?1:0;
}
