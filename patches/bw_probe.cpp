#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
// TCP vector-read bandwidth stress: stream-read a large buffer repeatedly.
__global__ void streamread(const float* in, float* out, size_t n, int reps){
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x*blockDim.x;
  float acc=0.f;
  for(int r=0;r<reps;r++) for(size_t j=i;j<n;j+=stride) acc += in[j];
  if(i<65536) out[i]=acc;
}
int main(int argc,char**argv){
  size_t mb = argc>1?atol(argv[1]):2048;   // buffer MB
  int reps  = argc>2?atoi(argv[2]):8;
  int iters = argc>3?atoi(argv[3]):10;
  size_t n = mb*1024ull*1024ull/sizeof(float);
  float *d_in=0,*d_out=0;
  if(hipMalloc(&d_in,n*sizeof(float))!=hipSuccess){printf("malloc fail\n");return 2;}
  hipMalloc(&d_out,65536*sizeof(float));
  hipMemset(d_in,1,n*sizeof(float));
  hipDeviceSynchronize();
  printf("=== bw_probe %s buf=%zuMB reps=%d iters=%d n=%zu ===\n","gfx1013",mb,reps,iters,n);
  dim3 grid(65536),block(256);
  setbuf(stdout,NULL); int fails=0;
  for(int it=0;it<iters;it++){
    hipLaunchKernelGGL(streamread,grid,block,0,0,d_in,d_out,n,reps);
    hipError_t e=hipDeviceSynchronize();
    double gb=(double)n*reps*4.0/1e9;
    printf("iter %d: %s  (%.1f GB read)\n",it,hipGetErrorString(e),gb);
    if(e!=hipSuccess){fails++; if(fails>=2)break;}
  }
  printf("RESULT fails=%d/%d\n",fails,iters);
  return fails?1:0;
}
