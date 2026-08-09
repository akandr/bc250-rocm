// mandelbrot.cpp - custom HIP kernel demo on the BC-250 (native gfx1013).
// Renders a Mandelbrot set to PGM. Custom C++ GPU kernels are the ROCm-only
// path; Vulkan would need hand-written compute shaders + a host harness.
// Build: /usr/lib64/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1013 \
//   mandelbrot.cpp -o mandelbrot -L/usr/lib64 -lamdhip64
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define W 2048
#define H 1536
__global__ void mandel(unsigned char *img, int w, int h){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x>=w||y>=h) return;
  double cr = (x - w*0.7) * 3.0/w, ci = (y - h/2.0) * 3.0/w;
  double zr=0, zi=0; int it=0, maxit=1000;
  while(zr*zr+zi*zi<4.0 && it<maxit){ double t=zr*zr-zi*zi+cr; zi=2*zr*zi+ci; zr=t; it++; }
  img[(size_t)y*w+x] = it==maxit ? 0 : (unsigned char)(255.0*it/120 > 255 ? 255 : 255.0*it/120);
}
int main(){
  unsigned char *d; hipMalloc(&d,(size_t)W*H);
  dim3 b(16,16), g((W+15)/16,(H+15)/16);
  hipLaunchKernelGGL(mandel,g,b,0,0,d,W,H);
  hipDeviceSynchronize();
  unsigned char *hst=(unsigned char*)malloc((size_t)W*H);
  hipMemcpy(hst,d,(size_t)W*H,hipMemcpyDeviceToHost);
  FILE*f=fopen("mandelbrot.pgm","w");
  fprintf(f,"P5\n%d %d\n255\n",W,H); fwrite(hst,1,(size_t)W*H,f); fclose(f);
  printf("mandelbrot.pgm written (%dx%d, FP64 iteration on GPU)\n",W,H);
  return 0;
}
