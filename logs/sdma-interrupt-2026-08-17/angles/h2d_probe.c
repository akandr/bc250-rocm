#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char**argv){
  size_t n = argc>1?(size_t)atol(argv[1]):16384;
  void *dA; hipMalloc(&dA,1<<20);
  void *h = malloc(1<<20); memset(h,1,1<<20);
  hipMemcpy(dA,h,n,hipMemcpyHostToDevice);
  hipDeviceSynchronize(); printf("done %zu\n",n); return 0;
}
