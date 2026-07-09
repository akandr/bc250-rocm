// ocl_vecadd.c - minimal OpenCL compute test.
// Proves that GPU compute works via the graphics/universal queue on the BC-250
// (RustiCL / radeonsi), as opposed to ROCm/HIP which uses the broken MEC compute queue.
//
// Build: cc ocl_vecadd.c -o ocl_vecadd -lOpenCL
// Run:   RUSTICL_ENABLE=radeonsi ./ocl_vecadd
//
// Computes c[i] = a[i] + b[i] on the GPU for N elements and verifies the result.

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 20)  // 1M elements
#define CK(x) do{ cl_int _e=(x); if(_e){printf("FAIL %s: %d\n",#x,_e); return 2;} }while(0)

static const char *SRC =
"__kernel void vadd(__global const float*a,__global const float*b,__global float*c){"
"  int i=get_global_id(0); c[i]=a[i]+b[i];"
"}";

int main(void){
    cl_platform_id plat; cl_uint np=0;
    CK(clGetPlatformIDs(1,&plat,&np));
    if(!np){ printf("no OpenCL platform\n"); return 1; }
    char pname[128]={0}; clGetPlatformInfo(plat,CL_PLATFORM_NAME,sizeof pname,pname,0);

    cl_device_id dev; cl_uint nd=0;
    CK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,&nd));
    if(!nd){ printf("no GPU device\n"); return 1; }
    char dname[256]={0}; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof dname,dname,0);
    printf("platform: %s\ndevice:   %s\n", pname, dname);

    cl_int e;
    cl_context ctx=clCreateContext(0,1,&dev,0,0,&e); CK(e);
    cl_command_queue q=clCreateCommandQueueWithProperties(ctx,dev,0,&e); CK(e);
    cl_program prog=clCreateProgramWithSource(ctx,1,&SRC,0,&e); CK(e);
    e=clBuildProgram(prog,1,&dev,0,0,0);
    if(e){ char log[4096]={0}; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,0); printf("build FAIL:\n%s\n",log); return 2; }
    cl_kernel k=clCreateKernel(prog,"vadd",&e); CK(e);

    size_t bytes=(size_t)N*sizeof(float);
    float *a=malloc(bytes),*b=malloc(bytes),*c=malloc(bytes);
    for(int i=0;i<N;i++){ a[i]=(float)i; b[i]=2.0f*(float)i; }
    cl_mem da=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,bytes,a,&e); CK(e);
    cl_mem db=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,bytes,b,&e); CK(e);
    cl_mem dc=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,bytes,0,&e); CK(e);
    CK(clSetKernelArg(k,0,sizeof da,&da));
    CK(clSetKernelArg(k,1,sizeof db,&db));
    CK(clSetKernelArg(k,2,sizeof dc,&dc));

    size_t gs=N;
    CK(clEnqueueNDRangeKernel(q,k,1,0,&gs,0,0,0,0));
    CK(clFinish(q));
    CK(clEnqueueReadBuffer(q,dc,CL_TRUE,0,bytes,c,0,0,0));

    long bad=0;
    for(int i=0;i<N;i++){ float exp=3.0f*(float)i; if(c[i]!=exp) bad++; }
    printf("N=%d  c[1]=%.1f (expect 3.0)  c[1000]=%.1f (expect 3000.0)  mismatches=%ld\n",
           N, c[1], c[1000], bad);
    printf(bad==0 ? "OPENCL_COMPUTE_OK\n" : "OPENCL_COMPUTE_WRONG\n");
    return bad==0?0:3;
}
