// ocl_compute_probe.c - OpenCL port of compute_probe.c for the gfx1013 compute-queue test.
//
// Same kernel logic as compute_probe.c (LCG hash loop, CPU-verified), but dispatched
// through OpenCL so it can run on any OpenCL implementation: ROCm OpenCL (KFD user
// queues), the mining-era ROCr OpenCL 2.0 from amdgpu 21.50.2 in a container, or
// Mesa RustiCL (graphics queue) for comparison.
//
// Build: gcc -O2 ocl_compute_probe.c -o ocl_compute_probe -lOpenCL
// Run:   ./ocl_compute_probe [nblocks] [inner] [iters]   (workgroup size 256, like the HIP probe)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#define CK(x) do{ cl_int e_=(x); if(e_!=CL_SUCCESS){ \
    printf("CL FAIL %s: %d\n",#x,e_); return 2; } }while(0)

static const char *SRC =
"__kernel void work(__global int *d, long n, int inner){\n"
"    long i = get_global_id(0);\n"
"    if(i>=n) return;\n"
"    int x = d[i];\n"
"    for(int k=0;k<inner;k++) x = x*1103515245 + 12345;\n"
"    d[i] = x + 1;\n"
"}\n";

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e3 + t.tv_nsec/1e6; }

int main(int argc,char**argv){
    setbuf(stdout,NULL);
    long nblocks = argc>1 ? atol(argv[1]) : 65536;
    int  inner   = argc>2 ? atoi(argv[2]) : 6000;
    int  iters   = argc>3 ? atoi(argv[3]) : 8;
    long n = nblocks*256;

    cl_platform_id plats[8]; cl_uint np=0;
    CK(clGetPlatformIDs(8,plats,&np));
    cl_device_id dev=NULL; char name[256]="", pname[256]="";
    for(cl_uint p=0;p<np;p++){
        cl_device_id ds[8]; cl_uint nd=0;
        if(clGetDeviceIDs(plats[p],CL_DEVICE_TYPE_GPU,8,ds,&nd)!=CL_SUCCESS) continue;
        if(nd){ dev=ds[0];
            clGetPlatformInfo(plats[p],CL_PLATFORM_NAME,sizeof pname,pname,NULL);
            clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof name,name,NULL);
            break; }
    }
    if(!dev){ printf("no GPU OpenCL device\n"); return 2; }
    printf("=== ocl_compute_probe [%s] %s nblocks=%ld (%ld threads) inner=%d iters=%d ===\n",
           pname, name, nblocks, n, inner, iters);

    cl_int err;
    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err); CK(err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx,dev,NULL,&err);
    if(err){ q = clCreateCommandQueue(ctx,dev,0,&err); } CK(err);
    cl_program prog = clCreateProgramWithSource(ctx,1,&SRC,NULL,&err); CK(err);
    err = clBuildProgram(prog,1,&dev,"",NULL,NULL);
    if(err){ char log[8192]; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,NULL);
        printf("build fail: %s\n",log); return 2; }
    cl_kernel k = clCreateKernel(prog,"work",&err); CK(err);
    cl_mem d = clCreateBuffer(ctx,CL_MEM_READ_WRITE,n*sizeof(int),NULL,&err); CK(err);

    int ref=0; for(int j=0;j<inner;j++) ref = ref*1103515245 + 12345; ref += 1;
    int *h=(int*)malloc(n*sizeof(int));
    int zero=0;
    long total_wrong=0;
    for(int it=0; it<iters; it++){
        CK(clEnqueueFillBuffer(q,d,&zero,sizeof zero,0,n*sizeof(int),0,NULL,NULL));
        CK(clFinish(q));
        size_t gws=n, lws=256;
        CK(clSetKernelArg(k,0,sizeof d,&d));
        cl_long nn=n; CK(clSetKernelArg(k,1,sizeof nn,&nn));
        CK(clSetKernelArg(k,2,sizeof inner,&inner));
        double a=now_ms();
        CK(clEnqueueNDRangeKernel(q,k,1,NULL,&gws,&lws,0,NULL,NULL));
        CK(clFinish(q));
        double w=now_ms()-a;
        CK(clEnqueueReadBuffer(q,d,CL_TRUE,0,n*sizeof(int),h,0,NULL,NULL));
        long wrong=0, firstbad=-1;
        for(long i=0;i<n;i++) if(h[i]!=ref){ wrong++; if(firstbad<0) firstbad=i; }
        total_wrong += wrong;
        printf("iter %d: sync=%.1f ms  wrong=%ld/%ld  %s\n", it, w, wrong, n, wrong?"MISMATCH":"ok");
        if(wrong) printf("   first mismatch at index %ld: got %d, want %d\n",
                         firstbad, h[firstbad], ref);
    }
    printf("RESULT: total_wrong=%ld -> %s\n",
           total_wrong, total_wrong ? "COMPUTE PRODUCES WRONG RESULTS" : "ALL CORRECT");
    free(h);
    return total_wrong ? 3 : 0;
}
