// loadmimic.c: replicate llama.cpp's --no-mmap async upload loop and time it,
// varying the chunk size. Written to test whether per-chunk overhead explains the
// model-load slowness. Result: it does not. This microbenchmark is usually fast and
// does not reproduce the real llama slowness, which points at the full compute path
// and the compute-queue defect rather than the upload primitive (see README).
//
// Pattern (matches llama-model-loader.cpp load_all_data):
//   - one big device buffer (model), N chunks of size S uploaded to advancing offsets
//   - nbuf pinned staging buffers, round-robin, one event each
//   - per chunk: event_synchronize(events[b]); memcpy host->host (stand-in for file read);
//                hipMemcpyAsync staging->device on a stream; hipEventRecord
//   - sync method selectable: sync (hipEventSynchronize) or query (hipEventQuery spin)
//
// Usage: loadmimic <total_mb> <chunk_kb> <nbuf> <sync|query> [hold]
//   total_mb: total bytes to "load" (e.g. 4096)
//   chunk_kb: per-chunk size in KB (e.g. 6144 for 6MB tensors, 65536 for 64MB)
//   nbuf:     staging buffers / events (llama uses 4)
//
// Prints per-chunk wait time distribution and totals, so we can compare 6MB vs 64MB chunks.

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("FAIL %s: %s\n",#x,hipGetErrorString(e));fflush(stdout);exit(1);} }while(0)
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}

int main(int argc,char**argv){
    setbuf(stdout,NULL);
    size_t total_mb = argc>1?atol(argv[1]):4096;
    size_t chunk_kb = argc>2?atol(argv[2]):6144;   // default 6 MB
    int nbuf        = argc>3?atoi(argv[3]):4;
    const char*mode = argc>4?argv[4]:"query";
    int hold        = argc>5 && !strcmp(argv[5],"hold");
    const char*fpath= (argc>6 && argv[6][0]=='/')?argv[6]:NULL; // optional: read chunks from this file
    FILE*fp = fpath ? fopen(fpath,"rb") : NULL;
    if(fpath && !fp){ printf("cannot open %s\n",fpath); return 1; }
    if(fp) printf("reading chunks from file: %s\n", fpath);

    size_t total = total_mb<<20, chunk = chunk_kb<<10;
    long nch = (long)(total/chunk);
    printf("=== loadmimic total=%zuMB chunk=%zuKB nbuf=%d mode=%s -> %ld chunks ===\n",
        total_mb, chunk_kb, nbuf, mode, nch);
    printf("HSA_ENABLE_INTERRUPT=%s HSA_ENABLE_SDMA=%s\n",
        getenv("HSA_ENABLE_INTERRUPT")?:"(unset)", getenv("HSA_ENABLE_SDMA")?:"(unset)");

    char *dev=NULL; CK(hipMalloc((void**)&dev, total));
    hipStream_t s; CK(hipStreamCreate(&s));
    char *host[16]; hipEvent_t ev[16]; int live[16];
    for(int i=0;i<nbuf;i++){
        CK(hipHostMalloc((void**)&host[i], chunk, 0)); memset(host[i],0xAB,chunk);
        CK(hipEventCreateWithFlags(&ev[i], hipEventDisableTiming)); live[i]=0;
    }
    // src for the "file read" memcpy (plain heap, like reading from page cache)
    char *src = (char*)malloc(chunk); memset(src,0xCD,chunk);

    double t0=now_ms(), wtot=0, wmax=0; long slow=0;
    double hist[6]={0,0,0,0,0,0}; // <1ms,<10,<100,<1000,<5000,>=5000
    for(long i=0;i<nch;i++){
        int b=i%nbuf;
        double a=now_ms();
        if(live[b]){
            if(!strcmp(mode,"query")){ hipError_t st; while((st=hipEventQuery(ev[b]))==hipErrorNotReady){} if(st!=hipSuccess){printf("qerr\n");exit(1);} }
            else CK(hipEventSynchronize(ev[b]));
        }
        double w=now_ms()-a; wtot+=w; if(w>wmax)wmax=w; if(w>1000)slow++;
        hist[ w<1?0: w<10?1: w<100?2: w<1000?3: w<5000?4:5 ]++;
        // fill pinned staging: from file (faithful) or heap memcpy (isolated)
        if(fp){ if(fread(host[b],1,chunk,fp)<chunk){ fseek(fp,0,SEEK_SET); } }
        else memcpy(host[b], src, chunk);
        // async upload to advancing device offset
        CK(hipMemcpyAsync(dev + (size_t)i%(total/chunk)*chunk, host[b], chunk, hipMemcpyHostToDevice, s));
        CK(hipEventRecord(ev[b], s));
        live[b]=1;
        if(i<6 || i==nch-1 || w>1000) printf("chunk %4ld: wait=%.2f ms\n", i, w);
    }
    CK(hipStreamSynchronize(s));
    double tt=now_ms()-t0;
    printf("SUMMARY: %ld chunks in %.1f s | wait tot=%.1f ms max=%.1f ms avg=%.2f ms | >1s: %ld\n",
        nch, tt/1000.0, wtot, wmax, wtot/nch, slow);
    printf("hist ms  <1:%.0f <10:%.0f <100:%.0f <1000:%.0f <5000:%.0f >=5000:%.0f\n",
        hist[0],hist[1],hist[2],hist[3],hist[4],hist[5]);
    if(hold){ printf("HOLDING pid=%d\n",getpid()); fflush(stdout); pause(); }
    for(int i=0;i<nbuf;i++){ CK(hipEventDestroy(ev[i])); CK(hipHostFree(host[i])); }
    CK(hipStreamDestroy(s)); CK(hipFree(dev)); free(src);
    printf("DONE\n");
    return 0;
}
