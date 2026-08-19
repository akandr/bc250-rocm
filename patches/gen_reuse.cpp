// gen_reuse.cpp - targeted reproducer for the stale-TLB-on-VA-reuse mechanism
// reported by GabriWar/bc250-rocm-working on the BC-250 (gfx1013).
//
// Mechanism under test: hipFree unmaps but the compute TLB is never
// invalidated (the PASID sweep matches zero VMIDs on gfx10 under HWS), so
// when hipMalloc reuses the same virtual address with new physical memory,
// the GPU keeps translating through the previous mapping. A read of the new
// buffer then returns the OLD generation's data.
//
// Method: N generations. Each generation hipMallocs a buffer, fills it with a
// generation-tagged pattern from a GPU kernel, syncs, reads back sample
// windows with hipMemcpy, and classifies every mismatch: does the value read
// match the pattern of an EARLIER generation that occupied the same VA
// (= stale translation, the smoking gun), or is it something else? Then
// hipFree, next generation. The allocator reliably hands back the same VA,
// which is exactly the reuse the mechanism needs.
//
// Build:
//   /usr/lib64/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1013 \
//       gen_reuse.cpp -o gen_reuse -L/usr/lib64 -lamdhip64
// Run:
//   HSA_ENABLE_SDMA=0 ./gen_reuse [mb] [generations] [fills_per_gen]
//
// Exit: 0 all correct, 3 mismatches seen (stale or other).

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do{ hipError_t e_=(x); if(e_!=hipSuccess){ \
    printf("HIP FAIL %s: %s\n",#x,hipGetErrorString(e_)); return 2; } }while(0)

// pattern(gen, i) tags every word with its generation and index
__device__ __host__ static inline unsigned pat(unsigned gen, unsigned long i){
    return (gen * 2654435761u) ^ (unsigned)(i * 40503u) ^ 0x5aa5c33cu;
}

__global__ void fill(unsigned *d, unsigned long n, unsigned gen){
    unsigned long i = (unsigned long)blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=n) return;
    d[i] = pat(gen, i);
}

int main(int argc, char **argv){
    setbuf(stdout, NULL);
    long mb    = argc>1 ? atol(argv[1]) : 256;
    int  gens  = argc>2 ? atoi(argv[2]) : 24;
    int  fills = argc>3 ? atoi(argv[3]) : 2;
    unsigned long n = mb*1024ul*1024ul/4ul;
    const int NWIN = 32;                  // sample windows per generation
    const unsigned long WIN = 1024;      // words per window

    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p,0));
    printf("=== gen_reuse %s mb=%ld gens=%d fills=%d ===\n", p.gcnArchName, mb, gens, fills);

    // VA history: which generation last owned which VA
    void   *va_hist[4096]; unsigned gen_hist[4096]; int nhist=0;

    unsigned *h = (unsigned*)malloc(WIN*4);
    long total_bad=0, total_stale=0, va_reuses=0;

    for(int g=1; g<=gens; g++){
        unsigned *d;
        CK(hipMalloc(&d, n*4ul));

        // is this VA a reuse of an earlier generation's?
        int prev_gen = -1;
        for(int k=nhist-1; k>=0; k--)
            if(va_hist[k]==(void*)d){ prev_gen = gen_hist[k]; break; }
        if(prev_gen>0) va_reuses++;

        unsigned long blocks = (n+255)/256;
        for(int f=0; f<fills; f++){
            hipLaunchKernelGGL(fill, dim3((unsigned)blocks), dim3(256), 0, 0, d, n, (unsigned)g);
            CK(hipDeviceSynchronize());
        }

        long bad=0, stale=0; unsigned long firstbad=~0ul; unsigned firstgot=0;
        for(int w=0; w<NWIN; w++){
            unsigned long off = (n/NWIN)*w;
            if(off+WIN>n) break;
            CK(hipMemcpy(h, d+off, WIN*4, hipMemcpyDeviceToHost));
            for(unsigned long i=0;i<WIN;i++){
                unsigned want = pat(g, off+i);
                if(h[i]!=want){
                    bad++;
                    if(firstbad==~0ul){ firstbad=off+i; firstgot=h[i]; }
                    // does it match an earlier generation's pattern at this index?
                    for(int pg=g-1; pg>=1 && pg>=g-8; pg--)
                        if(h[i]==pat(pg, off+i)){ stale++; break; }
                }
            }
        }
        total_bad += bad; total_stale += stale;
        printf("gen %2d va=%p%s bad=%ld stale=%ld %s\n", g, (void*)d,
               prev_gen>0 ? " (REUSED)" : "", bad, stale,
               bad ? "MISMATCH" : "ok");
        if(bad) printf("   first bad idx %lu: got 0x%08x want 0x%08x (gen-1 pat 0x%08x)\n",
                       firstbad, firstgot, pat(g,firstbad), pat(g-1,firstbad));

        if(nhist<4096){ va_hist[nhist]=(void*)d; gen_hist[nhist]=g; nhist++; }
        CK(hipFree(d));
    }
    printf("RESULT: va_reuses=%ld total_bad=%ld total_stale=%ld -> %s\n",
           va_reuses, total_bad, total_stale,
           total_bad ? (total_stale ? "STALE-TRANSLATION MISMATCHES" : "MISMATCHES (not prior-gen pattern)")
                     : "ALL CORRECT");
    free(h);
    return total_bad ? 3 : 0;
}
