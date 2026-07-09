// evtlat.c: HIP event sync latency microbenchmark for gfx1013 (BC-250)
//
// Measures how long the CPU waits for a HIP event after an async H2D copy,
// under three wait methods. Written early in the investigation to test whether
// model-load slowness was per-event sync latency. It is not: this microbenchmark
// is fast, and the real problem is the compute-queue defect (see the repository
// README). Kept as a tool and for the record.
//
// Usage: evtlat <mode> [iters] [sleep_ms] [timing]
//   mode:     sync   = hipEventSynchronize
//             query  = hipEventQuery spin loop
//             stream = hipStreamSynchronize
//   iters:    number of copy+wait iterations (default 5)
//   sleep_ms: usleep between record and wait, mimicking the model loader
//             reading the next tensor from disk (default 50)
//   timing:   "timed" = default event; anything else/absent = hipEventDisableTiming
//             (llama.cpp/ggml uses DisableTiming events)
//
// Env matrix to test: HSA_ENABLE_SDMA=0 (always, board stability)
//                     HSA_ENABLE_INTERRUPT=0 vs unset  (tested; did not change the outcome)
//
// Safety: uses only known-good paths on this board (hipHostMalloc + H2D).
// No D2H copies, no device printf, no kernels.

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("FAIL %s: %s\n", #x, hipGetErrorString(e_)); fflush(stdout); exit(1); } } while (0)

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    const char *mode = argc > 1 ? argv[1] : "sync";
    int iters      = argc > 2 ? atoi(argv[2]) : 5;
    int sleep_ms   = argc > 3 ? atoi(argv[3]) : 50;
    int timed      = argc > 4 && !strcmp(argv[4], "timed");
    size_t sz = 6u * 1024 * 1024;

    printf("=== evtlat mode=%s iters=%d sleep_ms=%d event=%s ===\n",
           mode, iters, sleep_ms, timed ? "timed" : "disable-timing");
    printf("HSA_ENABLE_INTERRUPT=%s HSA_ENABLE_SDMA=%s\n",
           getenv("HSA_ENABLE_INTERRUPT") ? getenv("HSA_ENABLE_INTERRUPT") : "(unset)",
           getenv("HSA_ENABLE_SDMA") ? getenv("HSA_ENABLE_SDMA") : "(unset)");

    double t0 = now_ms();
    void *h = NULL, *d = NULL;
    CHECK(hipHostMalloc(&h, sz, 0));
    CHECK(hipMalloc(&d, sz));
    memset(h, 0xAB, sz);
    hipStream_t s; CHECK(hipStreamCreate(&s));
    hipEvent_t ev;
    if (timed) CHECK(hipEventCreate(&ev));
    else       CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    printf("init done in %.1f ms\n", now_ms() - t0);
    fflush(stdout);

    for (int i = 0; i < iters; i++) {
        double ta = now_ms();
        CHECK(hipMemcpyAsync(d, h, sz, hipMemcpyHostToDevice, s));
        CHECK(hipEventRecord(ev, s));
        double tb = now_ms();
        if (sleep_ms > 0) usleep(sleep_ms * 1000);
        double tc = now_ms();
        long polls = 0;
        if (!strcmp(mode, "sync")) {
            CHECK(hipEventSynchronize(ev));
        } else if (!strcmp(mode, "query")) {
            hipError_t st;
            while ((st = hipEventQuery(ev)) == hipErrorNotReady) polls++;
            if (st != hipSuccess) { printf("query err: %s\n", hipGetErrorString(st)); exit(1); }
        } else if (!strcmp(mode, "stream")) {
            CHECK(hipStreamSynchronize(s));
        } else {
            printf("unknown mode %s\n", mode); exit(1);
        }
        double td = now_ms();
        printf("iter %d: submit=%.2f ms  wait=%.2f ms  polls=%ld\n",
               i, tb - ta, td - tc, polls);
    }

    printf("cleanup...\n");
    CHECK(hipEventDestroy(ev));
    CHECK(hipStreamDestroy(s));
    CHECK(hipFree(d));
    CHECK(hipHostFree(h));
    printf("DONE\n");
    return 0;
}
