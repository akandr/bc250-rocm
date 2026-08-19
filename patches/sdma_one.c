// Single-size SDMA copy loop, for the one-byte intervention.
//
// The existing probe walks a fixed list of sizes and takes a repetition count,
// not a size, which an earlier attempt at this experiment got wrong: it passed
// 16384 and 16385 as repetition counts and compared two identical 16 KiB runs.
// This one takes the size in bytes and loops, so the queue stays busy for as
// long as the descriptor is being sampled. That matters: a single 16 KiB copy
// finishes in microseconds, so sampling a completed copy shows an idle queue
// and proves nothing about whether descriptors track live work.
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    size_t bytes = (argc > 1) ? strtoul(argv[1], NULL, 10) : 16384;
    int secs = (argc > 2) ? atoi(argv[2]) : 20;
    printf("size=%zu bytes, looping for %d s\n", bytes, secs); fflush(stdout);

    void *dev = NULL; void *host = malloc(bytes);
    if (hipMalloc(&dev, bytes) != hipSuccess) { printf("hipMalloc failed\n"); return 1; }
    memset(host, 0xA5, bytes);

    struct timespec t0, now; clock_gettime(CLOCK_MONOTONIC, &t0);
    unsigned long n = 0;
    for (;;) {
        hipError_t e = hipMemcpy(dev, host, bytes, hipMemcpyHostToDevice);
        if (e != hipSuccess) { printf("copy %lu failed: %s\n", n, hipGetErrorString(e)); return 2; }
        n++;
        if ((n & 0x3F) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            if (now.tv_sec - t0.tv_sec >= secs) break;
            printf("copies=%lu\n", n); fflush(stdout);
        }
    }
    printf("COMPLETED copies=%lu\n", n); fflush(stdout);
    return 0;
}
