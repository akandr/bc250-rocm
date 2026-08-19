// sdma_probe.c - characterize WHERE the SDMA path fails on the BC-250.
// #29 established that HSA_ENABLE_SDMA=1 hangs every bulk model load, but not
// which transfer size or direction hangs, because llama.cpp only reports the
// end result. This walks sizes with a per-copy watchdog so the failing point
// is visible.
//   hipcc -O2 sdma_probe.c -o sdma_probe
//   HSA_ENABLE_SDMA=1 ./sdma_probe          (compare with =0)
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("    ERROR %s at line %d\n", hipGetErrorString(e), __LINE__); \
    fflush(stdout); return 1; } } while (0)

static volatile int in_flight = 0;
static volatile size_t in_flight_bytes = 0;
static volatile const char *in_flight_dir = "";

// watchdog: if a single copy takes more than 30 s, say so and abort loudly,
// because the failure mode being characterized is a hang, not an error code
static void *watchdog(void *arg) {
    (void)arg;
    for (;;) {
        int seen = in_flight;
        sleep(30);
        if (seen && in_flight == seen) {
            printf("\n    HUNG: %s of %zu bytes did not complete in 30 s\n",
                   in_flight_dir, in_flight_bytes);
            fflush(stdout);
            _exit(2);
        }
    }
    return NULL;
}

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 3;
    const char *sdma = getenv("HSA_ENABLE_SDMA");
    printf("HSA_ENABLE_SDMA=%s, %d reps per size\n", sdma ? sdma : "(unset)", reps);

    pthread_t wd; pthread_create(&wd, NULL, watchdog, NULL);

    // Fine steps through the 4 KiB to 64 KiB region, where the SDMA path was
    // found to stop working, then the coarse sweep.
    size_t sizes[] = {
        // The threshold is exactly 16 KiB: 16384 completes, 16385 hangs. The
        // byte-level steps are kept so the boundary can be re-confirmed rather
        // than taken on trust.
        4ul<<10, 8ul<<10, 12ul<<10, 16384ul, 16385ul, 16512ul, 20480ul,
        24ul<<10, 32ul<<10, 48ul<<10, 64ul<<10,
        1ul<<20, 16ul<<20, 64ul<<20,
        256ul<<20, 512ul<<20, 1024ul<<20, 2048ul<<20,
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < nsizes; i++) {
        size_t n = sizes[i];
        void *h_pageable = malloc(n);
        if (!h_pageable) { printf("%8zu MiB: host alloc failed, stopping\n", n>>20); break; }
        memset(h_pageable, 0x5a, n);
        void *h_pinned = NULL;
        int have_pinned = (hipHostMalloc(&h_pinned, n, 0) == hipSuccess);
        void *d = NULL;
        if (hipMalloc(&d, n) != hipSuccess) {
            printf("%8zu MiB: device alloc failed, stopping\n", n>>20);
            free(h_pageable); if (have_pinned) hipHostFree(h_pinned); break;
        }

        double best_h2d = 1e30, best_d2h = 1e30, best_h2d_pin = 1e30;
        // SDMA_SKIP_H2D=1 leaves the device-to-host direction testable at sizes
        // where host-to-device hangs, which is otherwise unreachable.
        const int skip_h2d = getenv("SDMA_SKIP_H2D") != NULL;
        for (int r = 0; r < reps; r++) {
            double t0, dt;
            if (!skip_h2d) {
                in_flight_bytes = n; in_flight_dir = "H2D pageable"; in_flight++;
                t0 = now_ms(); CHECK(hipMemcpy(d, h_pageable, n, hipMemcpyHostToDevice));
                CHECK(hipDeviceSynchronize());
                dt = now_ms() - t0; if (dt < best_h2d) best_h2d = dt;
            } else { best_h2d = 0.0; }

            in_flight_bytes = n; in_flight_dir = "D2H pageable"; in_flight++;
            t0 = now_ms(); CHECK(hipMemcpy(h_pageable, d, n, hipMemcpyDeviceToHost));
            CHECK(hipDeviceSynchronize());
            dt = now_ms() - t0; if (dt < best_d2h) best_d2h = dt;

            if (have_pinned) {
                in_flight_dir = "H2D pinned"; in_flight++;
                t0 = now_ms(); CHECK(hipMemcpy(d, h_pinned, n, hipMemcpyHostToDevice));
                CHECK(hipDeviceSynchronize());
                dt = now_ms() - t0; if (dt < best_h2d_pin) best_h2d_pin = dt;
            }
        }
        printf("%8zu %s: H2D %8.2f ms (%6.2f GB/s) | D2H %8.2f ms (%6.2f GB/s) | H2D pinned %8.2f ms\n",
               n >= (1ul<<20) ? n>>20 : n>>10, n >= (1ul<<20) ? "MiB" : "KiB",
               best_h2d, n / (best_h2d * 1e6), best_d2h, n / (best_d2h * 1e6),
               have_pinned ? best_h2d_pin : -1.0);
        fflush(stdout);

        hipFree(d); free(h_pageable); if (have_pinned) hipHostFree(h_pinned);
    }
    in_flight = 0;
    printf("ALL SIZES COMPLETED\n");
    return 0;
}
