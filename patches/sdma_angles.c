// sdma_angles.c - narrow down which SDMA path fails on the BC-250.
//
// Established already: with HSA_ENABLE_SDMA=1, a pageable host-to-device
// hipMemcpy of 16384 bytes completes and 16385 never returns, and no SDMA trap
// interrupt arrives for either. This asks which property of that operation
// actually matters, since "size" alone does not explain why the boundary sits
// on a power of two.
//
// Each case runs in its own watchdogged child so one hang does not end the run.
//
// Build:
//   /usr/lib64/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1013 -O2 \
//       sdma_angles.c -o sdma_angles
//   HSA_ENABLE_SDMA=1 ./sdma_angles
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

#define SMALL 16384
#define BIG   16385
#define WATCHDOG_S 20

static void *dA, *dB;
static void *hPageable, *hPinned;

static int run_case(const char *name, int (*fn)(void))
{
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        // child: the whole HIP context is re-created here
        alarm(WATCHDOG_S);
        int rc = fn();
        _exit(rc == 0 ? 0 : 2);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    const char *verdict;
    if (WIFEXITED(st) && WEXITSTATUS(st) == 0)      verdict = "completes";
    else if (WIFSIGNALED(st) && WTERMSIG(st) == SIGALRM) verdict = "HANGS";
    else                                            verdict = "error";
    printf("  %-46s %s\n", name, verdict);
    fflush(stdout);
    return 0;
}

static int setup(void)
{
    if (hipMalloc(&dA, 1 << 20) || hipMalloc(&dB, 1 << 20)) return 1;
    hPageable = malloc(1 << 20);
    if (!hPageable) return 1;
    memset(hPageable, 1, 1 << 20);
    if (hipHostMalloc(&hPinned, 1 << 20, 0)) return 1;
    memset(hPinned, 1, 1 << 20);
    return 0;
}

// The baseline, to confirm the threshold in this harness
static int c_pageable_small(void) { if (setup()) return 1; return hipMemcpy(dA, hPageable, SMALL, hipMemcpyHostToDevice) ? 1 : hipDeviceSynchronize(); }
static int c_pageable_big(void)   { if (setup()) return 1; return hipMemcpy(dA, hPageable, BIG,   hipMemcpyHostToDevice) ? 1 : hipDeviceSynchronize(); }

// Does pinned host memory change it? Pinned needs no bounce buffer, so if the
// failure is the staging handoff this should behave differently.
static int c_pinned_big(void)     { if (setup()) return 1; return hipMemcpy(dA, hPinned, BIG, hipMemcpyHostToDevice) ? 1 : hipDeviceSynchronize(); }
static int c_pinned_1m(void)      { if (setup()) return 1; return hipMemcpy(dA, hPinned, 1 << 20, hipMemcpyHostToDevice) ? 1 : hipDeviceSynchronize(); }

// Device to device never crosses the host at all
static int c_d2d_big(void)        { if (setup()) return 1; return hipMemcpy(dB, dA, BIG, hipMemcpyDeviceToDevice) ? 1 : hipDeviceSynchronize(); }
static int c_d2d_1m(void)         { if (setup()) return 1; return hipMemcpy(dB, dA, 1 << 20, hipMemcpyDeviceToDevice) ? 1 : hipDeviceSynchronize(); }

// Async on an explicit stream: a different submission path in the runtime
static int c_async_big(void)
{
    if (setup()) return 1;
    hipStream_t s;
    if (hipStreamCreate(&s)) return 1;
    if (hipMemcpyAsync(dA, hPageable, BIG, hipMemcpyHostToDevice, s)) return 1;
    return hipStreamSynchronize(s);
}

// hipMemset is a fill rather than a copy, and may take the same engine
static int c_memset_big(void)     { if (setup()) return 1; return hipMemset(dA, 7, BIG) ? 1 : hipDeviceSynchronize(); }
static int c_memset_1m(void)      { if (setup()) return 1; return hipMemset(dA, 7, 1 << 20) ? 1 : hipDeviceSynchronize(); }

// Device to host at the same size, to confirm direction is not the variable
static int c_d2h_big(void)        { if (setup()) return 1; return hipMemcpy(hPageable, dA, BIG, hipMemcpyDeviceToHost) ? 1 : hipDeviceSynchronize(); }

int main(void)
{
    printf("SDMA=%s\n", getenv("HSA_ENABLE_SDMA") ? getenv("HSA_ENABLE_SDMA") : "(unset)");
    printf("watchdog %d s per case, each in its own process\n\n", WATCHDOG_S);

    run_case("pageable H2D 16384 (known good)",        c_pageable_small);
    run_case("pageable H2D 16385 (known bad)",         c_pageable_big);
    run_case("pinned   H2D 16385",                     c_pinned_big);
    run_case("pinned   H2D 1 MiB",                     c_pinned_1m);
    run_case("device to device 16385",                 c_d2d_big);
    run_case("device to device 1 MiB",                 c_d2d_1m);
    run_case("async on a stream, pageable H2D 16385",  c_async_big);
    run_case("hipMemset 16385",                        c_memset_big);
    run_case("hipMemset 1 MiB",                        c_memset_1m);
    run_case("pageable D2H 16385",                     c_d2h_big);
    return 0;
}
