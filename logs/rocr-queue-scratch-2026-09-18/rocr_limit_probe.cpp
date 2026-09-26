// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// rocr_limit_probe: fill device-visible system memory in 512 MiB steps until hipMalloc fails, then create
// a stream. Against the ROCm 7.1.1 runtime the stream creation, whose queue scratch cannot be allocated
// past the KFD system-memory limit, segfaults at address 0x20 inside libhsa-runtime64; with rocm-systems
// PR #2850 ported it returns an error instead. usage: rocr_limit_probe [step MiB]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    size_t step = (argc > 1 ? atoi(argv[1]) : 512) * (size_t)1 << 20;
    std::vector<void*> blocks; size_t total = 0; hipError_t e;
    while (true) {
        void* p = nullptr; e = hipMalloc(&p, step);
        if (e != hipSuccess) { printf("hipMalloc failed after %zu MiB: %s\n", total >> 20, hipGetErrorString(e)); break; }
        blocks.push_back(p); total += step;
        if (blocks.size() % 4 == 0) printf("  allocated %zu MiB\n", total >> 20);
    }
    printf("creating a stream with %zu MiB held...\n", total >> 20);
    hipStream_t s; e = hipStreamCreate(&s);
    printf("hipStreamCreate: %s\n", e == hipSuccess ? "ok" : hipGetErrorString(e));
    if (e == hipSuccess) hipStreamDestroy(s);
    // release a block and try again: the stream should now come up
    if (!blocks.empty()) { hipFree(blocks.back()); blocks.pop_back(); }
    e = hipStreamCreate(&s);
    printf("after freeing one block, hipStreamCreate: %s\n", e == hipSuccess ? "ok" : hipGetErrorString(e));
    for (void* p : blocks) hipFree(p);
    return 0;
}
