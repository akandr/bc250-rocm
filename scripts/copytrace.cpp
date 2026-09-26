// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// copytrace: LD_PRELOAD roctracer tool that records every GPU memory copy, not every kernel,
// and at exit prints a histogram of transfer sizes with counts and summed device time. Written to
// answer which transfer sizes a decode actually issues, since the SDMA engine on this board is
// faster than the blit path just above 16 KiB and about four times slower at 16 MiB, so the cost of
// leaving SDMA enabled depends entirely on where a workload's copies land. Build:
//   hipcc -O2 -fPIC -shared copytrace.cpp -o libcopytrace.so -lroctracer64 -ldl
// Use:   LD_PRELOAD=./libcopytrace.so llama-bench ...      (COPYTRACE_OUT=file for the report)
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_ext.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>
#include <hip/hip_runtime_api.h>

namespace {
struct Acc { uint64_t n = 0, ns = 0, bytes = 0; };
std::map<int, Acc> g_bucket;          // log2(size) -> accumulator
std::map<uint64_t, Acc> g_exact;      // exact size -> accumulator, for the common sizes
std::mutex g_mu;
uint64_t g_copies = 0, g_copy_ns = 0, g_dispatches = 0;

int log2_bucket(uint64_t n) { int b = 0; while (n >>= 1) b++; return b; }

void buffer_cb(const char * begin, const char * end, void *) {
    std::lock_guard<std::mutex> lk(g_mu);
    const roctracer_record_t * r = (const roctracer_record_t *) begin;
    const roctracer_record_t * e = (const roctracer_record_t *) end;
    while (r < e) {
        if (r->domain == ACTIVITY_DOMAIN_HIP_OPS) {
            if (r->op == HIP_OP_ID_COPY) {
                const uint64_t d  = r->end_ns - r->begin_ns;
                const uint64_t sz = r->bytes;
                Acc & b = g_bucket[log2_bucket(sz ? sz : 1)]; b.n++; b.ns += d; b.bytes += sz;
                Acc & x = g_exact[sz]; x.n++; x.ns += d; x.bytes += sz;
                g_copies++; g_copy_ns += d;
            } else if (r->op == HIP_OP_ID_DISPATCH) {
                g_dispatches++;
            }
        }
        roctracer_next_record(r, &r);
    }
}

void report() {
    const char * path = getenv("COPYTRACE_OUT");
    FILE * f = path ? fopen(path, "w") : stderr;
    if (!f) f = stderr;
    fprintf(f, "copies %llu, dispatches %llu, summed copy time %.3f ms\n",
            (unsigned long long) g_copies, (unsigned long long) g_dispatches, g_copy_ns / 1e6);
    fprintf(f, "\n%-18s %10s %12s %14s %12s\n", "size range", "copies", "total MiB", "device time ms", "MB/s");
    for (auto & [b, a] : g_bucket) {
        const double ms = a.ns / 1e6;
        fprintf(f, "2^%-2d to 2^%-2d %10llu %12.2f %14.3f %12.1f\n", b, b + 1,
                (unsigned long long) a.n, a.bytes / 1048576.0, ms,
                ms > 0 ? a.bytes / 1e6 / (ms / 1e3) : 0.0);
    }
    fprintf(f, "\nthe ten most frequent exact sizes\n%14s %10s %14s\n", "bytes", "copies", "device time ms");
    std::vector<std::pair<uint64_t, Acc>> v(g_exact.begin(), g_exact.end());
    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second.n > b.second.n; });
    for (size_t i = 0; i < v.size() && i < 10; ++i)
        fprintf(f, "%14llu %10llu %14.3f\n", (unsigned long long) v[i].first,
                (unsigned long long) v[i].second.n, v[i].second.ns / 1e6);
    if (path) fclose(f);
}

struct Init {
    Init() {
        roctracer_properties_t p{};
        p.buffer_size = 0x400000;
        p.buffer_callback_fun = buffer_cb;
        roctracer_open_pool(&p);
        roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);
    }
    ~Init() {
        roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);
        roctracer_flush_activity();
        report();
    }
} g_init;
}  // namespace
