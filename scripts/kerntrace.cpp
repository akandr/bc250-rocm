// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// kerntrace: LD_PRELOAD roctracer tool. Records every GPU kernel dispatch (HIP_OPS activity) and at exit
// prints, per kernel name, the count, the summed device time and the share, plus the sum over all kernels
// and the wall time between the first dispatch and the last completion. Build:
//   hipcc -O2 -fPIC -shared kerntrace.cpp -o libkerntrace.so -lroctracer64 -ldl
// Use:   LD_PRELOAD=./libkerntrace.so llama-bench ...      (KERNTRACE_OUT=file for the report)
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_ext.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

namespace {
struct Acc { uint64_t n = 0, ns = 0; };
std::map<std::string, Acc> g_acc;
std::mutex g_mu;
std::map<std::pair<uint32_t,uint32_t>, uint64_t> g_seen;   // (domain, op) -> records, for debugging
uint64_t g_first = ~0ull, g_last = 0, g_total = 0, g_count = 0;
std::vector<std::pair<uint64_t,uint64_t>> g_spans;   // (begin, end) of every dispatch, for the gap histogram

void buffer_cb(const char * begin, const char * end, void *) {
    std::lock_guard<std::mutex> lk(g_mu);
    const roctracer_record_t * r = (const roctracer_record_t *) begin;
    const roctracer_record_t * e = (const roctracer_record_t *) end;
    while (r < e) {
        g_seen[{r->domain, (uint32_t) r->op}]++;
        // only dispatch records carry a kernel name; copies and barriers hold a byte count in the same union
        if (r->domain == ACTIVITY_DOMAIN_HIP_OPS && r->op == HIP_OP_ID_DISPATCH && r->kernel_name) {
            const uint64_t d = r->end_ns - r->begin_ns;
            Acc & a = g_acc[r->kernel_name]; a.n++; a.ns += d;
            g_total += d; g_count++;
            g_first = std::min(g_first, (uint64_t) r->begin_ns); g_last = std::max(g_last, (uint64_t) r->end_ns);
            g_spans.emplace_back(r->begin_ns, r->end_ns);
        }
        roctracer_next_record(r, &r);
    }
}

std::string shorten(const std::string & s) {   // strip the template-argument noise from a mangled-demangled name
    // KERNTRACE_FULL=1 keeps the template arguments and widens the field, which is the only way to tell
    // one instantiation of a heavily templated kernel from another in the per-kernel table.
    static const bool full = getenv("KERNTRACE_FULL") != nullptr;
    if (full) {
        return s.size() > 200 ? s.substr(0, 200) : s;
    }
    std::string o; int depth = 0;
    for (char c : s) { if (c == '<') depth++; else if (c == '>') depth--; else if (depth == 0) o += c; }
    if (o.size() > 90) o = o.substr(0, 90);
    return o;
}

bool g_enabled = false;
void ensure_enabled() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_enabled) return;
    g_enabled = true;
    roctracer_properties_t props{}; props.buffer_size = 1 << 24; props.buffer_callback_fun = buffer_cb;
    roctracer_open_pool(&props);
    roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);
    roctracer_start();
}

struct Tool {
    Tool() {}
    ~Tool() {
        if (!g_enabled) { fprintf(stderr, "kerntrace: no HIP call was intercepted\n"); return; }
        roctracer_stop();
        roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);
        roctracer_flush_activity();
        const char * out = getenv("KERNTRACE_OUT");
        FILE * f = out ? fopen(out, "w") : stderr;
        std::vector<std::pair<std::string, Acc>> v(g_acc.begin(), g_acc.end());
        std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second.ns > b.second.ns; });
        fprintf(f, "kerntrace: %llu kernel dispatches, %.3f ms of kernel time, %.3f ms first dispatch to last completion\n",
                (unsigned long long) g_count, g_total / 1e6, (g_last - g_first) / 1e6);
        for (auto & [k, n] : g_seen) fprintf(f, "  records domain %u op %u: %llu\n", k.first, k.second, (unsigned long long) n);
        // gaps between consecutive dispatches (by begin time): idle time the GPU spends waiting for the next kernel
        std::sort(g_spans.begin(), g_spans.end());
        const double edges[] = {1e3, 2e3, 5e3, 20e3, 100e3, 1e12}; uint64_t cnt[6] = {0}, sum[6] = {0}; uint64_t last_end = 0, overlap = 0;
        for (auto & [b, e] : g_spans) {
            if (last_end) { if (b > last_end) { uint64_t g = b - last_end; int i = 0; while (g > edges[i]) i++; cnt[i]++; sum[i] += g; } else overlap++; }
            last_end = std::max(last_end, e);
        }
        fprintf(f, "  gaps between consecutive kernels: <1us %llu (%.1f ms), 1-2us %llu (%.1f ms), 2-5us %llu (%.1f ms), 5-20us %llu (%.1f ms), 20-100us %llu (%.1f ms), >100us %llu (%.1f ms); overlapping starts %llu\n",
                (unsigned long long) cnt[0], sum[0]/1e6, (unsigned long long) cnt[1], sum[1]/1e6, (unsigned long long) cnt[2], sum[2]/1e6,
                (unsigned long long) cnt[3], sum[3]/1e6, (unsigned long long) cnt[4], sum[4]/1e6, (unsigned long long) cnt[5], sum[5]/1e6, (unsigned long long) overlap);
        fprintf(f, "%10s %12s %7s %10s  %s\n", "count", "total ms", "share", "avg us", "kernel");
        for (auto & [name, a] : v) {
            fprintf(f, "%10llu %12.3f %6.1f%% %10.2f  %s\n", (unsigned long long) a.n, a.ns / 1e6, 100.0 * a.ns / g_total,
                    a.ns / 1e3 / a.n, shorten(name).c_str());
        }
        if (out) fclose(f);
    }
} g_tool;
}

// The first HIP calls ggml-cuda makes; enabling from inside them means HIP is loaded and initialising.
extern "C" hipError_t hipGetDeviceCount(int * count) {
    static auto real = (hipError_t (*)(int *)) dlsym(RTLD_NEXT, "hipGetDeviceCount");
    ensure_enabled();
    return real(count);
}
extern "C" hipError_t hipSetDevice(int device) {
    static auto real = (hipError_t (*)(int)) dlsym(RTLD_NEXT, "hipSetDevice");
    ensure_enabled();
    return real(device);
}
extern "C" hipError_t hipMalloc(void ** ptr, size_t size) {
    static auto real = (hipError_t (*)(void **, size_t)) dlsym(RTLD_NEXT, "hipMalloc");
    ensure_enabled();
    return real(ptr, size);
}
