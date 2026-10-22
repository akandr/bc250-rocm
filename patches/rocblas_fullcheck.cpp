// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// rocblas_fullcheck.cpp: every output element of rocBLAS GEMM checked, at every iteration.
//
// sgemm_iter.cpp and rocblas_bench.cpp check two elements of an output buffer that is reused across
// iterations, so after the first iteration a dropped store keeps the previous correct value and
// passes. Here the output buffer is filled with NaN (all bits set) before every call, the full
// result is copied back, and each element is compared exactly with a CPU reference. Inputs are small
// integers (-3..3 for fp32/fp64, -1..1 for fp16), so every partial and final sum is an integer the
// format represents exactly and the comparison needs no tolerance; an element the kernel never wrote
// stays NaN and is counted separately.
//
// Build (OpenMP for the CPU reference):
//   /usr/lib64/rocm/llvm/bin/clang++ -std=c++17 -O3 -march=native -fopenmp -x hip \
//     --offload-arch=gfx1013 -I/opt/bc250-rocm/include rocblas_fullcheck.cpp -o rocblas_fullcheck \
//     -L/opt/bc250-rocm/lib64 -lrocblas -L/usr/lib64 -lamdhip64
// Run:
//   LD_LIBRARY_PATH=/opt/bc250-rocm/lib64:/usr/lib64 ./rocblas_fullcheck <type> <op> <N> <iters>
//   type: s (sgemm), d (dgemm), h (hgemm), hs (gemm_ex fp16 in/out, fp32 compute)
//   op:   nn or tn (A transposed, the layout llama.cpp's ROCm backend uses)
// Exit status 0 only if every element of every iteration matched.
// Negative controls: FULLCHECK_CONTROL=skip computes only N-1 rows (the last row must be reported
// unwritten), FULLCHECK_CONTROL=wrong uses alpha=2 (every nonzero element must be reported wrong).

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_OK(x)  do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("hip error %s at %d\n", hipGetErrorString(e), __LINE__); fflush(stdout); exit(2);} } while (0)
#define RB_OK(x)   do { rocblas_status s = (x); if (s != rocblas_status_success) { \
    printf("rocblas error %d at %d\n", (int) s, __LINE__); fflush(stdout); exit(2);} } while (0)

static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static int rnd(int lo, int hi) {
    lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
    return lo + (int) ((lcg >> 33) % (uint64_t) (hi - lo + 1));
}

template <typename T> static T from_int(int v) { return (T) v; }
template <> rocblas_half from_int<rocblas_half>(int v) { __half h = __float2half((float) v); rocblas_half r; memcpy(&r, &h, 2); return r; }
template <typename T> static double to_double(T v) { return (double) v; }
template <> double to_double<rocblas_half>(rocblas_half v) { __half h; memcpy(&h, &v, 2); return (double) __half2float(h); }

template <typename T>
static int run(const std::string &type, bool tn, int N, int iters, rocblas_handle h) {
    const size_t sz = (size_t) N * N;
    const int lo = (type == "h" || type == "hs") ? -1 : -3, hi = -lo;
    std::vector<int> Ai(sz), Bi(sz);
    for (auto &v : Ai) v = rnd(lo, hi);
    for (auto &v : Bi) v = rnd(lo, hi);
    // column-major: element (r,c) of a matrix stored at r + c*N.
    // op(A)(i,k) = A[i + k*N] for nn, A[k + i*N] for tn.
    std::vector<int32_t> ref(sz, 0);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int> Aop(sz);                            // op(A) stored column-major
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++) Aop[i + (size_t) k * N] = tn ? Ai[k + (size_t) i * N] : Ai[i + (size_t) k * N];
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {
        int32_t *cj = &ref[(size_t) j * N];
        for (int k = 0; k < N; k++) {
            const int32_t b = Bi[k + (size_t) j * N];
            const int *ak = &Aop[(size_t) k * N];
            for (int i = 0; i < N; i++) cj[i] += ak[i] * b;
        }
    }
    double ref_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int32_t maxabs = 0; for (auto v : ref) maxabs = std::max(maxabs, std::abs(v));

    std::vector<T> A(sz), B(sz), C(sz);
    for (size_t i = 0; i < sz; i++) { A[i] = from_int<T>(Ai[i]); B[i] = from_int<T>(Bi[i]); }
    T *dA, *dB, *dC;
    HIP_OK(hipMalloc(&dA, sz * sizeof(T))); HIP_OK(hipMalloc(&dB, sz * sizeof(T))); HIP_OK(hipMalloc(&dC, sz * sizeof(T)));
    HIP_OK(hipMemcpy(dA, A.data(), sz * sizeof(T), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dB, B.data(), sz * sizeof(T), hipMemcpyHostToDevice));
    const rocblas_operation opA = tn ? rocblas_operation_transpose : rocblas_operation_none;
    const char *ctl = getenv("FULLCHECK_CONTROL");
    const int M = (ctl && !strcmp(ctl, "skip")) ? N - 1 : N;
    const int al = (ctl && !strcmp(ctl, "wrong")) ? 2 : 1;
    if (ctl) printf("NEGATIVE CONTROL %s: M=%d alpha=%d\n", ctl, M, al);
    printf("type=%s op=%s N=%d iters=%d ref_cpu_s=%.1f max_abs_ref=%d\n", type.c_str(), tn ? "tn" : "nn", N, iters, ref_s, maxabs);
    fflush(stdout);

    long total_bad = 0, total_unwritten = 0; int bad_iters = 0;
    for (int it = 0; it < iters; it++) {
        HIP_OK(hipMemset(dC, 0xFF, sz * sizeof(T)));     // NaN in every element
        HIP_OK(hipDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        if (type == "s") { float a = al, b = 0; RB_OK(rocblas_sgemm(h, opA, rocblas_operation_none, M, N, N, &a, (float *) dA, N, (float *) dB, N, &b, (float *) dC, N)); }
        else if (type == "d") { double a = al, b = 0; RB_OK(rocblas_dgemm(h, opA, rocblas_operation_none, M, N, N, &a, (double *) dA, N, (double *) dB, N, &b, (double *) dC, N)); }
        else if (type == "h") { rocblas_half a = from_int<rocblas_half>(al), b = from_int<rocblas_half>(0);
            RB_OK(rocblas_hgemm(h, opA, rocblas_operation_none, M, N, N, &a, (rocblas_half *) dA, N, (rocblas_half *) dB, N, &b, (rocblas_half *) dC, N)); }
        else { float a = al, b = 0;
            RB_OK(rocblas_gemm_ex(h, opA, rocblas_operation_none, M, N, N, &a, dA, rocblas_datatype_f16_r, N, dB, rocblas_datatype_f16_r, N,
                                  &b, dC, rocblas_datatype_f16_r, N, dC, rocblas_datatype_f16_r, N, rocblas_datatype_f32_r,
                                  rocblas_gemm_algo_standard, 0, 0)); }
        hipError_t he = hipDeviceSynchronize();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        HIP_OK(hipMemcpy(C.data(), dC, sz * sizeof(T), hipMemcpyDeviceToHost));
        long bad = 0, unwritten = 0, first = -1;
        for (size_t i = 0; i < sz; i++) {
            double g = to_double<T>(C[i]);
            if (std::isnan(g)) { unwritten++; if (first < 0) first = (long) i; }
            else if (g != (double) ref[i]) { bad++; if (first < 0) first = (long) i; }
        }
        total_bad += bad; total_unwritten += unwritten; if (bad || unwritten) bad_iters++;
        printf("iter %3d sync=%d %8.2fms wrong=%ld unwritten=%ld first_bad=%ld\n", it, (int) he, ms, bad, unwritten, first);
        fflush(stdout);
    }
    printf("DONE type=%s op=%s N=%d iters=%d elements_checked=%zu bad_iters=%d wrong=%ld unwritten=%ld %s\n",
           type.c_str(), tn ? "tn" : "nn", N, iters, sz * (size_t) iters, bad_iters, total_bad, total_unwritten,
           bad_iters ? "FAIL" : "ALL_ELEMENTS_CORRECT");
    fflush(stdout);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return bad_iters ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 5) { printf("usage: %s s|d|h|hs nn|tn N iters\n", argv[0]); return 2; }
    std::string type = argv[1]; bool tn = std::string(argv[2]) == "tn";
    int N = atoi(argv[3]), iters = atoi(argv[4]);
    char ver[256] = {0}; rocblas_get_version_string(ver, sizeof ver);
    hipDeviceProp_t p; HIP_OK(hipGetDeviceProperties(&p, 0));
    printf("rocblas %s, device %s (%s), %d CUs\n", ver, p.name, p.gcnArchName, p.multiProcessorCount);
    rocblas_handle h; RB_OK(rocblas_create_handle(&h));
    int rc;
    if (type == "s") rc = run<float>(type, tn, N, iters, h);
    else if (type == "d") rc = run<double>(type, tn, N, iters, h);
    else rc = run<rocblas_half>(type, tn, N, iters, h);
    rocblas_destroy_handle(h);
    return rc;
}
