// hgemm_zero_probe2.cpp - standalone reproducer attempt for fp16 hipblasGemmEx
// intermittently returning an all-zero result on gfx1013.
//
// Traced from llama.cpp: the layer-0 value projection (F16 weight, F32
// activations, ne11=512) goes through hipblasGemmEx with fp16 compute and
// comes back all zeros on roughly half the calls, while the weight and the
// converted activations are both verified intact and the handle is bound to
// the right stream. Selecting fp32 compute avoids it entirely.
//
// This mirrors what ggml does around the call: a non-default stream bound to
// the handle, a fresh device buffer per call, an f32->f16 conversion kernel on
// that same stream immediately before the GEMM, then the GEMM.
//
//   hipcc -O2 -x hip --offload-arch=gfx1013 hgemm_zero_probe2.cpp -o hgemm_zero_probe2 -lhipblas
//   (--offload-arch is required. Without it the conversion kernel is never
//    built for the device, silently does not run, and the probe reports every
//    call as a failure. Check the reported converted-input sum is non-zero.)
//   LD_LIBRARY_PATH=<native-rocblas>/lib ./hgemm_zero_probe2 [iters] [gap] [ballast_gib]
//     iters       GEMM calls to make
//     gap         unrelated kernel launches between GEMMs (default 0)
//     ballast_gib device memory held resident (default 0)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)
#define BLAS_CHECK(x) do { hipblasStatus_t s = (x); if (s != HIPBLAS_STATUS_SUCCESS) { \
    printf("hipblas error %d at line %d\n", (int) s, __LINE__); exit(1); } } while (0)

static const int M = 1024;   // output width  (V projection)
static const int K = 4096;   // reduction     (n_embd)
static const int N = 512;    // tokens        (micro-batch)

__global__ void f32_to_f16(const float * __restrict__ src, _Float16 * __restrict__ dst, size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (_Float16) src[i];
}

// Busywork to imitate what a model does between two GEMMs of the same shape:
// roughly a thousand unrelated kernel launches on the same stream.
__global__ void filler(float * __restrict__ p, size_t n, int seed) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p[i] * 1.000001f + (float) (seed & 7) * 1e-6f;
}

int main(int argc, char ** argv) {
    const int iters = (argc > 1) ? atoi(argv[1]) : 100;

    std::vector<_Float16> hA((size_t) M * K);
    std::vector<float>    hB32((size_t) K * N);
    for (size_t i = 0; i < hA.size();  i++) hA[i]  = (_Float16) (0.002f * (float) ((int) (i % 97) - 48));
    for (size_t i = 0; i < hB32.size(); i++) hB32[i] = 0.002f * (float) ((int) (i % 89) - 44);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    hipblasHandle_t h;
    BLAS_CHECK(hipblasCreate(&h));
    BLAS_CHECK(hipblasSetStream(h, stream));

    void * dA; void * dB32; void * dC;
    HIP_CHECK(hipMalloc(&dA, hA.size() * 2));
    HIP_CHECK(hipMalloc(&dB32, hB32.size() * 4));
    HIP_CHECK(hipMalloc(&dC, (size_t) M * N * 2));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB32, hB32.data(), hB32.size() * 4, hipMemcpyHostToDevice));

    const _Float16 alpha_h = (_Float16) 1.0f, beta_h = (_Float16) 0.0f;
    std::vector<_Float16> out((size_t) M * N);
    std::vector<_Float16> ones((size_t) M * N, (_Float16) 1.0f);

    const int gap = (argc > 2) ? atoi(argv[2]) : 0;   // unrelated launches between GEMMs
    void * dfill = nullptr;
    HIP_CHECK(hipMalloc(&dfill, 1 << 20));
    HIP_CHECK(hipMemset(dfill, 0, 1 << 20));

    // Optional third argument: gibibytes of device memory to hold resident,
    // imitating a loaded model. The failure has only ever been seen in a
    // process with about 9 GiB in use.
    const int ballast_gib = (argc > 3) ? atoi(argv[3]) : 0;
    for (int b = 0; b < ballast_gib; b++) {
        void * p = nullptr;
        if (hipMalloc(&p, 1ull << 30) != hipSuccess) { printf("  ballast stopped at %d GiB\n", b); break; }
        HIP_CHECK(hipMemset(p, 0x11, 1ull << 30));
    }
    if (ballast_gib) printf("holding %d GiB resident\n", ballast_gib);

    int zero = 0, ok = 0;
    printf("M=%d K=%d N=%d, fp16 compute, own stream, %d iters, %d filler launches between GEMMs\n",
           M, K, N, iters, gap);
    for (int it = 0; it < iters; it++) {
        for (int g = 0; g < gap; g++) {
            hipLaunchKernelGGL(filler, dim3(16), dim3(256), 0, stream,
                               (float *) dfill, (size_t) (1 << 18), g);
        }

        // fresh buffer each call, like ggml taking one from its pool
        void * dB16 = nullptr;
        HIP_CHECK(hipMallocAsync(&dB16, (size_t) K * N * 2, stream));

        const size_t n = (size_t) K * N;
        hipLaunchKernelGGL(f32_to_f16, dim3((n + 255) / 256), dim3(256), 0, stream,
                           (const float *) dB32, (_Float16 *) dB16, n);
        HIP_CHECK(hipGetLastError());   // a silently failing launch would look like a zero GEMM

        HIP_CHECK(hipMemcpyAsync(dC, ones.data(), ones.size() * 2, hipMemcpyHostToDevice, stream));

        // verify the conversion actually landed, so a zero result cannot be
        // blamed on the GEMM when the input was empty
        if (it < 3) {
            std::vector<_Float16> chk(n);
            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipMemcpy(chk.data(), dB16, n * 2, hipMemcpyDeviceToHost));
            double cs = 0.0; for (auto v : chk) cs += fabs((float) v);
            printf("  iter %3d: converted-input abs-sum = %.6g\n", it, cs);
        }

        BLAS_CHECK(hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K,
                                 &alpha_h, dA, HIPBLAS_R_16F, K,
                                 dB16, HIPBLAS_R_16F, K, &beta_h,
                                 dC, HIPBLAS_R_16F, M,
                                 HIPBLAS_R_16F, HIPBLAS_GEMM_DEFAULT));

        HIP_CHECK(hipMemcpyAsync(out.data(), dC, out.size() * 2, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipFreeAsync(dB16, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        double s = 0.0;
        for (auto v : out) s += fabs((float) v);
        if (s == 0.0) { zero++; printf("  iter %3d: ALL ZERO\n", it); fflush(stdout); }
        else ok++;
    }

    printf("RESULT: all-zero results in %d of %d calls (%d ok)\n", zero, iters, ok);
    hipblasDestroy(h);
    hipStreamDestroy(stream);
    return zero > 0 ? 3 : 0;
}
