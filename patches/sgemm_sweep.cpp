// sgemm_sweep.cpp — minimal native-gfx1013 rocBLAS SGEMM, CPU-verified, with timing.
//
// Purpose: a tiny, dependency-light way to see what the gfx1013 compute queue does with a
// *real* library GEMM at different sizes and under sustained iteration, without llama.cpp in
// the way. Column-major, C = alpha*A*B + beta*C with alpha=1, beta=0. Two elements are checked
// against a double-precision CPU reference, and GFLOP/s is reported.
//
// Build (native gfx1013, against a locally-built rocBLAS — see scripts/build_rocblas_gfx1013.sh):
//   RB=~/rocBLAS/build/release
//   /usr/lib64/rocm/llvm/bin/clang++ -std=c++17 -x hip --offload-arch=gfx1013 \
//     -I$RB/rocblas-install/include sgemm_sweep.cpp -o sgemm_sweep \
//     -L$RB/rocblas-install/lib -lrocblas -L/usr/lib64 -lamdhip64
//
// Run (no HSA_OVERRIDE):
//   LD_LIBRARY_PATH=$RB/rocblas-install/lib:/usr/lib64 \
//   ROCBLAS_TENSILE_LIBPATH=$RB/rocblas-install/lib/rocblas/library \
//   ./sgemm_sweep <N> <iters>
//
// On this board we wrapped each invocation in `timeout 90` so a wedged queue could not hang forever.
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

int main(int argc, char** argv) {
  int N     = argc > 1 ? atoi(argv[1]) : 512;
  int iters = argc > 2 ? atoi(argv[2]) : 1;
  size_t sz = (size_t)N * N;
  std::vector<float> A(sz), B(sz), C(sz);
  for (size_t i = 0; i < sz; i++) { A[i] = (float)((i % 7) + 1); B[i] = (float)((i % 5) + 1); }

  float *dA, *dB, *dC;
  if (hipMalloc(&dA, sz*4) || hipMalloc(&dB, sz*4) || hipMalloc(&dC, sz*4)) {
    printf("N=%d hipMalloc fail\n", N); return 2;
  }
  hipMemcpy(dA, A.data(), sz*4, hipMemcpyHostToDevice);
  hipMemcpy(dB, B.data(), sz*4, hipMemcpyHostToDevice);

  rocblas_handle h; rocblas_create_handle(&h);
  float alpha = 1.0f, beta = 0.0f;
  auto t0 = std::chrono::high_resolution_clock::now();
  rocblas_status st = rocblas_status_success;
  for (int it = 0; it < iters; it++) {
    st = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
                       N, N, N, &alpha, dA, N, dB, N, &beta, dC, N);
    if (st != rocblas_status_success) break;
  }
  hipError_t he = hipDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  hipMemcpy(C.data(), dC, sz*4, hipMemcpyDeviceToHost);

  auto ref = [&](int i, int j) {
    double s = 0; for (int k = 0; k < N; k++) s += (double)A[i + (size_t)k*N] * B[k + (size_t)j*N];
    return (float)s;
  };
  float g0 = C[0], r0 = ref(0, 0);
  int i2 = N/3, j2 = N/2; float g1 = C[i2 + (size_t)j2*N], r1 = ref(i2, j2);
  bool ok = fabs(g0 - r0) < 1e-2*fabs(r0) + 1 && fabs(g1 - r1) < 1e-2*fabs(r1) + 1;
  double gflops = 2.0 * N * N * N * iters / (ms / 1e3) / 1e9;
  printf("N=%-5d iters=%-4d st=%d sync=%d %6.1fms %6.1f GFLOP/s  chk(%.0f/%.0f,%.0f/%.0f) %s\n",
         N, iters, (int)st, (int)he, ms, gflops, g0, r0, g1, r1, ok ? "CORRECT" : "WRONG");
  rocblas_destroy_handle(h); hipFree(dA); hipFree(dB); hipFree(dC);
  return ok ? 0 : 1;
}
