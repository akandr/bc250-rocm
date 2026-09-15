// gemm_fp16_min.cpp - smallest reproducer for the zeroed fp16 GEMM on gfx1013.
//
// One rocblas_gemm_ex in fp16 with the shape llama.cpp issues (transA=T transB=N, M=1024 N=512
// K=4096, alpha 1, beta 0), repeated REPS times, counting output elements that are exactly zero.
// Correct runs report 0; the defect reports all 524288.
//
// The defect depends on glibc's per-thread allocation cache: GLIBC_TUNABLES=glibc.malloc.tcache_count=1
// makes it happen on every call against the native gfx1013 rocBLAS, and tcache_count=0 removes it
// (logs/fp16-scalar-2026-09-15/). Cause: the half-conversion helpers linked into that rocBLAS read
// the wrong register (logs/fp16-root-cause-2026-09-15/); scripts/fix_half_helpers.py repairs a copy.
//
// Build on the board:
//   clang++ -O2 -D__HIP_PLATFORM_AMD__ gemm_fp16_min.cpp -o gemm_fp16_min \
//     -I<rocblas>/include -L<rocblas>/lib -lrocblas -lamdhip64
// Run:
//   GLIBC_TUNABLES=glibc.malloc.tcache_count=1 ./gemm_fp16_min [reps]
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
  const int M = 1024, N = 512, K = 4096;
  const int reps = argc > 1 ? atoi(argv[1]) : 3;
  rocblas_handle h;
  if (rocblas_create_handle(&h) != rocblas_status_success) { printf("handle failed\n"); return 2; }

  std::vector<_Float16> A((size_t)K * M), B((size_t)K * N), C((size_t)M * N);
  srand(1234);
  for (auto &v : A) v = (_Float16)((rand() % 200 - 100) / 400.0f);
  for (auto &v : B) v = (_Float16)((rand() % 200 - 100) / 400.0f);

  _Float16 *dA, *dB, *dC;
  if (hipMalloc(&dA, A.size() * 2) || hipMalloc(&dB, B.size() * 2) || hipMalloc(&dC, C.size() * 2)) {
    printf("hipMalloc failed\n"); return 2;
  }
  hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(dB, B.data(), B.size() * 2, hipMemcpyHostToDevice);

  const _Float16 alpha = (_Float16)1.0f, beta = (_Float16)0.0f;
  int bad = 0;
  for (int r = 0; r < reps; r++) {
    for (auto &v : C) v = (_Float16)0.5f;  // stamp, so "not written" and "wrote zeros" differ
    hipMemcpy(dC, C.data(), C.size() * 2, hipMemcpyHostToDevice);
    rocblas_status st = rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, M, N, K,
                                        &alpha, dA, rocblas_datatype_f16_r, K, dB, rocblas_datatype_f16_r, K,
                                        &beta, dC, rocblas_datatype_f16_r, M, dC, rocblas_datatype_f16_r, M,
                                        rocblas_datatype_f16_r, rocblas_gemm_algo_standard, 0,
                                        rocblas_gemm_flags_none);
    hipDeviceSynchronize();
    hipMemcpy(C.data(), dC, C.size() * 2, hipMemcpyDeviceToHost);
    size_t zeros = 0, stamp = 0;
    for (auto v : C) { if (v == (_Float16)0.0f) zeros++; else if (v == (_Float16)0.5f) stamp++; }
    printf("rep %d status %d zeros %zu stamp %zu of %zu\n", r, (int)st, zeros, stamp, C.size());
    if (zeros == C.size()) bad++;
  }
  printf("RESULT bad_reps=%d/%d\n", bad, reps);
  return bad ? 1 : 0;
}
