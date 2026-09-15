// half_abi_test.c - does clang-compiled half-precision conversion go wrong through libgcc?
//
// rocBLAS built with ROCm clang imports __extendhfsf2 / __truncsfhf2 from libgcc_s
// (GCC_12.0.0 symbol version). If clang passes the half value where libgcc's helper does not look
// for it, the helper converts whatever is left in the register it does read, and the result depends
// on unrelated earlier code. This converts a constant half 1.0 after loading different values into
// the SSE register, and prints what comes back.
//
// Result: libgcc_s converts correctly whatever is left in %xmm0, so this hypothesis is refuted. The
// broken helpers are the ones statically linked from ROCm's compiler-rt builtins archive
// (logs/fp16-root-cause-2026-09-15/).
//
// Build on the board: /usr/lib64/rocm/llvm/bin/clang -O2 half_abi_test.c -o half_abi_test
#include <stdio.h>

__attribute__((noinline)) float half_to_float(const _Float16 *p) { return (float)*p; }
__attribute__((noinline)) int half_is_nonzero(const _Float16 *p) { return *p ? 1 : 0; }
__attribute__((noinline)) double load_xmm0(double v) { return v; }

int main(void) {
  const _Float16 one = 1.0f;
  const double junk[] = {0.0, 3.5, -2.0e30, 7.0};
  for (int i = 0; i < 4; i++) {
    (void)load_xmm0(junk[i]);  // leaves junk[i] in %xmm0
    float f = half_to_float(&one);
    (void)load_xmm0(junk[i]);
    int nz = half_is_nonzero(&one);
    printf("xmm0 preloaded with %g: (float)half(1.0) = %g, half(1.0) != 0 -> %d\n", junk[i], f, nz);
  }
  return 0;
}
