// half_builtins_abi.c - GPU-free check of the half-precision helpers in ROCm's compiler-rt builtins.
//
// Converts half 1.0 to float and float 1.0 to half through out-of-line calls. Correct helpers print
// 1 and 0x3c00 on every line; the integer-register variant shipped in Fedora 43's
// rocm-clang-runtime-devel 19-14.rocm6.4.2 converts whatever is in %edi instead
// (logs/fp16-root-cause-2026-09-15/). The load_edi calls were meant to vary that value, but the
// optimiser drops the unused argument, so on the board every line shows the same wrong result.
//
// Build on the board (no -mf16c, so the conversions become calls to the builtins):
//   /usr/lib64/rocm/llvm/bin/clang -O2 --rtlib=compiler-rt half_builtins_abi.c -o half_builtins_abi
#include <stdio.h>

__attribute__((noinline)) void load_edi(int v) { (void)v; }
__attribute__((noinline)) float half_to_float(_Float16 h) { return (float)h; }
__attribute__((noinline)) _Float16 float_to_half(float f) { return (_Float16)f; }

int main(void) {
  const int junk[] = {0, 0x3c00, 0x7c00, 0x1234};
  int bad = 0;
  for (int i = 0; i < 4; i++) {
    load_edi(junk[i]);
    float f = half_to_float((_Float16)1.0f);
    load_edi(junk[i]);
    _Float16 h = float_to_half(1.0f);
    unsigned bits = 0;
    __builtin_memcpy(&bits, &h, 2);
    printf("edi=0x%04x: (float)half(1.0) = %g, (half)1.0f bits = 0x%04x\n", junk[i], f, bits);
    bad += (f != 1.0f) + (bits != 0x3c00);
  }
  printf("RESULT %s\n", bad ? "BROKEN" : "OK");
  return bad != 0;
}
