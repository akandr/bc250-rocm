# What the RDNA1 macro actually buys the integer dot product, 2026-09-17

llama.cpp patch 3 ([`patches/llamacpp/0003-gfx1013-rdna1-macro.patch`](../../patches/llamacpp/0003-gfx1013-rdna1-macro.patch))
adds `__gfx1013__` to the device-side `RDNA1` macro in `ggml/src/ggml-cuda/vendors/hip.h`. It is worth
3.70x end-to-end prefill ([`logs/macro-remeasure-2026-08-18/`](../macro-remeasure-2026-08-18/)), and
[INVESTIGATION.md](../../INVESTIGATION.md) credited that to the hand-written RDNA1 integer-dot emulation
the macro enables. This checks that attribution at the instruction level, on the board, with
`hipcc` 7.1.52802 (clang 20).

## What was run

`sdot4-probe.cpp` asks for `__builtin_amdgcn_sdot4` on gfx1013. `dp4a.cpp` compiles both arms of
`ggml_cuda_dp4a`, the generic byte-wise `#else` branch gfx1013 gets without the macro and the
RDNA1 inline asm it gets with it, into one device module, with `threadIdx.x`-indexed inputs so the
compiler cannot scalarise them into SALU. `dp4a-gfx1013.s` is the resulting ISA; `log` has the
counts.

## Result

**gfx1013 has no `v_dot4_i32_i8`, so the emulation is the right path and not a workaround.**

    error: '__builtin_amdgcn_sdot4' needs target feature dot1-insts

which matches LLVM: `FeatureISAVersion10_1_3` in `llvm/lib/Target/AMDGPU/AMDGPU.td` adds only
`FeatureBVHRayTracingInsts` and `FeatureMSAALoadInsts` to the common 10.1 set, while gfx1011 and
gfx1012 add `FeatureDot1Insts` and its siblings. gfx1013 belongs on gfx1010's path.

**But the emulation is only about 20 percent cheaper per dot, not several times cheaper.**

| arm | vector ALU instructions |
|---|---|
| generic `#else` branch (gfx1013 without the macro) | 11 |
| RDNA1 inline asm (gfx1013 with it) | 9 |

One instruction in each is the shared address `v_lshlrev_b32`, so the dot itself is 10 against 8.
The fallback is not the naive four-byte-extract loop it reads as in C: the compiler contracts it
into `v_mul_i32_i24` with SDWA byte selects, landing close to what the inline asm writes by hand.

## What this means

A 3.70x end-to-end difference cannot come from a 20 percent saving in one inner-loop instruction, so
the attribution was too narrow. The macro flips three things on the current tree, not one:

| file | with the macro | without it |
|---|---|---|
| `common.cuh` `ggml_cuda_dp4a` | RDNA1 inline asm | generic byte-wise branch |
| `fattn-tile.cuh` | `ggml_cuda_fattn_tile_get_config_amd_rdna` | generic AMD table |
| `fattn-vec.cuh` | `nthreads_KQ_q = 2` | `nthreads_KQ_q = 4` |

(A fourth, `get_mmq_y_device` returning 64 instead of 128, existed when the macro was first measured
but has since been removed upstream.) Which of the three carries the prefill gain is not separated
here. The macro as a whole is worth 3.70x; the split inside it is open.

## Also checked

The macro gap is still present on llama.cpp master and, as far as GitHub search shows, has never
been reported: no issue, no pull request, and no commit in the repository contains the string
`gfx1013` on the HIP side. The host side disagrees with the device side, since
`GGML_CUDA_CC_IS_RDNA1(cc)` in `common.cuh` is the range check `cc >= 0x1010 && cc < 0x1030` and is
therefore true for gfx1013 while the device macro is a hard-coded list that omits it.
