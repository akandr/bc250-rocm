# The zeroed fp16 GEMM: root cause and fix, 2026-09-15

**The defect is a toolchain ABI mismatch, not a gfx1013 kernel, runtime or silicon problem.** The
native rocBLAS built on this board statically contains `__extendhfsf2` and `__truncsfhf2` from
Fedora 43's ROCm compiler-rt, compiled for the pre-`_Float16` convention that passes a half value
in an integer register. The ROCm clang 19 code calling them passes and expects the value in `%xmm0`,
as the x86-64 psABI specifies for `_Float16`. So every half-to-float conversion in the library
converts whatever happens to be in `%edi`. When that garbage converts to zero, rocBLAS decides alpha
is zero, rewrites the GEMM as a K=0 problem, and returns exactly zero.

Replacing the two helpers with the F16C instruction for the same conversion
([`scripts/fix_half_helpers.py`](../../scripts/fix_half_helpers.py)) removed the defect in every run tested, below.

## The chain, each link measured

Minimal reproducer [`patches/gemm_fp16_min.cpp`](../../patches/gemm_fp16_min.cpp): one rocBLAS fp16
`gemm_ex` with the shape llama.cpp issues, no llama.cpp involved.

1. **A trigger that makes it deterministic.** glibc's per-thread allocation cache changes what earlier
   code leaves in registers. With `GLIBC_TUNABLES=glibc.malloc.tcache_count=1` every fp16 GEMM returns
   all 524288 elements zero; with `tcache_count=0` none do (`minimal_repro_matrix.txt`, rows "orig").
   This is what made the chain below observable at all; found through the investigation in
   [`../fp16-scalar-2026-09-15/`](../fp16-scalar-2026-09-15/).
2. **Tensile receives a K=0 problem.** With `TENSILE_DB=0x42` Tensile prints the problem key and the
   packed kernel arguments: `Object key: 1024, 512, 0` and `size_3: 0` in the bad state against
   `1024, 512, 4096` and `size_3: 4096` in the good one, and it selects a different kernel for the
   K=0 problem (`tensile_db_tcache*.txt`). The same difference seen from the runtime side, the
   captured launch arguments (recorded inside the rebuilt HIP runtime by
   [`scripts/apply_hip_argdump.py`](../../scripts/apply_hip_argdump.py)), is in `rocclr_argdump_tcache*.txt`: argument 17 is `0x1000` (4096) in the
   good state and 0 in the bad one, while alpha (argument 4, `0x3c00`) and beta (argument 5) are the
   same in both.
3. **rocBLAS chooses K=0 because it reads alpha as zero.** `ConstructTensileProblem` in
   `tensile_host.cpp` has `auto k = prob.k && *prob.alpha ? prob.k : 0;` ("We set K=0 when
   alpha==0"). Stopped just after the half-to-float call that evaluates `*prob.alpha`
   (`gdb_alpha_branch_tcache*.txt`): `prob.k` is 4096 in both states; the conversion of alpha, whose
   bits are `0x3c00` (1.0, as the launch arguments in `rocclr_argdump_tcache*.txt` show), returns `0x00000000` in the bad state and the k=0 branch is taken, and
   returns `0x33800000` (about 6e-8, also wrong, but nonzero) in the good state. The input to the
   helper is correct in both; the output is wrong in both.
4. **The helper reads the wrong register.** The conversion binds to a copy inside librocblas itself
   (`gdb_helper_identity.txt`: `__extendhfsf2 in section .text of .../librocblas.so.4`, a LOCAL HIDDEN
   symbol, so it cannot be interposed). Its first instruction is `mov %edi,%ecx`; its caller loads the
   value with `pinsrw $0x0,(%rax),%xmm0`. `__truncsfhf2` has the mirror image: it takes its float input
   from `%xmm0` correctly but never writes `%xmm0`, leaving the half result in `%eax`, while its callers
   read `pextrw $0x0,%xmm0,%eax` (`binary_evidence.txt`).
5. **It comes from the toolchain archive.** rocBLAS links `--rtlib=compiler-rt`
   (`library/CMakeLists.txt:82`), which pulls
   `/usr/lib64/rocm/llvm/lib/clang/19/lib/linux/libclang_rt.builtins-x86_64.a`, owned by
   `rocm-clang-runtime-devel-19-14.rocm6.4.2.fc43`. Its `extendhfsf2.c.o` starts with the same
   `mov %edi,%ecx` (`binary_evidence.txt`).

6. **It needs no GPU to show.** [`patches/half_builtins_abi.c`](../../patches/half_builtins_abi.c),
   built with the same clang and `--rtlib=compiler-rt`, converts half 1.0 to 5.96046e-08 (`0x33800000`,
   the value seen in rocBLAS's good state) and float 1.0 to half `0x0000`. The same source linked
   against libgcc's helpers, or built with `-mf16c` so no helper is called, converts correctly
   (`half_builtins_abi.txt`).

## Why the toolchain is inconsistent (external, from the build log and upstream history)

Reported by a web search the same day and checked against the cited sources by that search; not
re-verified by hand here. Fedora 43's `rocm-compilersupport` spec builds compiler-rt's builtins
twice and both passes write the same archive path. The first pass, configured from
`compiler-rt/lib/builtins`, detects `COMPILER_RT_HAS_x86_64_FLOAT16` and compiles with
`-DCOMPILER_RT_HAS_FLOAT16`; the second, configured from the runtimes tree, fails the detection and
compiles `extendhfsf2.c` without it, and that second archive is the one installed
([Koji build log](https://kojipkgs.fedoraproject.org/packages/rocm-compilersupport/19/14.rocm6.4.2.fc43/data/logs/x86_64/build.log),
around lines 36069, 36931, 39780, 40037, 45855). The failing detection is an LLVM CMake bug
(the test program had no `main()`), fixed upstream in
[llvm/llvm-project#104478](https://github.com/llvm/llvm-project/pull/104478) and backported to
19.1, but not present in ROCm's LLVM fork at `rocm-6.4.2`. The same failure was reported for rocFFT on
Fedora in 2023 ([ROCm/rocFFT#439](https://github.com/ROCm/rocFFT/issues/439)) and for rocBLAS on
Gentoo ([ROCm/rocBLAS#1350](https://github.com/ROCm/rocBLAS/issues/1350)). rocBLAS stopped compiling
with `-mf16c` in 2024, which had turned these conversions into inline instructions and hidden the
broken builtins.

Two further points from a second search the same day. Fedora's own system compiler-rt carried a
patch for the same detection bug from 17.0.0~rc3-2 (August 2023, "Fix FLOAT16 detection"), prompted
by the rocFFT report, but `rocm-compilersupport` for ROCm 6.4 never took an equivalent; ROCm's LLVM
fork has the upstream fix from `rocm-7.0.0`, and no 6.x tag has it. And the later Fedora branches are
correct: disassembling the `rocm-clang-runtime-devel` RPMs from Koji, `__extendhfsf2` reads `%edi` in
19-14.rocm6.4.2.fc43 and `%xmm0` in 20-12.rocm7.1.1.fc44, 22-14.rocm7.2.1.fc45 and 23-3.rocm7.14.0.fc46
(`fedora_builds_extendhfsf2.txt`, checked by hand here). So the problem is specific to Fedora 43's
ROCm 6.4.2 toolchain.

## The fix, measured

`patched_rocblas_results.txt` and `minimal_repro_matrix.txt`, library patched with
[`scripts/fix_half_helpers.py`](../../scripts/fix_half_helpers.py) (byte-identical to the hand patch):

| test | original library | patched library |
|---|---|---|
| minimal reproducer, `tcache_count=1` | 5 of 5 zero | 0 of 5 |
| minimal reproducer, default and `tcache_count=0` | 0 of 5 | 0 of 5 |
| qwen3-8B fp16 compute, 2 chunks | a different wrong value most runs, for example 18.9806 in [`heap_knobs.txt`](../fp16-scalar-2026-09-15/heap_knobs.txt) | **9.1117** four times, including under `tcache_count=1`; 0 of 288 GEMMs zero |
| qwen3-8B f32 compute | 9.0975 | 9.0975 |
| qwen3-14B fp16 compute, 2 chunks | 16.7385 | **7.7645** twice |
| qwen3-14B f32 compute | 7.7600 | 7.7600 |
| qwen2.5-1.5B gate | 8.9442 | 8.9442 |

With both repaired libraries on the path, rocBLAS and llama.cpp's `libggml-hip.so`, and no compute-type
variable set, so the default fp16 path is taken (`both_repaired/`, script `run.sh` there, library
resolution checked with `ldd` in `library_resolution.txt` rather than from the running process):
qwen3-8B 9.1117 on the default path twice, with `f16` set explicitly once and under `tcache_count=1`
once; f32 9.0975; qwen3-14B 7.7645.

9.1117 is the value the system rocBLAS's gfx1010 code objects gave for the same fp16 configuration in
[`../fp16-arch-2026-08-20/`](../fp16-arch-2026-08-20/), which that page used as the evidence pointing
at the gfx1013 kernels. The gfx1010 result was clean because the system rocBLAS 6.4.4 used for it does not
link the helpers in: it imports both from libgcc_s (`__extendhfsf2@GCC_12.0.0`, `binary_evidence.txt`),
whose versions convert correctly ([`patches/half_abi_test.c`](../../patches/half_abi_test.c)). The
architecture was a proxy for which library was loaded.

## Scope beyond rocBLAS

Scanned on the board: the llama.cpp HIP backend built here, `libggml-hip.so`, carries the same broken
pair, called from `ggml_cuda_op_clamp` and `ggml_cuda_op_fill` (float bounds and fill values converted
to half for f16 tensors) and from this repository's own `BC250_ALPHA` debug print in
`ggml_cuda_mul_mat_cublas_impl<F16>`. The rocBLAS, hipBLASLt and hipSPARSELt bundled in the PyTorch
wheel on the board also contain local helpers, but the correct variant (they read `%xmm0`), and
the repair script leaves them alone. Anything else built with this toolchain and half-precision host
arithmetic should be assumed affected until checked.

## What this overturns

- The defect table and INVESTIGATION.md named the native gfx1013 Tensile fp16 kernels as the
  narrowest suspect. The kernels were never wrong; they were given a K=0 problem.
- The "argument corruption" traces of August, where alpha printed as `-0.0075111389` from llama.cpp's
  debug hook, were attributed to calling a device function on the host. The debug hook's conversion
  went through the same broken helper in `libggml-hip.so`, which is the likelier explanation; it was not
  re-run to confirm.
- Every suppressor found earlier the same day in
  [`../fp16-scalar-2026-09-15/`](../fp16-scalar-2026-09-15/), the HIP API trace, a matched no-op at the
  entry of `hipExtModuleLaunchKernel`, disabling the tcache, is consistent with this mechanism, since each changes what earlier code leaves in `%edi`
  before `ConstructTensileProblem` converts alpha.
  That was not shown register by register for each case, so it is stated as consistent rather than
  proven.
- `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` remains a valid workaround and costs nothing here. It is no longer
  the only one.
