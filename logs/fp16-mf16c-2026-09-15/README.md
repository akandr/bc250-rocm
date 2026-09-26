# llama.cpp built with -mf16c, and which libraries carry the broken helpers, 2026-09-15

Follows [`../fp16-root-cause-2026-09-15/`](../fp16-root-cause-2026-09-15/). Kernel 7.1.8, production
amdgpu settings, llama.cpp 7ba604f with the three patches, `HSA_ENABLE_SDMA=0`.

## Which libraries on the board are affected

[`scripts/scan_half_helpers.sh`](../../scripts/scan_half_helpers.sh) over 4688 shared objects under
`/usr/lib64`, `/opt/rocm*` and the home directory (`library_scan.txt`). Every library with a broken
local `__extendhfsf2` was built on the board with the Fedora 43 ROCm toolchain: the native rocBLAS
copies and the two llama.cpp master HIP builds that contain the f16 clamp and fill paths. Nothing
under `/usr/lib64` is affected. The PyTorch wheels' rocBLAS, hipBLASLt and hipSPARSELt carry the
correct variant, reported here as `?(mov %eax,%edx)` because the scan prints the instruction after
the `%xmm0` read.

## llama.cpp with -mf16c

The HIP backend configured as before plus `-DCMAKE_HIP_FLAGS=-mf16c` (`cmake_and_bench.txt`), with
`--offload-arch=gfx1013` and the other flags unchanged. The resulting `libggml-hip.so` has no half
helper symbols and no calls to them; the five call sites of the previous build are inline F16C
instructions (`helper_scan.txt`).

Perplexity, with the repaired rocBLAS from the root-cause page and no other change
(`results.txt`, script `run.sh`):

| run | perplexity |
|---|---|
| qwen2.5-1.5B gate, ctx 4096, 8 chunks | 8.9442 |
| qwen3-8B, default compute type, two runs | 9.1117, 9.1117 |
| qwen3-8B, `f16` under `tcache_count=1` | 9.1117 |
| qwen3-8B, `f32` | 9.0975 |
| qwen3-14B, default compute type | 7.7645 |

Throughput, qwen2.5-1.5B Q4_K_M, `llama-bench -p 512 -n 64 -r 3`, run in the order previous build,
`-mf16c`, previous, `-mf16c`: pp512 806.36 and 808.20 against 807.99 and 807.61 t/s, tg64 113.69 and
113.72 against 113.67 and 113.60. No difference. This model takes neither the rocBLAS path nor the
f16 clamp and fill paths, so it shows the flag costs nothing elsewhere instead of measuring those
paths.

## A second symptom: f16 CLAMP

`test-backend-ops -b ROCm0 -o CLAMP` compares the GPU result against the CPU. In the previous build,
whose `ggml_cuda_op_clamp` converts the float bounds to half through the broken helper, all three
f16 cases fail with error `inf` and the three f32 cases pass, 3 of 6, at the default allocator
setting and under `tcache_count=1`. The `-mf16c` build passes 6 of 6 in both, and the previous build
passes 6 of 6 twice with its `libggml-hip.so` repaired by `fix_half_helpers.py`
(`test-backend-ops/`). FILL passes 4 of 4 everywhere, but its tests cover only f32, so its f16 path
is not exercised by them.

## The full op suite on the corrected stack

`test-backend-ops -b ROCm0` with no op filter, `-mf16c` build, repaired rocBLAS
(`test-backend-ops/full_suite_f16c.txt.gz`, one run): 12801/12801 tests passed against the CPU, 7990
reported as not supported by the backend and skipped, both backends passed, exit code 0.

## What this leaves

llama.cpp can be fixed at build time with one flag, which fixes f16 CLAMP as well as the fp16 GEMM. rocBLAS still needs the binary repair: its build
directories kept no object files, so it could not be relinked, and a full rebuild with `-mf16c` was
not run.
