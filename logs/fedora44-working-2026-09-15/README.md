# Fedora 44 with ROCm 7.1.1: two fixes make it work, 2026-09-15

Follows [`../fedora44-rocm711-2026-09-15/`](../fedora44-rocm711-2026-09-15/), where stock Fedora 44 ran no
model. Two fixes make it work, and every correctness gate then matches Fedora 43 exactly with no binary
repair of rocBLAS.

**The throughput section below is withdrawn.** It reported Fedora 44 as 23 to 29 percent faster. Those runs
were taken before anyone checked the GPU clock: the Fedora 44 upgrade had replaced the oberon governor
configuration with the package default, which overheats this board and makes it oscillate between 1000 and
2000 MHz. Re-measured at the same clock policy, the two systems are within 2 percent of each other on every
model and both backends ([`../fedora44-benchmarks-2026-09-15/clock-corrected/`](../fedora44-benchmarks-2026-09-15/clock-corrected/)).
The section is kept because it is what the alternated boots recorded, and because it shows how convincing a
clock difference looks: five boots, tight spreads, and a control for SELinux mode.

Same board, same kernel (7.1.8-100.fc43 with the production amdgpu module and arguments), the Fedora
44 copy described on the earlier page, and llama.cpp 7ba604f with the three patches built against ROCm
7.1.1 (`build-hip-f44`, no `-mf16c`).

## Fix 1: a native gfx1013 rocBLAS 7.1.1

[`scripts/apply_gfx1013_rocblas711.py`](../../scripts/apply_gfx1013_rocblas711.py) adds gfx1013 to rocBLAS
and Tensile in `ROCm/rocm-libraries` tag `rocm-7.1.1` (commit f322e9ab61): 9 files, 67 lines
(`rocblas_patch_stat.txt`). The files are the same as in PR #8838 and Fedora's Tensile gfx1036 and gfx1153
patches. Every edit is anchored on the existing gfx1012 line and must match once, and a second run
changes nothing. [`scripts/build_rocblas711_gfx1013.sh`](../../scripts/build_rocblas711_gfx1013.sh)
builds it for `GPU_TARGETS=gfx1013` with the monorepo's Tensile, without tests or hipBLASLt.
[`scripts/f44_chroot.sh`](../../scripts/f44_chroot.sh) runs the build inside the Fedora 44 copy from
the running Fedora 43.

Three environment problems had to be cleared, each only in the copy:

1. `rocm-cmake` and `rocminfo` were not installed. Tensile requires `rocm_agent_enumerator` to exist.
2. A hand-written msgpack CMake config carried over from the Fedora 43 build defined only
   `msgpackc-cxx`; Tensile 4.44 asks for `msgpack-cxx`. It now defines both.
3. Without `/dev/shm` in the chroot, joblib falls back to running in-process. On that path Tensile's
   `OverwriteGlobalParameters` clears the global parameter dictionary and refills it from itself,
   leaving it empty (`KeyError: 'PrintIndexAssignments'`). Mounting `/dev/shm` avoids it.

The build finished in about 50 minutes with 56 gfx1013 files in the Tensile library
(`rocblas_build_summary.txt`). It links the half-precision helpers statically like the Fedora 43
build, but these are the correct `%xmm0` variant from Fedora 44's clang 20 runtime, so
[`../fp16-root-cause-2026-09-15/`](../fp16-root-cause-2026-09-15/) does not apply.

**The build was repeated from a clean clone on 16 September**, following the README steps with no manual
intervention, after the script was extended to generate the msgpack config itself and to check for
`/dev/shm` (`clean_rebuild_verify.txt`). It produced 56 gfx1013 files and a library that returns the 8B
gate value 9.1117. Its `TensileLibrary_lazy_gfx1013.dat` is byte-identical to the first build's; the code
object differs, which is expected since build paths are embedded in it. So this part of the recipe is
reproducible, which the Fedora 43 rocBLAS build never was.

## Fix 2: comgr's gfx10 VGPR count

HIP 7.1 reads each device's VGPR budget from comgr's ISA metadata table, where every gfx10 row says 256
total VGPRs. HIP 6.4.2 hard-coded 1024. With 256, `hipOccupancyMaxActiveBlocksPerMultiprocessor` computes
zero blocks for the flash-attention tile kernel and llama.cpp stops on `GGML_ASSERT(max_blocks_per_sm >
0)`. ROCm/llvm-project commit 4f5ae331f659 ("[Comgr] Correct total VGPR counts for gfx10 devices", first
in rocm-7.2.0) changes 13 rows from 256 to 1024: gfx1010 to gfx1036 and the two gfx10 generic rows.
[`scripts/fix_comgr_gfx10_vgprs.py`](../../scripts/fix_comgr_gfx10_vgprs.py) makes the same change in a
copy of `libamd_comgr.so.3` from `rocm-comgr-20-13.rocm7.1.1.fc44`. It locates the rows by their integer
fields, checks each row's name against the 13 the commit changed, and changes 13 bytes
(`comgr_fix_output.txt`). HIP loads comgr with `dlopen`, so `LD_LIBRARY_PATH` selects the copy. The runtime
log then reads `totalNumVGPRs=1024` (`key_lines.txt`).

## Correctness

`results.txt`, script `run.sh`, one run each unless stated:

| run | Fedora 44, both fixes | Fedora 43 corrected stack |
|---|---|---|
| native rocBLAS, stock comgr, qwen2.5-1.5B gate | `GGML_ASSERT(max_blocks_per_sm > 0)` | |
| qwen2.5-1.5B gate, ctx 4096, 8 chunks | 8.9442 | 8.9442 |
| qwen3-8B, default compute type, two runs | 9.1117, 9.1117 | 9.1117 |
| qwen3-8B, `f16` under `tcache_count=1` | 9.1117 | 9.1117 |
| qwen3-8B, `f32` | 9.0975 | 9.0975 |
| qwen3-14B, default compute type | 7.7645 | 7.7645 |
| qwen2.5-1.5B, ctx 1024, 2 chunks, `-fa off` | 8.1702 | 8.1702 ([`../kqv-ladder-2026-08-21/`](../kqv-ladder-2026-08-21/)) |
| `test-backend-ops -b ROCm0` | 12801/12801 passed | 12801/12801 passed |

The Fedora 43 column is [`../fp16-mf16c-2026-09-15/`](../fp16-mf16c-2026-09-15/): the `-mf16c` build with
the repaired native rocBLAS 6.4.2.

## Throughput

[`scripts/os_ab_bench.sh`](../../scripts/os_ab_bench.sh), `llama-bench -mmp 0 -ngl 99 -fa on -p 512 -n 64
-r 5`. Runs alternated across boots in the order Fedora 44, 43, 44, 43. Rounds 2 to 4 started 60 seconds
after a fresh boot; round 1 followed the correctness runs in the same boot. A fifth boot ran Fedora 43
with `enforcing=0`, because the Fedora 44 boots run SELinux permissive. Each boot's OS, library path,
tuned profile, GPU governor state, DPM level, kernel command line and temperatures are in
`bench/*.env`, and they differ only in OS, root subvolume and SELinux mode. Fedora 43 uses the production
build with the repaired rocBLAS 6.4.2; Fedora 44 uses both fixes.

| model | test | Fedora 43 (r2, r4, r5 permissive) | Fedora 44 (r1, r3) | difference |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | pp512 | 809.63, 809.78, 809.84 | 995.96, 996.68 | +23 % |
| qwen2.5-1.5B Q4_K_M | tg64 | 113.62, 113.84, 113.69 | 146.16, 146.19 | +29 % |
| qwen3-8B Q8_0 | pp512 | 243.45, 243.10, 243.35 | 306.62, 306.75 | +26 % |
| qwen3-8B Q8_0 | tg64 | 39.38, 39.38, 39.24 | 40.41, 40.45 | +3 % |

Within each OS the rounds agree to within 1 t/s across boots, so the differences are not boot-to-boot
variation, and SELinux mode is not the cause either. Both controls held, and the conclusion was still wrong:
the `bench/*.env` files record `dpm=auto` and `oberon=active` on every boot, which looked like the same
clock policy on both sides and was not. What they do not record is the governor's configuration file or the
clock actually reached under load. An 8B `pp512` of 178.61 +/- 12.89 measured immediately after the op suite
is left out as a warm-machine outlier.

## What this changes

On Fedora 44 with ROCm 7.1.1, rocBLAS is rebuilt from a script, not from hand edits and the fp16
helper repair is unnecessary. Speed is unchanged. Two custom pieces remain: a patched comgr copy, which ROCm
7.2 would make unnecessary, and the native rocBLAS. The kernel side is unchanged: the same amdgpu module and
parameters as on Fedora 43.

Tested on Fedora 44 afterwards and written up elsewhere: PyTorch, Vulkan, the allocation-churn A/B/A and a
three-hour soak ([`../fedora44-validation-2026-09-15/`](../fedora44-validation-2026-09-15/),
[`../fedora44-soak-2026-09-16/`](../fedora44-soak-2026-09-16/)). Not tested: the stock comgr with a llama.cpp
workaround instead of the patched copy.
