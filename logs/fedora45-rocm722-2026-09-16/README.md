# Fedora 45 with ROCm 7.2.2: one custom piece instead of two, 2026-09-16

Tested the same way Fedora 44 was: the `root-f44` btrfs subvolume snapshotted to `root-f45`, the copy
upgraded with `dnf --installroot --releasever=45 distro-sync` (kernels excluded, third-party repositories
disabled, `--allowerasing` after removing a Fedora 43 `kernel-tools` left behind by the earlier upgrade),
its own `fstab` root line, the navi12 SDMA microcode copied in again, a relabel, a fresh swapfile and its
own boot entry on the unchanged 7.1.8 kernel. Fedora 44 stays the default and is one boot entry away.

Versions (`environment.txt`): ROCm 7.2.2 with comgr 22-14.rocm7.2.1 and rocBLAS 7.2.0, glibc 2.44, Mesa
26.2.0. The governor configuration survived this upgrade, unlike the Fedora 44 one, and was checked before
booting. The `/opt/bc250-rocm` override from Fedora 44 was disabled here so nothing built against ROCm 7.1
could be loaded by mistake.

## The comgr patch is no longer needed

On stock ROCm 7.2.2 the qwen2.5-1.5B gate, which runs flash attention, returns 8.9442 with no override in
place. On 7.1.1 the same run stopped at `GGML_ASSERT(max_blocks_per_sm > 0)` until
[`scripts/fix_comgr_gfx10_vgprs.py`](../../scripts/fix_comgr_gfx10_vgprs.py) was applied. The script
confirms why: run against Fedora 45's `libamd_comgr.so.3`, it reports no rows matching the broken pattern,
because ROCm 7.2 carries the upstream correction.

## rocBLAS still needs the native build

Fedora 45's rocBLAS 7.2.0 ships the same gfx1010 symlinks in place of gfx1013 files, and the qwen3-8B gate
against it fails with `CUBLAS_STATUS_INTERNAL_ERROR`.
[`scripts/apply_gfx1013_rocblas711.py`](../../scripts/apply_gfx1013_rocblas711.py) applies unchanged to the
`rocm-7.2.2` tag, every anchor matching once, and
[`scripts/build_rocblas711_gfx1013.sh`](../../scripts/build_rocblas711_gfx1013.sh) built it with only the
prefix changed: 56 gfx1013 files, exit 0.

## Correctness

Gates with the native rocBLAS 7.2.2 and stock everything else, default compute type:

| gate | Fedora 45 | Fedora 44 |
|---|---|---|
| qwen2.5-1.5B, ctx 4096, 8 chunks | 8.9442 | 8.9442 |
| qwen3-8B, ctx 2048, 2 chunks | 9.1117 | 9.1117 |
| qwen3-14B, ctx 2048, 2 chunks | 7.7645 | 7.7645 |

The 8B value is the fp16 path, so the toolchain defect is absent here too. `test-backend-ops -b ROCm0`
passes 12801 of 12801 (`tbo_full.txt.gz`), the same count as on Fedora 44.

## Throughput

Same llama.cpp binaries as the Fedora 44 campaign, same clock policy, one test per invocation, medians of
nine samples:

| model | ROCm pp512 | | ROCm tg64 | | Vulkan pp512 | | Vulkan tg64 | |
|---|---|---|---|---|---|---|---|---|
| | **F45** | F44 | **F45** | F44 | **F45** | F44 | **F45** | F44 |
| qwen2.5-1.5B | 792.95 | 792.88 | 120.30 | 117.49 | 1787.28 | 1849.10 | 219.58 | 212.29 |
| qwen3-8B | 243.47 | 243.57 | 39.35 | 38.94 | 396.11 | 394.77 | 39.14 | 39.07 |
| deepseek-r1-14B | 94.36 | 94.45 | 21.58 | 21.38 | 196.06 | 199.78 | 35.35 | 35.06 |
| qwen3-14B | 96.40 | 96.78 | 21.93 | 21.80 | 201.49 | 204.67 | 35.09 | 34.76 |
| qwen3.6-35B MoE | 289.73 | 289.66 | 34.92 | 34.29 | 444.35 | 457.17 | 88.38 | 87.30 |
| qwen3.8-27B | 69.41 | 69.43 | 7.89 | 7.85 | 104.60 | 104.95 | 17.70 | 17.62 |

ROCm prefill is identical, within 0.4 percent on every model. ROCm decode is 0.5 to 2.4 percent higher on
Fedora 45. Vulkan on Mesa 26.2.0 prefills 1.6 to 3.4 percent slower than Mesa 26.1.8 on four models and
decodes 0.3 to 3.4 percent faster. None of these is large, and after the clock episode small differences on
this board should be treated as provisional until repeated in alternated boots, which these were not.

## What it means for the recipe

Fedora 45 removes the comgr patch and leaves one custom userspace piece, the native gfx1013 rocBLAS, plus
the kernel module. Speed is unchanged. Fedora 45 was still a development release on the day of this test,
which is why it is recorded here, not recommended.
