# Fedora 44 userspace with ROCm 7.1.1 on the production kernel, 2026-09-15

**Superseded the same day by [`../fedora44-working-2026-09-15/`](../fedora44-working-2026-09-15/):** a native
gfx1013 rocBLAS 7.1.1 and a one-row-per-device comgr correction make this stack run correctly and
faster. The flash-attention failure below is comgr reporting 256 VGPRs for gfx10 devices.

Fedora 44 was tried because its ROCm toolchain has correct half-precision helpers
([`../fp16-root-cause-2026-09-15/`](../fp16-root-cause-2026-09-15/)). **Out of the box it runs no model
on this board.** The system rocBLAS 7.1.1 fails every GEMM, f32 included, and HIP 7.1.1 trips an
assertion in llama.cpp's flash-attention path.

## Setup

Fedora 43 was left in place. The btrfs `root` subvolume was snapshotted to `root-f44`, and only the copy
was upgraded, with `dnf --installroot=<copy> --releasever=44 --setopt=installonly_limit=0
--exclude='kernel*' distro-sync`: 83 packages installed, 2246 upgraded, 1 downgraded, 4 GiB downloaded,
exit code 0. Scriptlets that needed `/proc` were re-run in a chroot afterwards. The copy then
received:
- its own `fstab` root line;
- the navi12 SDMA microcode the upgrade had replaced;
- a one-time boot entry (`boot_entry.txt`): kernel 7.1.8-100.fc43 with the production amdgpu module
  and arguments, `rootflags=subvol=root-f44`, and `enforcing=0` because file labels from an
  installroot upgrade are not trustworthy.

`/boot` and the ESP were checksummed before and after the upgrade and did not change. Swap had to be
turned off for the snapshot, since btrfs refuses to snapshot a subvolume holding an active
swapfile, and the copy's swapfile was deleted so the original could be re-enabled.

The Fedora 44 boot came up with 40 CU, the production parameters and no failed units
(`environment.txt`): ROCm 7.1.1 (`rocm-hip`, `rocm-runtime`, `rocblas`), clang 20, gcc 16 and Mesa 26.1.8.
llama.cpp 7ba604f with the three patches was built against it, without `-mf16c`.

## Results

`results.txt`, script `run.sh`, no `LD_LIBRARY_PATH`, so the system rocBLAS 7.1.1 is used:

| run | result |
|---|---|
| qwen2.5-1.5B gate, ctx 4096, `-fa on` | abort: `GGML_ASSERT(max_blocks_per_sm > 0)` in `launch_fattn` (`gate_q15_abort.txt`) |
| qwen2.5-1.5B, ctx 1024, `-fa off`, two runs | abort: `CUBLAS_STATUS_INTERNAL_ERROR` in `hipblasGemmBatchedEx` |
| qwen3-8B default, two runs; `f16` under `tcache_count=1`; `f32` | abort: `CUBLAS_STATUS_INTERNAL_ERROR` in `hipblasGemmEx` or `hipblasSgemm` (`gemm_abort.txt`) |
| qwen3-14B default | abort, same error |
| `llama-bench`, qwen2.5-1.5B, `-fa on` and `-fa off` | abort |
| `test-backend-ops -b ROCm0` | 3858 cases OK, none failing, then abort at the first f32 `MUL_MAT` that reaches rocBLAS (`test_backend_ops_tail.txt`) |

Two separate problems:

- **rocBLAS.** The Fedora 44 rocBLAS package ships no gfx1013 files. The gfx1013 names in its library
  directory are symlinks to the gfx1010 files, made by hand in July on Fedora 43 and carried over by
  the snapshot. Through them, rocBLAS 7.1.1 fails even f32 GEMMs, as the Fedora 43 system rocBLAS
  does for several inference shapes. The Fedora 44 route therefore needs a native gfx1013 rocBLAS
  7.1.1 build, which was not attempted.
- **Flash attention.** llama.cpp sizes the tile kernel's launch from
  `hipOccupancyMaxActiveBlocksPerMultiprocessor`, which returned a non-positive value under HIP 7.1.1 on
  this device, so the assertion fires before any kernel runs. Under HIP 6.4.2 the same code works.
  Why the runtime returns that was not investigated.

Every op before the first rocBLAS call passed against the CPU, including the GPU kernels compiled
by clang 20. So the HIP 7.1.1 runtime and the production kernel do run compute correctly for those
ops.

## State afterwards

The board was booted back into Fedora 43 with the default entry, whose one-time pointer had been
cleared. The checks came back as before: `subvol=root`, SELinux enforcing, Mesa 25.3.4, ROCm 6.4.2,
swap at priority 10, 40 CU, and the qwen2.5-1.5B gate at 8.9442 with the production build. The
`root-f44` subvolume and its boot entry remain on the board as an option; the default is unchanged.
