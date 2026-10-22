# Correctness with checks that can fail, 2026-10-22

Fedora 44 user space (ROCm 7.1.1, the native gfx1013 rocBLAS in `/opt/bc250-rocm`), hardware
scheduling and the KIQ bypass (`amdgpu.bc250_flush_pasid_kiq=0`) throughout. Every directory is one
boot; `header.txt` records the kernel, the command line, the module parameters, the KFD SIMD count
(two per CU) and the rocBLAS library actually loaded.

Earlier correctness evidence rested partly on checks that could not see the failure: the SGEMM
programs check two elements of an output buffer reused across iterations, and several kernel-log
counts matched neither the hardware scheduler's failure messages nor the `[gfxhub] page fault`
line. These runs replace them:

- `patches/compute_probe.c`, the bare probe: zero-fills its buffer, then checks every element.
- `patches/rocblas_fullcheck.cpp`: fills the GEMM output with NaN before every call and compares
  every element exactly with a CPU reference; inputs are small integers, so every sum is exact in
  each format. Its negative controls (`FULLCHECK_CONTROL=skip` leaves one row unwritten,
  `=wrong` doubles the result) fail as they should; see the `control` runs in
  `7.2.5-40cu-hws-rl3-rocblas/battery.log`.
- The kernel log written during each battery is kept (`kernel_log_battery.txt`), and
  `kernel_log_summary.txt` holds its amdgpu/KFD lines matching page faults, preemption,
  eviction, the "unrecoverable state" warning, timeouts, resets and errors. The `ab11` boot shows
  that this search does fire on a real GPU page fault.

Scripts: `scripts/rocblas_fullcheck.sh` (the large rocBLAS battery, no reboot),
`scripts/hws_fullcheck_boot.sh` (one boot's battery), `scripts/hws_fullcheck_orchestrate.sh` with
`scripts/hws_fullcheck_ab.conf` (boot switching; the boot entries are backed up and restored
byte for byte at the end).

## rocBLAS, full working configuration, Linux 7.2.5

`7.2.5-40cu-hws-rl3-rocblas/`: sgemm, dgemm, hgemm and gemm_ex (fp16 in and out, fp32 compute),
A normal and transposed, N = 256, 512, 1000, 1024, 2048, 3001, 4096, 8192, five iterations each,
then 200 iterations of sgemm N=4096 and of gemm_ex fp16 N=4096 (transposed A). All 66 runs
correct in every element, about 10.7 billion elements in total; the four negative controls failed;
no kernel-log line during the battery.

## The bare probe and the runlist rebuild

`amdgpu.bc250_flush_by_runlist`: 0 no rebuild, 1 rebuild on unmap, 3 on unmap and map. The probe
runs at 4096, 16384 and 32768 blocks of 256 threads (three dispatches each), and again at 32768
after the GEMMs.

| rebuild | boots | probe |
|---|---|---|
| none | 5: four on 7.1.5 (`7.1.5-40cu-hws-bypassonly`, `ab01`, `ab05`, `ab08`), one on 7.2.5 (`ab12`); SDMA on in three, off in two | wrong elements with no error raised in two of the four runs of every boot, both at 8.4M threads, first wrong element 0 (the pre-kernel value); in every boot the first of those runs then aborted with glibc's "double free or corruption" |
| unmap | 4 on 7.1.5 (`ab02`, `ab04`, `ab07`, `ab10`) | correct |
| unmap | 1 on 7.2.5 (`ab11`) | one run took a GPU page fault ("Memory access fault by GPU", 14 matching kernel-log lines); the others correct |
| unmap | 1 on 7.1.2 (`7.1.2-40cu-hws-rl1`), whose module carries the earlier version of the rebuild | as with no rebuild: two runs wrong, one aborted |
| unmap and map | 8: 7.1.5 at 40 CUs (`7.1.5-40cu-hws-rl3`, `ab03`, `ab06`, `ab09`, `ab14`) and 24 CUs (`7.1.5-24cu-hws-rl3`), 7.1.8, 7.2.5 (`ab13`) | correct, no kernel-log line |

The rocBLAS GEMMs in these boots (sgemm N=256 to 4096, sgemm and gemm_ex fp16 with A transposed,
dgemm N=2048, ten iterations each, and 200 iterations of sgemm and gemm_ex fp16 at N=4096) were
correct in every element in all of them, the failing ones included, so on this stack the probe and
not GEMM is the sensitive test. SDMA on or off made no difference.

Every wrong result came on the second or third dispatch of an invocation, never the first; the
one-dispatch follow-up is in `../probe-dispatch-2026-10-22/`.

Neither the August factorial (`../factorial/`) nor the August 7.1.2 rung (`../ladder-2026-08-13/`)
recorded the runlist parameter; the 7.1.2 rung's probe at 8.4M threads was correct on the Fedora 43
stack, where the 7.1.2 boot here, with the earlier rebuild, is not.
