`reproduce.sh` run from the repository as shipped, on the board, under the
working configuration. Kept because the script is what the write-up tells a
reader to run first, and it had never been executed from a clean copy of the
repository rather than from the working directory it was developed in.

It passes end to end:

- CU count reads 40 (simd_count 80), so the unlock gate added to the script
  works against the live KFD topology
- the system rocBLAS fails a 512-square SGEMM with
  `rocblas_status_internal_error`, confirming the missing gfx1013 code objects
- the same probe is correct both with `HSA_OVERRIDE_GFX_VERSION=10.1.0` and with
  the native gfx1013 build, which is the pair the write-up describes
- RustiCL compute through the graphics queue is correct
- the HIP compute probe is correct at 1M and at 16.7M threads, the size that
  used to fault under the stock flush and under `sched_policy=2`

Environment as run: kernel 7.1.5, rocblas-6.4.4 as the system package,
mesa 25.3.4, 40 CU, the patched module with `bc250_flush_pasid_kiq=0` and
`bc250_flush_by_runlist=3`, no `sched_policy`.

Run as `bash reproduce.sh` from a clean copy of this repository, with `ROCBLAS_NATIVE` pointing at the native build's install prefix.
