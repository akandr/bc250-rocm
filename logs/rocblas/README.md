# Native rocBLAS SGEMM sweeps, July 2026

Two sweeps from the original investigation, kept as recorded.

`sgemm_sweep_stock_40cu.log`: the stock module at 40 CU.

`sgemm_sweep_patched_24cu.log`: the patched module, which on that boot came up
at 24 CU, where every size wedged including N=256.

**Both files carry an explanation in their headers that later work retracted.**
They state that the 40-CU unlock write lives inside
`gfx_v10_0_kiq_reset_hw_queue()` and therefore stops firing once the corrected
flush removes the KIQ activity that triggered it. Two things are now known:

- No released version of the community unlock patch writes those registers from
  the reset path; every version writes them from `gfx_v10_0_get_cu_info()`. The
  tree used here almost certainly carried a misapplied variant, a mistake the
  companion project's own notes warn about, whose symptom is exactly a module
  that loads while the board stays at 24 CU.
- The wedge in the 24-CU sweep is not caused by the CU count. That boot carried
  `amdgpu.sched_policy=2`. Crossing CU count with scheduler policy shows both
  hardware-scheduling cells clean at 24 and 40 CU and both policy-2 cells
  wedging at both, and at 24 CU with hardware scheduling the board passes a full
  battery including perplexity
  (`../kernel-equivalence-2026-08-17/`).

The measurements in these files reproduce; the reasoning in their headers does
not. They are left unedited because they are the record of what was observed.

Historical rocBLAS probe output, collected by `patches/rocblas_probe.c` invocations before `reproduce.sh` wrapped them.
