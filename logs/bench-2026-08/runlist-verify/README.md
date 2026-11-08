The A/B for the original unmap-side runlist flush, using the deterministic
allocation-reuse reproducer in `patches/seq_probe.c` (heavy dispatch, free,
realloc, dispatch again).

`param_at_boot.txt` records the parameter value the boot carried. That is the
relevant check: the running module comes from the initramfs, and an earlier
round of this work spent a day measuring a module it was not running.

`seq_boot_runlist1_*.log` are three runs with the flush on, `seq_runlist0_control.log`
is the control with it off, and `moe_load_runlist1.log` is the 10.7 GiB model load
that the same flush made reliable.

This covers the unmap side only. The map side, which is what closes the
fault under allocation churn, is in `../../svm-flush-2026-08/`.
