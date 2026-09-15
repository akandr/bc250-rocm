# Suspend as a hardware reset, 2026-09-15

Tried by hand, recorded here with the commands. The reasoning: every path that re-initialises this
GPU without a hardware reset hangs, and suspend is the one operation that might power the GFX block
down for real (the SMU message list has `ConfigureS3PwrOffRegisterAddress`).

The platform offers no S3. `/sys/power/state` lists `freeze mem disk`, `mem_sleep` offers only
`[s2idle]`, and ACPI reports `(supports S0 S4 S5)`.

- `s2idle_healthy.*`: amdgpu bound, healthy GPU, `rtcwake -m freeze -s 20`. The kernel log reaches
  `PM: suspend entry (s2idle)` and `Suspending console(s)`, then nothing; the board did not answer
  for three minutes and needed a power cycle. The persistent journal holds nothing after the marker
  (`s2idle_healthy.journal_prevboot.txt`), since suspend froze journald before the lines were flushed.
- `s2idle_nogpu_control.txt`: amdgpu unbound first (which is harmless, see
  [`../rebind-recovery-2026-09-15/`](../rebind-recovery-2026-09-15/)), same command. Also never came
  back.

Each was tried once, so this is one capture per arm rather than a rate, which is enough to rule
the path out for remote use but not to characterise it. So s2idle does not return on this board even
without the GPU driver, and whether that is the
platform failing to wake or the Realtek NIC failing to resume cannot be told apart without a local
console. Either way it is not usable as a remote recovery. S4 (a 16 GiB swapfile exists) would power
everything off and is in effect a reboot that keeps process state; it was not tried.
