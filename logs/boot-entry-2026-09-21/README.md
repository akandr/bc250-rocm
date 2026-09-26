# A plain reboot came up on the wrong boot entry, 2026-09-21

A reboot taken to get a clean fault count for a soak came back with a kernel command line that was not
the one this page documents. The board had been up three days on the right one, so nothing had been
edited; the bootloader had two entries for the same kernel and the saved default pointed at the
wrong one.

| | index 0, the documented entry | index 1, what booted |
|---|---|---|
| `amdgpu.bc250_flush_by_runlist` | 3 | absent |
| `amdgpu.bc250_flush_pasid_kiq` | 0 | absent |
| `amdgpu.sched_policy` | unset | **2** |
| `amdgpu.gpu_recovery` | 0 | 0 |
| `amdgpu.bc250_cc_write_mode` | 3 | 3 |

Both of the differences matter, and both are in this page's own known-issues table. Without
`bc250_flush_by_runlist=3`, freed and reallocated GPU memory faults. And `sched_policy=2` selects the
software scheduler, which wedges sustained compute; the table says to leave it unset. So a single plain
reboot put the board into a configuration that is documented to fail under exactly the load that was
about to be started.

Six of the nine entries carry the same wrong arguments, because they predate the recipe and were never
cleaned up. `grubby --set-default-index=0` fixed it and the next boot came up correct
([`cmdline-after-fix.txt`](cmdline-after-fix.txt), [`grubby-entries.txt`](grubby-entries.txt)).

The lesson is small and cheap, and the tool for it already existed: [`reproduce.sh`](../../reproduce.sh)
reads both `bc250_flush_by_runlist` and `sched_policy` and would have refused this configuration in its
first few lines. **Run it after any reboot, before trusting a measurement taken on that boot.** Nothing in this repository was measured on the wrong entry, because the check was made
before the soak started, but the only reason the check happened is that a fault count was being read at
the same time.
