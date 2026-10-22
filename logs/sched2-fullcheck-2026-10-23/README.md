# The software-scheduler rows on the same stack, 2026-10-23

The policy-2 half of the CU-by-scheduler table, rerun on the stack and with the checks of the
hardware-scheduling rows in `../hws-fullcheck-2026-10-22/` (`7.1.5-40cu-hws-rl3`,
`7.1.5-24cu-hws-rl3`): Fedora 44, Linux 7.1.5, the KIQ bypass and the rebuild on map and unmap,
with `amdgpu.sched_policy=2` the only difference (the rebuild parameter is set as in the hardware
rows, but the rebuild does nothing without the hardware scheduler). `header.txt` records the command line, module
parameters and SIMD count. Script `scripts/sched2_boot.sh` (short timeouts, since this policy is
expected to wedge), boots from `scripts/sched2.conf` via `scripts/hws_fullcheck_orchestrate.sh`
(`BATTERY=sched2_boot.sh RDIR=sched2`).

| boot | CUs | probe, 1M/4.2M/8.4M threads | rocBLAS sgemm N=256, 1024 | kernel log during the battery |
|---|---|---|---|---|
| `sc1-7.1.5-40cu-sched2` | 40 | all three hang (killed at 120 s) | both wedge | 18 `cp queue preemption time out`, 5 `Failed to evict process queues` |
| `sc2-7.1.5-24cu-sched2` | 24 | all three hang (killed at 120 s) | both wedge | 18 `cp queue preemption time out`, 5 `Failed to evict process queues` |

With the hardware scheduler the same module, stack and checks gave correct results at both CU
counts and no kernel-log line, so on one stack the wedge follows the scheduler policy and not the
CU count. The original Fedora 43 cells are in
`../kernel-equivalence-2026-08-17/factorial-cells/`.
