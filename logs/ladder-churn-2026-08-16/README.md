# Cross-kernel churn evidence, 2026-08-16

Second, independent pass over the kernel ladder, with one directory per boot so
every row of the cross-kernel table in the top-level README has its own files.
An earlier pass reused a single directory and only the last rung survived, so
these runs replace it. All results below replicate that earlier pass.

Each directory holds:

- `config.txt`: kernel release, the `amdgpu.bc250_flush_by_runlist` value in
  force, whether the map-side hook is present in the module that actually
  loaded, and the SIMD count read from the KFD topology
- `result.txt`: exit code, elapsed seconds, whether the sweep completed, fault
  count, and the verdict
- `churn.log`: the sweep output

The workload is the allocation-churn sweep, roughly ten minutes when healthy.
The timeout is 1500 s.

## Results

| directory | runlist | map hook in loaded module | SIMDs | verdict |
|---|---|---|---|---|
| `7.1.5-100.fc43.x86_64` | 3 | yes | 80 | CLEAN, 591 s |
| `7.1.2-200.fc43.x86_64` | 1 | no | 80 | FAULT, 11 s |
| `7.0.13-100.fc43.x86_64` | 1 | no | 80 | FAULT, 11 s |
| `6.19.14-200.fc43.x86_64` | 3 | yes | 80 | CLEAN, 592 s |
| `6.18.16-200.fc43.x86_64` | 1 | no | 80 | STALL, rc 124 at 1500 s |
| `6.18.9-200.fc43.x86_64-nofix` | 1 | no | 80 | STALL, rc 124 at 1500 s |
| `6.18.9-200.fc43.x86_64-withfix` | 3 | yes | 80 | CLEAN, 590 s |

## Why the 6.18.9 pair is here

The two 6.18 rungs fail by stalling rather than by faulting. A stall is only the
absence of completion, so on its own it does not establish a cause: it fits the
defect, and it equally fits a sweep that simply needs longer on those kernels.
No 6.18 kernel had been measured with the fix present, so nothing distinguished
the two readings.

The pair settles it. Same kernel, same board, same boot arguments, same module
source including the unconditional 40-CU unlock. The only differences are the
map-side hook in the module and the runlist value. Without them the sweep does
not finish in 1500 s; with them it finishes in 590 s.

The healthy runtime is also worth noting: 590 s here, 591 s on 7.1.5, 592 s on
6.19.14. The timeout carried about 2.5 times the headroom a healthy sweep needs,
on the same kernel that stalled, which rules out the timeout being too short.

The 6.x stall and the 7.x fault are therefore the same defect presenting
differently, not two separate problems.

## Reading the SIMD count

The unlock is a module parameter, `bc250_cc_write_mode`, and
`/etc/modprobe.d/bc250-40cu.conf` sets it to 3 from inside the initramfs. So a
kernel command line with no mention of it still boots with the unlock active,
and the cmdline alone cannot confirm the state either way. The SIMD count is
the check. Read it across all KFD topology nodes and take the maximum:

    grep -h simd_count /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u

Node 0 is the CPU and reports 0. Reading only the first node gives 0 on a board
where the unlock is live and working.

## Module state left behind

The 6.18.9 module under `/lib/modules` carries the map-side hook after this
run, and the original is saved alongside it. Every other rung is untouched. The
board's production kernel is 7.1.5.
