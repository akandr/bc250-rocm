# Context ceilings on Fedora 44, and two depths that no longer work, 2026-09-16

[`scripts/ceilings_f44.sh`](../../scripts/ceilings_f44.sh) on the default Fedora 44 configuration with the
GPU clock policy corrected (1500 MHz), ROCm backend, mmap on as `llama-bench` defaults, one depth per
invocation, `-r 2`. Decode rate in tokens per second after the given context depth; the Fedora 43 column is
the August ladder on the front page of that time, measured the same way.

| model | depth | Fedora 44 | Fedora 43, August |
|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 8192 | 98.93 | 84.8 |
| | 16384 | 86.83 | 73.6 |
| | 32768 | 68.88 | 62.7 |
| | 131072 | **fails** | 27.6 |
| qwen3-8B Q8_0 | 8192 | 26.70 | 22.6 to 23.9 |
| | 16384 | 20.07 | 16.2 to 18.7 |
| qwen3-14B Q4_K_M | 8192 | 12.39 | 12.6 |
| | 16384 | **fails** | 7.3 |
| qwen3.8-27B UD-IQ3_XXS | 8192 | 7.14 | 7.0 |
| | 16384 | fails | fails |

## The two failures are not a Fedora 44 regression

Both failing cases print nothing and dump core. The kernel says why: `amdgpu: SVM mapping failed, exceeds
resident system memory limit`, 33510 lines of it in that boot (`svm_limit.txt`), followed by a segfault
inside `libhsa-runtime64.so`, which does not handle the failed mapping.

Booting Fedora 43 an hour later, with `ollama` stopped and 12 GiB free, qwen3-14B at depth 16384 fails there
too, with the same kernel message. So the August figure of 7.3 t/s is not reproducible on the system that
produced it, and the difference is not between the two distributions. What changed since August is not
established; the KFD limit depends on resident system memory, so anything holding memory at the time shifts
it.

qwen2.5-1.5B at 131072 was not retested on Fedora 43: the attempt ran after the 14B failure in the same
boot, and the runtime then reported no ROCm device at all. That is the one recorded case here of a failure
affecting a later process. Killing the stuck benchmark restored the GPU without a reboot, which separates
this from the page-fault wedge, where a reboot is the only way back.

The rates that do complete are 10 to 18 percent above the August figures on the 1.5B and 8B and equal on the
14B and 27B. The August ladder was measured on a different day with a different llama.cpp build, and the
1.5B and 8B rows there also predate the clock check, so the comparison is indicative, not a
measurement of Fedora 44 against Fedora 43.

## What the limit is, and what removing it does (`summary.txt`, `limit_on.log`)

The message comes from `svm_range_new()` in `drivers/gpu/drm/amd/amdkfd/kfd_svm.c`, which charges every SVM
range against `kfd_mem_limit.max_system_mem_limit` in `amdgpu_amdkfd_gpuvm.c`. That limit is 63/64 of total
RAM minus 1.5 GiB and is computed from total, not free, memory. On this board the driver reports it
directly:

    # cat /sys/kernel/debug/kfd/mem_limit
    System mem used 0M out of 13422M
    TTM mem used 0M out of 16384M

13422 MiB is 63/64 of 15196 MiB minus 1.5 GiB, so the arithmetic matches. Reading it immediately after the
qwen3-14B failure at depth 16384 shows the ceiling being hit exactly: `System mem used 13412M out of
13422M`. The `ttm.pages_limit=4194304` boot argument this repository uses raises the *other* counter, to
16384 MiB, and cannot help here: the SVM path charges system memory only. `HSA_XNACK=1` would skip the
charge entirely, but `kfd_process_xnack_mode()` refuses XNACK for every GC 10.1.x part, so it is not
available on gfx1013.

The kernel returns `-ENOMEM` cleanly. What crashes is the runtime: `segfault at 20 ... in
libhsa-runtime64.so.1.18.0`. Fault address 0x20 in that library version is the signature of a ROCr defect
fixed upstream in February 2026 (rocm-systems PR #2850, a scope guard releasing scratch that was never
allocated), which is not in the ROCm 7.1.1 packages here. That is a match on the symptom, not a proof that
this is the same call path.

`amdgpu.no_system_mem_limit=1`, writable at runtime through
`/sys/module/amdgpu/parameters/no_system_mem_limit`, disables the check. Tested here:

| run | limit enforced (default) | limit disabled |
|---|---|---|
| qwen3-14B, depth 16384 | segfault after about 30 s | no crash; still running when a 1800 s timeout killed it, `System mem used 14636M out of 13422M` |
| qwen2.5-1.5B, depth 131072 | segfault | no crash; still running when a 3000 s timeout killed it |

So the flag converts the crash into a run that does not finish: past the limit the working set no longer
fits in 14.8 GiB and the machine swaps. The limit is protecting something real, and it is left at its
default here. The accounting released correctly after every run, crash or timeout, `System mem used 0M`, so
a failed run does not leak the budget.
