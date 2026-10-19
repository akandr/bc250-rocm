# Which GPU process endings leave system memory behind, 2026-10-18

## What was open

Three times in two days the board started a campaign with far less memory than it has. On the morning of
2026-10-17 `MemAvailable` read 4.0 GiB instead of about 14.4, with no process holding anything, KFD's own
accounting at zero (`/sys/kernel/debug/kfd/mem_limit`), the GPU's GTT counter at 13 MiB and the TTM page
pool empty; a reboot gave the 14.4 back. The same afternoon it fell from 14.1 to 1.2 GiB right after a
ROCm run aborted at load with a core dump (deepseek-14B behind 24576 tokens,
[`ladders-rest-2026-10-17/`](../ladders-rest-2026-10-17/)). And on the morning of 2026-10-18 it read 4.4 GiB
again after a night in which the only GPU user had been ollama's Vulkan runner, called once by a timer job
on an 8B model (8.2 GiB of weights and 1.4 GiB of cache on `Vulkan0`, per ollama's log). Which ending leaks?

## Method

[`scripts/leak_test.sh`](../../scripts/leak_test.sh) on a freshly rebooted board, every other GPU user
stopped, reading `MemAvailable`, KFD's `mem_limit` and `mem_info_gtt_used` before and ten seconds after
each of:

1. a ROCm `llama-bench` on the 1.5B that exits normally
2. a Vulkan one that exits normally
3. a ROCm one killed with SIGKILL while generating, 1.4 GiB mapped
4. a Vulkan one killed with SIGKILL while generating, 1.1 GiB mapped
5. a ROCm run that cannot create its context (the 8B behind 40960 tokens)
6. ollama 0.20.0 loading `qwen2.5:1.5b` (its Vulkan runner, 1.75 GiB on the GPU) and unloading it, first by
   its own scheduler after `keep_alive` expired, then with `keep_alive: 0`

All `llama-bench` runs above used `-mmp 0`, as the campaigns do. A second round by hand afterwards repeated
the Vulkan exit, the Vulkan SIGKILL and a ROCm exit with mmap on, ollama's way of loading.

## Result

| ending | `MemAvailable` before | after | leaked |
|---|---|---|---|
| ROCm, normal exit | 14506 MiB | 14506 | 0 |
| Vulkan, normal exit | 14506 | 14502 | 0 |
| ROCm, SIGKILL while running | 14512 | 14512 | 0 |
| Vulkan, SIGKILL while running | 14512 | 14516 | 0 |
| ROCm, context creation fails | 14516 | 14578 | 0 |
| **ollama load and unload, `qwen2.5:1.5b`** | 14578 | 12997 | **about 1.5 GiB** |
| ollama, a second cycle with `keep_alive: 0` | 12997 | 12976 | 0 more |
| Vulkan with mmap, normal exit | 13052 | 13036 | 0 |
| Vulkan with mmap, SIGKILL | 13036 | 13045 | 0 |
| ROCm with mmap, normal exit | 13045 | 13044 | 0 |

**`llama-bench` gives everything back however it ends**, on either backend, with or without mmap, killed or
not, and so does a run that never gets its context. **ollama's runner does not**: one load of a 1.75 GiB
model and its unload left about 1.5 GiB of system memory unaccounted for, with KFD at zero, GTT back at
12 MiB, the TTM pool empty and no runner process left, and the overnight 8B call (9.6 GiB on the GPU) matches
the 10 GiB missing both mornings. The second cycle in the same ollama session cost nothing more, so it is
per model loaded, not per unload. The memory came back only with a reboot. What in ollama's runner holds it
is not identified here; its own Vulkan ggml build and its way of mapping the model are the differences from
`llama-bench`, and the kernel is 7.2.5 with the BC-250 patches, so the fault may be on either side.

The 2026-10-17 afternoon drop, 13 GiB gone right after a ROCm abort with a core dump, is not reproduced by
anything above: a context-creation failure returns cleanly, and an 8B abort after the fill on 2026-10-18
(`ladders-rest-2026-10-17/d/`) leaked nothing. One observation, no mechanism.

Practical consequence for this board: check `MemAvailable` before a measurement, and if ollama has loaded a
model since the last boot, reboot first. The campaign scripts here now log it at the start of every run.
