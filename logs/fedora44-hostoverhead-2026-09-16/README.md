# Where Fedora 44's lower host CPU time comes from, 2026-09-16

The one measured difference between the two systems that survived the GPU clock correction was host CPU
time: the same decode run costs far less of it on Fedora 44. This is the cause.

Same board, same kernel 7.1.8 with the bc250 amdgpu module, same llama.cpp source, GPU clock pinned at
1500 MHz on both, `ollama` stopped, qwen2.5-1.5B Q4_K_M, `llama-bench -mmp 0 -ngl 99 -fa on -p 0 -n 256
-r 1`, `/usr/bin/time` around the whole invocation (`f4*_time_*.txt`, second run of each shown):

| | Fedora 43, ROCm 6.4.2 | Fedora 44, ROCm 7.1.1 |
|---|---|---|
| wall | 7.59 s | 3.12 s |
| user | 5.10 s | 2.75 s |
| system | 4.41 s | 0.40 s |
| decode rate | 119.15 t/s | 117.39 t/s |

The rate is the same; only the CPU cost differs.

## It is the event-wait ioctl

`strace -f -c` over the same invocation (`f4*_strace.txt`):

| | Fedora 43 | Fedora 44 |
|---|---|---|
| ioctl calls | 127743 | 4379 |
| time in ioctl | 4.31 s | 0.84 s |
| read calls | 5766 | 5007 |

Tracing the calls themselves over a shorter run, `-n 32` (`f4*_ioctl.txt.gz`):

| ioctl | Fedora 43 | Fedora 44 |
|---|---|---|
| `AMDKFD_IOC_WAIT_EVENTS` | 11387 | 1037 |
| `AMDKFD_IOC_SET_EVENT` | 616 | 152 |
| `AMDKFD_IOC_CREATE_EVENT` / `DESTROY_EVENT` | 143 each | 82 each |
| memory map/unmap/alloc/free | 76 to 79 each | 73 each |

So ROCm 6.4.2's runtime enters the kernel about eleven times as often waiting for GPU completion signals,
for the same work at the same rate. The memory-management calls are the same in both. Three runtime knobs
were tried on Fedora 43 and none of them changes it: `ROC_ACTIVE_WAIT_TIMEOUT=0` and `=100000` leave the
figures untouched, and `HSA_ENABLE_INTERRUPT=0` moves the cost from system to user time (7.58 s user, 2.90 s
system) without reducing the total.

This matters for a board whose CPU is also feeding the GPU: it is not a throughput win here, but it leaves
more CPU for everything else.

## A newer ROCr does not fix the deep-context crash (`rocr72_*`)

Fedora 45's `rocm-runtime-7.2.1-4.fc45` extracted beside the system one and selected with
`LD_LIBRARY_PATH` loads and runs on Fedora 44 (`ldd` confirms
`libhsa-runtime64.so.1 => ~/rocr72/usr/lib64/...`; the two files differ, both are soname 1.18.0).
Throughput is unchanged: pp512 784.7, 793.3, 793.6 against the system library's 784.3, 793.0, 793.0, and
tg64 117.6, 117.8, 116.9 against 116.6, 117.1, 116.6.

It does **not** fix the crash at a context depth past the KFD memory limit
([`../fedora44-ceilings-2026-09-16/`](../fedora44-ceilings-2026-09-16/)): qwen3-14B at depth 16384 still
dies with `segfault at 20 ... in libhsa-runtime64.so`, the same fault address as the system library. So
the upstream scratch-allocation fix of February 2026 either is not in this build or is not the path being
taken here. Whatever the fix for that crash is, it is not "use a newer ROCm runtime".
