# ROCm / HIP on the AMD BC-250 (gfx1013, Cyan Skillfish)

This document collects what an investigation into AMD ROCm/HIP compute on the BC-250 mining
board turned up: what works, what does not, why, and the workarounds that were tried. The
board is built around AMD's Cyan Skillfish APU (Zen 2 CPU plus a gfx1013 RDNA1 GPU, 16 GB
GDDR6 unified memory), a design related to the PlayStation 5 silicon and sold cheaply as
salvaged mining hardware. gfx1013 is not on ROCm's supported-GPU list, so nothing works out
of the box.

These are field notes from a single board with a community-patched BIOS. Some conclusions are
best-effort inferences from observed behaviour and from upstream reports rather than confirmed
silicon errata, and that is noted where it applies.

Environment used throughout: Fedora 43, kernel 6.18.9-200.fc43 (also cross-checked on stock
6.17.1), ROCm 6.4.2 with rocBLAS 6.4.4, LLVM 19, Mesa RADV for the Vulkan comparison, the
community 40-CU unlock applied, and the oberon governor capping the GPU at 1500 MHz.

## TL;DR

- The gfx1013 compute-only queue is defective. Mesa/RADV disables it and routes compute through
  the graphics queue; its driver comment reads `GFX1013 is known to have broken compute queue`.
  A minimal test here reproduces the defect directly: small compute kernels are fine, large ones
  intermittently hang or return wrong results, and the queue degrades under sustained load.
- ROCm/HIP runs all compute on that compute queue via KFD and cannot avoid it, so on real
  (large-kernel) workloads it is slow, non-deterministic, hangs, and degrades the board. The
  process-exit hang can be mitigated (userspace `sched_policy=2`, or a kernel-side change tracked
  in ROCm/ROCm#6313), but no tweak tested here makes large-kernel compute reliable or correct,
  because a driver cannot repair a broken compute queue.
- Compute works through the graphics queue: Vulkan (llama.cpp) runs full inference reliably,
  and RustiCL runs an OpenCL kernel correctly on the same board.
- On this board, Vulkan is the practical choice for inference and RustiCL for OpenCL, and
  ROCm/HIP is probably best treated as unsupported for now.

## Summary

The short version: on this board the Vulkan (RADV) backend seems to be the better path. It works
well, needs no workarounds, and runs on the part of the GPU that is actually functional. ROCm/HIP
can be coaxed into running, but it tends to be slow, non-deterministic, and unstable, because it
depends on a part of the GPU that appears to be hardware-broken on this APU.

The root cause is not specific to ROCm. The compute-only queue (the MEC, or MicroEngine
Compute, pipes) on gfx1013 has a hardware defect. Mesa/RADV already ships a workaround for it:
[merge request !33116](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/33116),
included in Mesa 25.1 and enabled by default for gfx1013, detects the chip and disables the
compute-only queue, routing all work through the graphics/universal queue. The merge request
description states it plainly: "Compute-only queue doesn't work properly, leading to massive
Vulkan CTS tests failures and visual glitches in games. This MR disables it for now, until the
root cause is known." The disabling code that commit `7271b8ee` added to
`src/amd/common/ac_gpu_info.c` is direct (later refactored in Mesa main, but unchanged in intent):

```c
/* GFX1013 is known to have broken compute queue */
if (device_info.family == FAMILY_NV && ASICREV_IS(device_info.external_rev, GFX1013)) {
   info->ip[AMD_IP_COMPUTE].num_queues = 0;
}
```

Setting the number of compute queues to zero forces all work onto the graphics/universal
queue. HIP/ROCm dispatches all compute to the MEC compute queues through KFD, and unlike RADV,
RustiCL, or Vulkan, it has no way to avoid them. That is why Vulkan and OpenCL-via-RustiCL
work while ROCm does not. Even RADV's author does not know the exact silicon mechanism (hence
"until the root cause is known"), so "hardware defect in the compute queue" is the accurate
framing, but the precise cause is not published.

## Backends and the compute queue

A quick model of what runs where explains the whole situation:

| Path | Queue used | Status on BC-250 |
|------|-----------|------------------|
| Vulkan (RADV) | graphics / universal | works, fast, reliable |
| OpenCL (RustiCL) | graphics / universal | runs a compute kernel correctly |
| ROCm / HIP | MEC compute (via KFD) | broken: slow, non-deterministic, hangs |

An OpenCL vector-add of about a million elements ([`patches/ocl_vecadd.c`](patches/ocl_vecadd.c))
run under RustiCL (`RUSTICL_ENABLE=radeonsi`) computes correctly with zero mismatches on the
`AMD BC-250 (radeonsi, gfx1013)` device. That confirms the silicon can do compute when the work
goes through the graphics queue. The problem is confined to the dedicated compute queue that
ROCm requires. This was only a minimal kernel, though. How well RustiCL holds up on larger or
real-world OpenCL workloads was not tested here, so it is better read as a working proof of
concept than as a verified general OpenCL path (TBD).

A bare HIP reproducer ([`patches/compute_probe.c`](patches/compute_probe.c), native gfx1013, no
rocBLAS, no override) narrows what "broken" means. It dispatches a plain arithmetic kernel of a
chosen size and checks every element against a CPU reference. Findings, on freshly reset boards
(a fresh board is needed per measurement because of the degradation described below):

- Kernels up to about two million threads (block size 256) are correct, fast, and deterministic,
  and sync time scales cleanly with the work done.
- From about four million threads upward the dispatch becomes unreliable. On repeated runs of the
  same size it will sometimes complete correctly, sometimes hang (a single dispatch never returns,
  and is killed by a timeout while the board survives if `amdgpu.sched_policy=2` is set), and
  sometimes complete with wrong results. Wrong elements keep their pre-kernel value, meaning those
  wavefronts' outputs were lost. The magnitude varies widely between runs: one sixteen-million-thread
  run had 896 of 16,777,216 elements wrong, an eight-million-thread run had 1,764,388 of 8,388,608
  wrong (about twenty-one percent).
- The same size gives different outcomes on repeat, so this is a reliability defect, not a fixed
  cutoff.
- It is cumulative. After enough large-dispatch stress in one boot, all compute hangs, including
  small kernels that were fine minutes earlier, until a hard reset. GPU temperature stays around
  64 C, so it is not thermal.

So the precise statement is that large or complex compute dispatches on the gfx1013 compute queue
are unreliable, both for correctness and for liveness, and the queue degrades under load. That
matches what Mesa observed (simple apps work, CTS tests and games fail) and explains the ROCm/HIP
behaviour directly, since LLM inference is entirely large GEMM and attention kernels.

Because of this, a driver-side fix for the compute-queue defect itself is not expected to be
possible. A driver cannot repair broken silicon. RADV's fix is avoidance, and ROCm has no
equivalent option. The separate process-exit hang described below is mitigable, and is being
worked on kernel-side upstream, but making the hang go away does not make the compute queue's
results reliable.

## Getting HIP to run at all

Even though it is not usable for real work, getting HIP far enough to demonstrate the above
took three independent workarounds. They are documented here because each corresponds to a
separate real issue.

### 1. Board freeze when a HIP process exits

Any HIP process, including a trivial one that only creates and destroys a stream, could hang
the whole board when it exited (clean exit, abort, or SIGKILL). dmesg showed a cascade of
`timeout waiting for kiq fence`, `TLB flush failed for PASID`, `Failed to quiesce KFD`,
followed by a freeze.

Setting the KFD scheduler away from the hardware scheduler mitigates this:

```bash
sudo grubby --update-kernel=ALL --args="amdgpu.sched_policy=2"
sudo reboot
```

With this set, HIP processes exit cleanly, including under SIGKILL, and the board stays up.
This was re-checked across a fresh reboot (two clean exit cycles and a SIGKILL, no KFD errors),
and with the parameter absent a process abort hung the board again. It is a mitigation, not a
complete fix: a process that aborts while the GPU is already in a fault state can still wedge
the board. Underlying issue: [ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313).

The same hang is being addressed kernel-side in that thread. Contributors there report that
setting `adev->gmc.flush_pasid_uses_kiq = false` in `gmc_v10_0.c` (routing PASID TLB flushes
through MMIO instead of the KIQ ring) removes the `kiq fence` timeouts and the full-system hang,
at the cost of building a patched amdgpu module. That is a more thorough fix for the exit hang
than the userspace `sched_policy=2` mitigation used here, though reports in the thread note the
board is still not fully stable under all compute workloads. Neither change is aimed at the
compute-queue correctness defect above: both concern the TLB-flush and process-exit path rather
than compute execution. The kernel patch was not tested here against the wrong-results behaviour,
but Mesa's evidence suggests the two are separate, since its compute-queue failures appear on the
pure-Vulkan path with no KFD TLB flush involved.

### 2. rocBLAS has no gfx1013 code objects

GEMM, and therefore inference, aborts immediately with:

```
Cannot find CO in the bundle /usr/lib64/librocblas.so.4.4 for ISA: amdgcn-amd-amdhsa--gfx1013:xnack-
```

`strings /usr/lib64/librocblas.so.4.4 | grep -oE 'gfx10[0-9]+'` shows the embedded code
objects cover gfx1010/1011/1012 and gfx1030 through 1035, but not gfx1013. On rocBLAS 6.4.4
the kernels are code objects embedded in `librocblas.so`, matched by exact ISA, so the older
workaround of symlinking the external `library/*gfx1010*` Tensile files to gfx1013 names no
longer covers this path.

gfx1013 is the same GFX10.1 instruction set as gfx1010, so the fix is to make the whole stack
report and target gfx1010:

```bash
# build llama.cpp for gfx1010
cmake -B build-g1010 -S . -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1010 -DGPU_TARGETS=gfx1010 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/usr -DHIP_PATH=/usr -G Ninja
ninja -C build-g1010 llama-bench llama-cli

# run with the ISA override so runtime, rocBLAS, and llama.cpp all agree on gfx1010
export HSA_ENABLE_SDMA=0
export HSA_OVERRIDE_GFX_VERSION=10.1.0
export LD_LIBRARY_PATH=$PWD/build-g1010/bin
./build-g1010/bin/llama-cli --no-mmap --gpu-layers 999 -fa off -m model.gguf -p "..." -n 16
```

A standalone rocBLAS SGEMM ([`patches/rocblas_probe.c`](patches/rocblas_probe.c)) confirmed
the mechanism: native gfx1013 returns `rocblas_status_internal_error`; with
`HSA_OVERRIDE_GFX_VERSION=10.1.0` it returns a correct result. The override needs to be paired
with a gfx1010 build; using it with a gfx1013-compiled binary produces
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` from the ISA-tag mismatch.

The clean fix belongs in rocBLAS, which should ship gfx1013 code objects since it is the same
ISA as gfx1010. That is tracked in
[ROCm/rocm-libraries PR #8838](https://github.com/ROCm/rocm-libraries/pull/8838).

### 3. Flash attention faults

With `-fa on`, compute aborts with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` as soon as it
starts, most likely in the flash-attention kernel since that is what the flag changes. This was
hidden at first because full-offload runs did not always reach compute. Running with `-fa off`
avoids the fault, so HIP inference on this board additionally requires `-fa off`. Whether this
is a distinct bug or another face of the compute-queue defect was not determined.

## What the slowness actually is

Early on the symptom looked like pathologically slow model loading (tens of minutes for a few
GB). Instrumenting the ggml loader to time each phase of its asynchronous upload loop showed
that this framing was wrong.

The upload loop is fast when the GPU-to-CPU completion signalling cooperates:

```
LOADTIMING: n_sync=1206 sync=1ms read=183ms upload=12ms
```

The entire weight upload (1206 chunk syncs, file reads, and async submits) completes in about
0.2 seconds, yet the run still took many minutes and produced no throughput numbers. The time
is not spent loading. It is spent in compute and in per-operation completion.

The real problem is that HIP's GPU-to-CPU completion signalling on this board is unreliable.
Each `hipEventQuery` or `hipStreamSynchronize` can return in microseconds or take several
seconds, non-deterministically, for the same work. Two identical minimal runs, both on a
freshly reset board, produced opposite outcomes: one finished a prompt-processing pass in about
15 seconds, the next timed out at 200 seconds. Because every inference step depends on these
syncs, and because a larger workload issues more of them, prompt processing occasionally
squeaks through while token generation never completed. The board also degrades under sustained
HIP use: after enough activity even trivial allocations become slow, and eventually systemd and
dbus stop responding and a hard power-cycle is required. GPU temperature stayed normal (around
64 C), so this is not thermal throttling.

All of this is consistent with the compute-queue hardware defect. It is the same class of
failure as the freezes in [ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313) and the
`Fence fallback timer expired on ring sdma0` messages that appear at every boot, tracked more
generally in [kernel bug #216645](https://bugzilla.kernel.org/show_bug.cgi?id=216645).

## Things that did not help

These were tried against the slowness and instability, and are listed so others do not repeat
them:

- `amdgpu.vm_update_mode=3` (CPU-based page-table updates): caused `APERTURE_VIOLATION` in
  compute.
- `amdgpu.msi=0` (force legacy INTx): delivered no amdgpu interrupts at all during GPU activity,
  strictly worse.
- A kernel-module patch reducing the fence fallback timer from 500 ms to 2 ms
  ([`patches/amdgpu-fence-fallback-2ms.patch`](patches/amdgpu-fence-fallback-2ms.patch)):
  moved the stall out of the kernel in stack samples but did not change total run time.
- `HSA_ENABLE_INTERRUPT=0` (memory-polled signals): no improvement.
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (managed memory, in-place weights on UMA): no improvement.
- `amdgpu.gartsize`: hardware-fixed at 512 MB on this ASIC, no effect.
- `amdgpu.num_kcq=1` with `amdgpu.compute_multipipe=0` (restrict compute to a single queue and
  pipe): still hangs, so it is not one bad pipe, the whole compute queue is broken.
- `amdgpu.cwsr_enable=0` (disable compute wave save/restore preemption) plus `amd_iommu=on
  iommu=pt`: still hangs. These parameters help on a functioning compute queue but cannot
  revive a broken one.
- Stock 24 CU instead of the 40-CU unlock: just as flaky, so the unlock is not the cause.
- Larger or smaller staging chunks, and reduced `--gpu-layers`: no effect, because the cost is
  in per-operation completion, not in the upload.

An isolated microbenchmark of the upload pattern (`hipMalloc` plus copy plus event,
[`patches/loadmimic.c`](patches/loadmimic.c)) is usually fast and does not reproduce the full
llama slowness, which reinforces that the problem lives in the full compute path plus the
board's degrading state rather than in any single primitive.

The fence-fallback patch is kept in this repository for the record even though it is
ineffective, so the attempt is documented rather than silently dropped.

## Performance

Model: qwen2.5-1.5B Q4_K_M. Board at 24 CU, `-fa off`, so that the ROCm and Vulkan arms match.

| backend | pp128 (t/s) | tg16 (t/s) | reliable |
|---------|:-----------:|:----------:|----------|
| Vulkan (RADV) | 1000.6 | 156.8 | yes, every run |
| ROCm / HIP | about 670 (a single run) | not measurable | no, usually hangs or segfaults |

Even in the one run where HIP produced a number, Vulkan was roughly 1.5 times faster on prompt
processing, and HIP never produced a token-generation figure. That single pp128 result did not
reproduce on subsequent attempts.

For the general picture, here are reliable Vulkan numbers (llama.cpp, `-fa on`, `-p 128 -n 128`,
median of 3), at stock 24 CU and at the 40-CU unlock, tokens per second:

| model | params | pp128, 24 -> 40 CU | tg128, 24 -> 40 CU |
|-------|:------:|:------------------:|:------------------:|
| qwen2.5-1.5B Q4_K_M | 1.8 B | 1011 -> 1276 | 158 -> 207 |
| granite-4.0-h-tiny Q4_K_M (hybrid) | 6.9 B | 398 -> 576 | 107 -> 133 |
| gpt-oss-20b MXFP4 (MoE) | 20.9 B | 197 -> 309 | 77 -> 105 |
| moe-coder-30b IQ2_M (MoE) | 30.5 B | 144 -> 230 | 69 -> 96 |
| qwen3.5-35b-a3b IQ2_M (MoE) | 34.7 B | 157 -> 244 | 65 -> 87 |
| qwen3.6-35b-a3b IQ2_M (MoE) | 34.7 B | 158 -> 246 | 65 -> 87 |
| deepseek-r1-14b Q4_K_M (dense) | 14.8 B | 114 -> 178 | 23 -> 34 |
| mistral-small-3.2 Q3_K_M (dense) | 23.6 B | 66 -> 99 | 12 -> 19 |
| qwq-32b IQ2_M (dense) | 32.8 B | 51 -> 81 | 10 -> 16 |

The 40-CU unlock scales prefill by roughly +45 to +60 percent for the larger models (only about
+26 percent for the smallest, qwen2.5-1.5B, which is less compute-bound), and generation by
roughly +25 to +60 percent. The sparse MoE models decode far faster than their total parameter
count would suggest, close to their active parameter count, which is why the 20 to 35 billion
parameter MoEs generate at 87 to 105 tokens per second while the dense 32 billion qwq manages
16. More models, long-context behaviour, and full methodology are in the reference repository
below.

## Reproducing

Exact versions used for the results here:

- kernel 6.18.9-200.fc43.x86_64 (also cross-checked on 6.17.1-300.fc43)
- rocm-hip 6.4.2-2.fc43, rocblas 6.4.4-1.fc43, ROCm LLVM/clang 19
- mesa-vulkan-drivers 25.3.4-7.fc43 (RADV, RustiCL)
- llama.cpp build 2da6686
- GPU memory as reported by the driver: 512 MB VRAM carveout, 16 GB GTT

[`reproduce.sh`](reproduce.sh) builds and runs the probes on the board: it prints the rocBLAS
embedded ISAs (no gfx1013), runs the rocBLAS SGEMM native and with the override, runs the
OpenCL vector-add on RustiCL, and runs the HIP compute probe at a small and a large size. Boot
with `amdgpu.sched_policy=2` first: the HIP probes can hang the machine on process exit at the
default scheduler (Problem 1 above), and the script refuses to run without it unless `FORCE=1`
is set. The large-kernel result is non-deterministic, so a few repeats are expected.

## Files

| Path | What it is |
|------|------------|
| [`reproduce.sh`](reproduce.sh) | Builds and runs the probes below and prints the key results |
| [`patches/compute_probe.c`](patches/compute_probe.c) | Bare HIP compute reproducer: small kernels correct and fast, large kernels intermittently wrong and degrade the queue |
| [`patches/ocl_vecadd.c`](patches/ocl_vecadd.c) | Minimal OpenCL vector-add that proves compute works on the graphics queue via RustiCL |
| [`patches/rocblas_probe.c`](patches/rocblas_probe.c) | Standalone rocBLAS SGEMM that tests gfx1013 kernel availability and returns on error instead of aborting |
| [`patches/loadmimic.c`](patches/loadmimic.c) | Microbenchmark that replicates llama.cpp's asynchronous upload loop, used to isolate the slowness |
| [`patches/evtlat.c`](patches/evtlat.c) | Small HIP event-synchronisation latency microbenchmark |
| [`patches/amdgpu-fence-fallback-2ms.patch`](patches/amdgpu-fence-fallback-2ms.patch) | Kernel-module patch reducing the fence fallback timer, kept for the record; ineffective |

## References

- [ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313): BC-250 system freeze after compute workloads (the ROCm-side tracking issue).
- [Mesa merge request !33116](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/33116) and [Mesa 25.1 release notes](https://docs.mesa3d.org/relnotes/25.1.0.html): RADV disables the broken gfx1013 compute-only queue. Commit `7271b8ee`, "radv,radeonsi: disable compute queue for BC250".
- [Mesa issue #11982](https://gitlab.freedesktop.org/mesa/mesa/-/issues/11982): AMD Cyan Skillfish support discussion, including that RustiCL works and the compute-queue artifacting.
- [drm/amd issue #3356](https://gitlab.freedesktop.org/drm/amd/-/issues/3356): a kernel-side Cyan Skillfish system-freeze regression (now closed).
- [AMD BC250 documentation, RADV page](https://elektricm.github.io/amd-bc250-docs/drivers/radv/): the compute-queue issue and the `RADV_DEBUG=nocompute` workaround.
- [kernel bug #216645](https://bugzilla.kernel.org/show_bug.cgi?id=216645): Fence fallback timer expired.
- [ROCm/rocm-libraries PR #8838](https://github.com/ROCm/rocm-libraries/pull/8838): adding gfx1013 support to rocBLAS.
- [duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock): the 40-CU unlock and the module-build pipeline reused here.
- [github.com/akandr/bc250](https://github.com/akandr/bc250): the production Vulkan stack, full benchmark methodology, and the paper.

## Conclusion

ROCm/HIP on the BC-250 leans on the compute queue, which seems to be the part of this APU that
does not work. The slow and non-deterministic completion, the wrong results on large dispatches,
and the degradation under load all look downstream of that defect, and a driver-level fix for it
is probably not possible, for the same reason RADV could only work around it by avoidance. The
separate board freeze on process exit is more tractable: it can be mitigated with the userspace
`amdgpu.sched_policy=2`, and a kernel-side change for it is being worked on in ROCm/ROCm#6313,
though stopping the freeze does not make the compute queue's results reliable. Vulkan runs on the
functional graphics queue, tends to be faster, and looks like the better backend to reach for
here. RustiCL ran a simple OpenCL kernel correctly on the same graphics queue and may be worth a
look if OpenCL compute is needed, though its suitability for larger workloads was not verified in
this investigation.

## Author and license

Author: Artur Andrzejczak. Prepared with assistance from Claude Opus.

Code: [AGPL-3.0](LICENSE) · Docs: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

The code in this repository (the probes and patches under `patches/`, and `reproduce.sh`) is
licensed under the GNU Affero General Public License v3.0. The documentation (this README) is
licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
