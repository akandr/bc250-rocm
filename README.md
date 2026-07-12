# ROCm / HIP on the AMD BC-250 (gfx1013, Cyan Skillfish): field notes

These are notes from testing GPU compute on an AMD BC-250, from one board and one software stack.
The measurements are reproducible and included as logs; the explanations are working theories and
may be wrong or incomplete. Corrections and "have you tried X" comments are welcome; see
[Open questions](#open-questions) at the end.

This is the ROCm/HIP companion to [akandr/bc250](https://github.com/akandr/bc250), which covers the
board itself and its (working) Vulkan setup; that background is not repeated here. The Vulkan side
is the settled one: for running LLM inference on the board today, Vulkan via Mesa RADV is the
reliable, fast path. What follows is the other side, how far the ROCm/HIP compute stack can be
pushed, and where it keeps hitting a wall.

Environment throughout: Fedora 43, kernel 6.18.9-200.fc43, ROCm 6.4.2 (rocBLAS 6.4.4 as shipped,
plus a native gfx1013 rocBLAS built locally), LLVM/clang 19, Mesa 25.3 RADV for the Vulkan
comparison, the community 40-CU unlock, the oberon governor around 1500 MHz.

## Contents

- [A short primer: the AMD compute stack](#a-short-primer-the-amd-compute-stack)
- [The claim this repo tests](#the-claim-this-repo-tests)
- [Observation 1: occasional silent wrong results](#observation-1-occasional-silent-wrong-results)
- [Observation 2: the compute queue wedges under load](#observation-2-the-compute-queue-wedges-under-load)
- [Building a native gfx1013 rocBLAS](#building-a-native-gfx1013-rocblas)
- [Observation 3: the unlock, the fix, and the wedge are entangled](#observation-3-the-unlock-the-fix-and-the-wedge-are-entangled)
- [How far ROCm inference gets](#how-far-rocm-inference-gets)
- [ROCm vs Vulkan](#rocm-vs-vulkan)
- [Status snapshot](#status-snapshot)
- [Open questions](#open-questions)
- [Reproducing](#reproducing)
- [Files](#files)
- [References](#references)

## A short primer: the AMD compute stack

This section lays out the vocabulary the rest of the document uses, working from an application down
to the silicon. It is a simplified picture.

### The one-picture version

A program like llama.cpp can reach the same GPU by two completely separate software roads. One is
built for graphics (and works on this board), the other for general-purpose compute (and is the one
that struggles):

```
                    llama.cpp  (the application)
                   /                            \
        COMPUTE road (ROCm)              GRAPHICS road (Vulkan)
   HIP        a CUDA-like API          Vulkan      graphics + compute API
   rocBLAS    math libraries           (shaders)   the GPU programs
   ROCr/HSA   userspace runtime        Mesa RADV   userspace driver
   KFD        in the amdgpu driver     amdgpu DRM  kernel driver
        |                                    |
   MEC compute queue                  graphics (universal) queue
         \                                  /
                 one shared set of GPU shader cores
```

Everything below just names the boxes in that diagram.

### Architectures, cores, and the word "kernel"

**GPU architectures have ISA names.** AMD GPUs carry an instruction-set name such as `gfx900`,
`gfx1030`, or `gfx1100`, and GPU programs are compiled for a specific one. This board's GPU is
**gfx1013** (RDNA1-class), which is not on ROCm's official supported-GPU list, so "is gfx1013
supported?" recurs throughout.

**Shader cores, CUs, and wavefronts.** A GPU runs work on many small parallel cores grouped into
**compute units (CUs)**; vendors often disable some at the factory ("harvesting"), and threads
execute in lockstep groups called **wavefronts**. How many CUs end up enabled turns out to matter
later.

**"Kernel" means two things.** A **GPU kernel** is a
small program that runs on the GPU (one launch of it is a **dispatch**). The **Linux kernel** is
the operating system, and the `amdgpu` **kernel driver** lives inside it. "A compute kernel wedges"
means a GPU program; "kernel 6.18" means Linux.

### The two software roads

**Graphics (works here).** OpenGL and Vulkan are served on Linux mostly by **Mesa**; AMD's Vulkan
driver in Mesa is **RADV**. llama.cpp's Vulkan backend uses this road, and it runs well on the
BC-250.

**Compute (the hard one).** AMD's general-purpose compute stack is **ROCm**. Its CUDA-like
programming API is **HIP** (close enough to CUDA that code often ports with a rename), and on top
sit math libraries such as **rocBLAS**. Underneath HIP is the **ROCr / HSA runtime**, the userspace
layer that talks to the driver and hands work to the GPU; environment variables like
`HSA_OVERRIDE_GFX_VERSION` and errors like `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` come from
here. llama.cpp's HIP backend uses this road, and it is the one that struggles on this chip.

**KFD.** The kernel-side half of ROCm is the **KFD** (Kernel Fusion Driver), part of the `amdgpu`
module. It sets up the compute queues, doorbells, and per-process GPU memory maps that HIP programs
use. "The compute queue is broken" points at something in this path.

### How work actually reaches the GPU: queues

The driver hands work to the GPU through hardware **queues** (command rings the GPU pulls from, like
a to-do list). Two matter here:

- the **graphics / universal queue**, driven by the GFX engine, and
- the **compute queue(s)**, driven by the **MEC** (MicroEngine Compute), a small firmware processor
  on the GPU dedicated to compute dispatches.

The single most important fact for this whole document: **ROCm/HIP sends its compute to the MEC
compute queue**, while Vulkan/RADV normally uses the graphics queue. Same shader cores at the
bottom, different route to reach them. (An OpenCL path called **RustiCL**, part of Mesa, also goes
by the graphics-queue route, which makes it a handy control later.)

**KIQ, PASID, and TLB flushes.** The **KIQ** (Kernel Interface Queue) is a special ring the driver
uses to ask the MEC firmware to do privileged jobs. One such job is invalidating the GPU's
address-translation cache (a **TLB flush**, the GPU equivalent of a CPU's TLB) for a given process,
which is identified by a **PASID**. Newer kernels route that PASID TLB flush through the KIQ/MEC
firmware; older kernels did it directly from the CPU over memory-mapped registers (**MMIO**). That
choice is the crux of Observation 1.

**rocBLAS, Tensile, and code objects.** rocBLAS is AMD's matrix-multiply (BLAS) library; **Tensile**
is the part that generates its GPU programs per architecture. Those compiled GPU programs are
**code objects** (files ending `.hsaco`, an ELF holding GPU machine code). Stock rocBLAS ships no
gfx1013 code objects, so matrix operations have nothing to run and fall over. Building them is one
of the sections below.

### How Mesa handles it

Mesa's source, for this chip, carries the comment `GFX1013 is known to have broken compute queue`,
and RADV
[disables the compute-only queue for it](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/33116),
routing compute through the graphics queue instead. ROCm has no equivalent escape hatch: its
compute goes to the compute queue, so it cannot side-step the problem the same way.

## The claim this repo tests

Getting ROCm working would open up the wider GPGPU ecosystem on the board (rocBLAS, PyTorch-shaped
workloads, image generation, and so on). The stock answer is that it cannot: the compute queue is
broken. The notes below test that claim. A tentative reading of the results is that the single label
"broken compute queue" appears to cover **two different problems** that behave very differently.
That split is an interpretation of the observed behaviour, not a proven account.

## Observation 1: occasional silent wrong results

### What the probe showed

A bare HIP kernel ([`patches/compute_probe.c`](patches/compute_probe.c), native gfx1013, no
rocBLAS, no override) fills an array by arithmetic and checks every element against a CPU
reference. On the stock driver it sometimes returns wrong answers with no error reported. In one
run at about 8.4 million threads, 525,308 elements were wrong, and each wrong element still held its
pre-kernel value, as if some stores were dropped. Silent wrong results are the most dangerous
failure mode, so this was worth chasing.

| module | 1M | 4M | 8M | 16M | silent-wrong runs | kiq-fence freeze |
|--------|:--:|:--:|:--:|:---:|:-----------------:|:----------------:|
| stock (`flush_pasid_uses_kiq = true`) | ok | ok / abort | 525,308 wrong / hang | hang | yes | yes |
| patched (`= false`) | ok | ok | ok (mostly) | 16.7M correct once; some recoverable hangs | 0 | no |

Full logs: [`logs/stock/`](logs/stock/), [`logs/patched/`](logs/patched/).

### A likely explanation (tentative)

The dropped-store pattern points at address translation. The PASID TLB flush
(`amdgpu_gmc_flush_gpu_tlb_pasid()`) branches on `adev->gmc.flush_pasid_uses_kiq`. When true (the
mainline default), the flush is submitted to the KIQ ring, so the MEC firmware performs it while
compute is in flight, and this path is also the source of a `timeout waiting for kiq fence`
board-freeze. When false, the flush is done from the CPU over MMIO, with no MEC involvement.

A plausible reading, and it is only that, is that routing the invalidation through the MEC while a
compute kernel is running lets a translation get invalidated out from under an in-flight wavefront
on this Navi-1x part, so the store lands nowhere. Setting that field to false appears to make the
wrong answers go away in these tests:

```c
/* BC-250/gfx1013: route PASID TLB flushes via MMIO, not the KIQ ring */
adev->gmc.flush_pasid_uses_kiq = false;
```

The change is [`patches/amdgpu-flush-pasid-mmio.patch`](patches/amdgpu-flush-pasid-mmio.patch),
built with [`scripts/build_patched_amdgpu.sh`](scripts/build_patched_amdgpu.sh). The same change
also stops the `kiq fence` board-freeze and greatly speeds up model loading.

Credit for the `flush_pasid_uses_kiq = false` idea belongs to **anrp** and **ahorek** in
[ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313), who found that it stops the freeze. What
these tests seem to add is that it also removes the silent wrong results in an A/B comparison. That
is an n=1 hardware result, so independent reproduction or refutation would be valuable.

This comes with an important caveat, though. The patched runs above were captured on a boot where
the patched module happened to come up at 40 CU. More often the same patch seems to leave the board
at 24 CU, where even a trivial compute dispatch wedges, so getting the correct flush and a working
compute queue at the same time was not something these tests could do reliably. Why that might
happen is [Observation 3](#observation-3-the-unlock-the-fix-and-the-wedge-are-entangled); it is part
of why "fixed" would overstate this.

### A useful contrast: the graphics queue runs the same compute cleanly

The clearest single test runs the identical kernel on the graphics queue instead of the compute
queue, by porting it to OpenCL ([`patches/ocl_compute_probe.c`](patches/ocl_compute_probe.c)) and
running under **RustiCL** (`RUSTICL_ENABLE=radeonsi`), which dispatches through the
graphics/universal queue as RADV does:

| size (threads) | HIP (MEC compute queue) | RustiCL (graphics queue) |
|----------------|-------------------------|--------------------------|
| 1,048,576 | ok | 0 wrong, 2.0 ms |
| 4,194,304 | ok | 0 wrong, 46 ms |
| 8,388,608 | 525,308 wrong / wedge | 0 wrong, 92 ms |
| 16,777,216 | wrong / hang | 0 wrong, 184 ms |

The graphics queue was correct and fast at every size, including a sustained
many-small-dispatch pattern (1M threads times 200 sequential launches), with no wedges
([`logs/rusticl_graphics_queue_ok.log`](logs/rusticl_graphics_queue_ok.log),
[`logs/rusticl_sustained_ok.log`](logs/rusticl_sustained_ok.log)). One reading is that the shader
hardware, memory, and ALUs are fine, and the fault lives specifically in the MEC compute-queue
path, which would also explain why Mesa's route-through-graphics fix works and why ROCm, unable to
do that, is stuck. That is offered as the most consistent interpretation, not a proof.

## Observation 2: the compute queue wedges under load

Separately from the wrong results, a large or sustained stream of compute dispatches intermittently
wedges the queue. This appears even in the 40-CU configuration where smaller dispatches are correct,
and the measurements below are from that configuration. The teardown in dmesg reads:

```
amdgpu: cp queue preemption time out.
amdgpu: Pasid 0x8004 destroy queue 1 failed, ret -62      (-62 = ETIME)
amdgpu: Resetting wave fronts (nocpsch) on dev ...
```

A lost completion interrupt was one theory, plausible on a board that prints
`Fence fallback timer expired` every boot. But memory-polled completion (`HSA_ENABLE_INTERRUPT=0`)
hangs the same way, and the message is specifically a preemption timeout: the driver asks the MEC
to preempt a queue and the kernel never yields. The current reading is therefore that a compute
kernel gets stuck mid-flight and cannot be preempted, rather than a signal going missing. This
resembles the "compute-only queue doesn't work properly" that Mesa documented and chose to route
around.

No driver knob removed it in these tests. Tried without effect: `amdgpu.sched_policy` 0 and 2, CWSR
on and off, `HSA_ENABLE_INTERRUPT` 0 and 1, HIP graphs on and off, flash-attention on and off, 24
versus 40 CU, and native-versus-override builds. That is a list of things that did not help, not a
claim that nothing can.

### Seeing it with a real library GEMM

To watch this without llama.cpp in the way, a native gfx1013 rocBLAS was built (next section) and a
tiny CPU-checked SGEMM run at various sizes ([`patches/sgemm_sweep.cpp`](patches/sgemm_sweep.cpp)):

| GEMM | stock module, 40 CU | note |
|------|---------------------|------|
| N=256 / 512 / 1024 / 2048 (single) | correct | up to about 1660 GFLOP/s at N=2048 |
| N=512 times 2000 (sustained) | correct, about 2327 GFLOP/s | thousands of small GEMMs are fine |
| N=1024 times 500 (sustained) | correct, about 3746 GFLOP/s | |
| N=2048 times 200 (sustained) | wedge (timeout) | |
| N=4096 (single) | wedge (timeout) | |

Full log: [`logs/rocblas/sgemm_sweep_stock_40cu.log`](logs/rocblas/sgemm_sweep_stock_40cu.log). Two
points stood out. Every GEMM that completed was numerically exact; with rocBLAS's structured
kernels these tests saw no silent-wrong results at these sizes, only the wedge. And the wedge
looked intermittent rather than a clean size threshold: N=1024 hung once and then ran fine on
retry. That intermittency is why "just keep dispatches small" seems unlikely to be made reliable.

## Building a native gfx1013 rocBLAS

A long-standing workaround for the missing gfx1013 matrix kernels is to build for **gfx1010** and
run with `HSA_OVERRIDE_GFX_VERSION=10.1.0`, since gfx1010 and gfx1013 share an ISA. In these tests
that override is a dead end for real workloads: the memory-aperture layout differs, so anything
using scratch or private addressing hits `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. Only a tiny
scratch-free SGEMM survives, which is probably why the override has looked promising.

Following the approach of
[ROCm/rocm-libraries PR #8838](https://github.com/ROCm/rocm-libraries/pull/8838), rocBLAS was
instead built natively for gfx1013 on Fedora's system ROCm. That meant working through a chain of
Fedora-specific issues: system ROCm lives in `/usr` rather than `/opt/rocm`; `amdclang++`,
`msgpack-cxx`, and `roctracer` were missing; and gfx1013 had to be added to Tensile's `SupportedISA`
and `AsmCaps` and to the Tensile and rocBLAS C++ architecture enums. The full worked recipe is in
[`scripts/build_rocblas_gfx1013.sh`](scripts/build_rocblas_gfx1013.sh).

The result is a real `librocblas.so.4.4` (about 38.8 MB) with 56 gfx1013 Tensile libraries and
genuine gfx1013 code objects (`Kernels.so-000-gfx1013.hsaco` reports ELF machine "AMD GPU" with
gfx1013 flags, native rather than an override). It runs on the board with no `HSA_OVERRIDE` and
computes correct GEMMs, per the table above. The main value is that it removes the override from
the picture: where a native rocBLAS GEMM works it is correct, and where it does not it is the
wedge, not an aperture mismatch. On its own it is not enough for reliable inference, because the
wedge still applies to the large fused matmuls that inference leans on.

## Observation 3: the unlock, the fix, and the wedge are entangled

Two facts collided here. First, without the 40-CU unlock the board runs at 24 CU, and at 24 CU even
a trivial compute dispatch wedges: `compute_probe` returns correct results at 40 CU and hangs at 24
CU. So the community
**40-CU unlock** ([duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock))
appears to be a prerequisite for any ROCm compute here, not just inference, which is
counterintuitive (more CUs, more stable) and hints the wedge is tied to the harvested-CU / WGP-mask
configuration.

Second, in the kernel tree used here the 40-CU unlock's register write lives inside
`gfx_v10_0_kiq_reset_hw_queue()`, a function that only runs when a KIQ hardware queue is reset. On
the stock driver, the KIQ-fence bug triggers such a reset during boot, which incidentally fires the
unlock, so the board comes up at 40 CU.

Together those appear to undercut the correctness change on this board:
`flush_pasid_uses_kiq = false` removes the KIQ activity that was triggering the reset, so the unlock
tends not to fire, the board comes up at 24 CU, and at 24 CU compute wedges. Testing this directly with
the native rocBLAS GEMM, on the patched module at 24 CU every size wedged, including N=256
([`logs/rocblas/sgemm_sweep_patched_24cu.log`](logs/rocblas/sgemm_sweep_patched_24cu.log)). An
earlier session did once boot the patched module at 40 CU, which is where the correct patched
`compute_probe` results (Observation 1) and the prefill pass below came from, but that
patched-and-40-CU state did not reproduce on later boots.

So on this board the available states appear to be the correct TLB flush at 24 CU (where compute
wedges) or the working 40-CU configuration with the buggy flush (wrong results and freeze), but not
both. The obvious escape, moving the unlock write out of the reset path into normal init, did not
compute cleanly in attempts here: the module reported 40 CU but wedged on the first dispatch, as if
reconfiguring the shader arrays around init corrupts the compute setup. That relocation may simply
have been done wrong; a clean way to decouple the two would be very welcome.

The direct "is the wedge a regression" experiment was attempted two ways, both inconclusive for
frustrating reasons. A stock older kernel (Fedora 6.6.14) does not bring this board up at all:
amdgpu's display code faults during KMS init
([`logs/older-kernel-6.6-display-oops.log`](logs/older-kernel-6.6-display-oops.log)), and on the
kernels tested, BC-250 support appears only from about kernel 6.18 (Fedora's 6.18.9 amdgpu exposes
`bc250_cc_write_mode`; its 6.17.1 does not). Reverting the one named TLB regression on 6.18 lands
back in the 24-CU-wedges-everything state above. So whether the wedge itself is a regression or a
hardware limit is unresolved here; the graphics-queue contrast leans toward a hardware or firmware
cause, held loosely.

## How far ROCm inference gets

With the patched module, and subject to its 40-CU caveat, llama.cpp's HIP backend gets further than
before, though not to a usable state.

A correct prompt-processing pass was achievable with a native gfx1013 build and rocBLAS kept out of
the hot path:

```bash
cmake -B build-hip -S . -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 \
  -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja -C build-hip llama-bench

HSA_ENABLE_SDMA=0 GGML_CUDA_FORCE_MMQ=1 \
  ./build-hip/bin/llama-bench -m qwen2.5-1.5b-q4km.gguf -ngl 999 -fa 1 -p 128 -n 0 -mmp 0
# qwen2.5-1.5B Q4_K_M, pp128 about 35 tok/s, correct, RC=0
```

`-fa 1` (flash attention) and `GGML_CUDA_FORCE_MMQ=1` route around rocBLAS via ggml's own gfx1013
kernels; `HSA_ENABLE_SDMA=0` is still needed on this board. Recipe:
[`scripts/native_fa.sh`](scripts/native_fa.sh).

Token generation faults immediately, non-deterministically as a hang, a segfault, or an aperture
violation, even for a single token, on the native build. The high-frequency decode dispatch pattern
appears to hit the wedge reliably, and even repeated prefill (`-r 3`) is only marginally reliable.
Logs: [`logs/inference/`](logs/inference/). This is the wedge from Observation 2, not a separate
problem. For scale, even where ROCm prefill completes it was roughly 36 times slower than Vulkan on
the same model and did not survive repetition, so it is "interesting that it runs at all" rather
than a usable backend.

## ROCm vs Vulkan

Vulkan appears here only as the baseline the ROCm path is measured against; the full Vulkan
characterization of the board (many models, context scaling, memory ceilings) lives in
[akandr/bc250](https://github.com/akandr/bc250) and is not repeated here. On the one model where both
backends reach a working point, qwen2.5-1.5B Q4_K_M (`llama-bench`, flash attention on, `-mmp 0`, 40
CU, tokens/sec):

| backend | pp128 | pp256 | pp512 | tg128 | reliable? |
|---------|:-----:|:-----:|:-----:|:-----:|-----------|
| Vulkan (RADV) | 1275.8 | 1597.7 | 1845.6 | 210.7 | yes, every run |
| ROCm / HIP (patched, native gfx1013, 40 CU) | about 35 | fault | fault | fault | no |

Even where ROCm prompt processing completes it is roughly 36 times slower than Vulkan on pp128, does
not survive repetition, and does not generate tokens at all
([`logs/inference/bench_rocm_vs_vulkan.log`](logs/inference/bench_rocm_vs_vulkan.log)).

## Status snapshot

Offered as a current read, not a verdict; several rows could change with better ideas or newer
firmware.

| Thing | Where it landed |
|-------|-----------------|
| Silent wrong results on large compute | Appears addressable in isolation via `flush_pasid_uses_kiq = false`, but entangled with the 40-CU unlock (Observation 3), so not usable end to end yet |
| KIQ board-freeze on HIP exit | Mitigable: the same patch, or userspace `amdgpu.sched_policy=2` |
| Multi-minute model load | Much faster with the same patch |
| rocBLAS has no gfx1013 kernels | Buildable natively (the PR #8838 approach), or bypass via flash attention plus `GGML_CUDA_FORCE_MMQ=1` |
| gfx1010 override aperture violations | Avoided by building native gfx1013 |
| Compute wedges at 24 CU or under large / sustained load | Not resolved here; looks like a MEC/compute-queue liveness limit, consistent with why Mesa disables the queue |
| Token generation | Not working, blocked by the wedge |

In short, the correctness side looks like it may be software (a change that removes it in isolation
was found), and the liveness/wedge side looks like hardware or firmware. Both are inferences from
one board, and the entanglement in Observation 3 prevented turning the correctness finding into a
working end-to-end ROCm setup. Any of it could be wrong.

## Open questions

Places where other eyes would help most:

- Can the 40-CU unlock be decoupled from the KIQ reset, so a `flush_pasid`-patched module also comes
  up at 40 CU and computes? Relocation attempts here wedged.
- Is the compute-queue wedge a regression or a hardware/firmware limit? The clean test, an old
  pre-regression kernel that also supports the BC-250, is blocked because BC-250 support landed
  upstream only around 6.18. A backported old kernel might settle it.
- Would newer or different MEC/MES firmware change the preemption behaviour? (MES, MicroEngine
  Scheduler, is a newer firmware scheduler that can manage the compute queues instead of the older
  path.) This board runs with `mes=0` and no Cyan Skillfish MES firmware present, which feels like
  the most likely source of a
  real fix, but is out of reach here.
- Did the mining stacks really run sustained compute on this exact path, and if so what did their
  kernel and firmware combination do differently? [ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313)
  hints that gfx1013 worked under older ROCm and kernel combinations.

Data, corrections, or a "you are holding it wrong" are all welcome, as an issue here or a note on
[ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313).

## Reproducing

- kernel 6.18.9-200.fc43, rocm-hip 6.4.2, rocblas 6.4.4, ROCm LLVM/clang 19, mesa 25.3, llama.cpp
  build 2da6686.
- [`reproduce.sh`](reproduce.sh) runs the compute probe (small and large) and the rocBLAS probe.
  Boot with `amdgpu.sched_policy=2` first, since the HIP probes can hang the board on process exit
  at the default scheduler.
- Correctness change: build the patched module
  ([`scripts/build_patched_amdgpu.sh`](scripts/build_patched_amdgpu.sh)), reboot, rerun the probe.
- Native gfx1013 rocBLAS: [`scripts/build_rocblas_gfx1013.sh`](scripts/build_rocblas_gfx1013.sh),
  then [`patches/sgemm_sweep.cpp`](patches/sgemm_sweep.cpp).

Two things that cost time. A rebuilt kernel module must be compressed with `xz --check=crc32` (the
build script does this): the xz default is CRC64, which loads via userspace `modprobe` but fails the
in-kernel decompressor from the initramfs (`decompression failed with status 6`), while `xz -t`
still passes, so it looks like a bricked board. Keep a verified-good module backup. And after a
compute wedge a soft reboot often does not recover the queue; a hard power-cycle does. Always verify
`active_cu_number 40` in dmesg and a good `compute_probe` or GEMM run before trusting a setup.

## Files

| Path | What it is |
|------|------------|
| [`patches/amdgpu-flush-pasid-mmio.patch`](patches/amdgpu-flush-pasid-mmio.patch) | The one-line kernel change for the correctness observation |
| [`patches/compute_probe.c`](patches/compute_probe.c) | Bare HIP compute reproducer (native gfx1013, CPU-checked) |
| [`patches/ocl_compute_probe.c`](patches/ocl_compute_probe.c) | OpenCL port of the probe (graphics-queue comparison via RustiCL) |
| [`patches/sgemm_sweep.cpp`](patches/sgemm_sweep.cpp) | Native gfx1013 rocBLAS SGEMM sweep with CPU check and timing |
| [`patches/rocblas_probe.c`](patches/rocblas_probe.c) | Standalone rocBLAS SGEMM availability/correctness test |
| [`scripts/build_patched_amdgpu.sh`](scripts/build_patched_amdgpu.sh) | Build the patched amdgpu module (module-only) |
| [`scripts/build_rocblas_gfx1013.sh`](scripts/build_rocblas_gfx1013.sh) | Build a native gfx1013 rocBLAS on Fedora system ROCm |
| [`scripts/native_fa.sh`](scripts/native_fa.sh) | The native gfx1013 plus FA plus MMQ inference recipe |
| [`logs/`](logs/) | Captured run logs: correctness, rocBLAS sweeps, RustiCL comparison, inference, benchmarks, older-kernel attempt |

## References

- [ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313): BC-250 system freeze after compute workloads. anrp and ahorek found `flush_pasid_uses_kiq = false`; AMD is engaged in the thread.
- [Mesa MR !33116](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/33116) and [Mesa 25.1 release notes](https://docs.mesa3d.org/relnotes/25.1.0.html): RADV disables the gfx1013 compute-only queue (commit `7271b8ee`). The MR is by Ivan Avdeev (`provod` on GitLab, `w23` on GitHub), a community contributor, not an AMD employee and not "RADV's author".
- [Mesa issue #11982](https://gitlab.freedesktop.org/mesa/mesa/-/issues/11982): AMD Cyan Skillfish support discussion.
- [ROCm/rocm-libraries PR #8838](https://github.com/ROCm/rocm-libraries/pull/8838): adding gfx1013 support to rocBLAS.
- [kernel bug #216645](https://bugzilla.kernel.org/show_bug.cgi?id=216645): a different system (a Dell laptop with a Navi/RDNA1 RX 5600M) hanging with "Fence fallback timer expired" and amdgpu interrupts ceasing. Not a BC-250 report, but the same fence-fallback / lost-interrupt symptom this board prints every boot, so it is useful background.
- [duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock): the 40-CU unlock and the module-build pipeline reused here.
- [github.com/akandr/bc250](https://github.com/akandr/bc250): the related BC-250 Vulkan setup.

## Author and license

Author: Artur Andrzejczak. Prepared with assistance from Claude.

Code: [AGPL-3.0](LICENSE). Docs: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
