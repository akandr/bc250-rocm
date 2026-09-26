# BC-250 (gfx1013): the investigation

Companion to the [README](README.md), which carries the recipe and the measurements. This file is
the road: how each defect was found, what was measured along the way, and what each measurement does
and does not establish. It is long because the board has a lot of separate problems and each one
needed its own experiment.

These are notes from one board and one software stack. The measurements are reproducible and
included as logs; the explanations are working theories and may be wrong or incomplete. Corrections
and "have you tried X" comments are welcome; see [Open questions](#open-questions) at the end.

If you only want the board working, the [README](README.md) has the whole recipe. One result from
below is worth repeating at the top, because it saves the most time: the kernel version is not the
determining factor. The same recipe measures identically on six kernels from 6.18.9 to 7.2.5, so
what matters is a patch set and a set of boot arguments, not a kernel to upgrade to.

Native gfx1013 rocBLAS sustained 4.5 to 4.6 TFLOP/s of verified-correct SGEMM. PyTorch, once built
for gfx1013, runs everything tried here including training, where a 50-step loop tracks the CPU to
1.799e-05 per step on the losses and returns an identical final loss on all fourteen runs of an
eight-hour soak; the stock ROCm wheel cannot run a library operation at all, aborting at the first
one because it ships no Tensile library for this architecture. llama.cpp, patched and gated as
described below, ran a five-model campaign with every number passing a wikitext perplexity gate
against the Vulkan backend on the same build and boot: 113.5 t/s decode and 805.6 t/s prefill on a
1.5B. A build compiled with forced cuBLAS was reported at 936, but that is not part of the recipe
here: the flag is a compile-time option instead of an environment variable, and no capture in this
repository supports the figure. Decode came within a
few percent of Vulkan on an 8B at Q8_0, 34 t/s on a 35B MoE, and a 27B model released after this
work runs on both backends with the two agreeing on perplexity to 0.3 percent. How much of this is
stable over time, not a good run on one board, is not something a single board can answer.

Those figures are the three-patch build, which is where this document's narrative begins. The patch
set has since grown to thirteen and the same 1.5B reads 196.7 t/s decode as shipped, 213.1 with the
multi-stream option the README now recommends, and 1799 t/s prefill, with two of the six models
faster than Vulkan on both prefill and decode. The [README](README.md) carries the current
numbers; this file keeps the ones each section was written against, because the point of it is how
the conclusions moved.

Limits that looked like hardware kept dissolving into software as the instruments improved:
current llama.cpp needs three small patches on this board (a device-flag regression, a missing
precision request, and a missing architecture-macro entry that had flash attention garbling for
months and the quantized kernels running 9x slow), plus the native rocBLAS and one environment
variable, each traced and verified; see the caveats.
What remained after them, as this was written: prefill trailed Vulkan by 1.4x to 2.3x depending on the
model and narrowing as models grow, which is a gap between llama.cpp's quantized matmul kernels and
Vulkan's and not anything failing, and which ten further patches have since closed to between 0.96
and 1.29 of Vulkan; the allocation-reuse defect needs its kernel-side flush (traced to the
end and fixed here, but a
workaround rather than something upstream has taken); the fp16 GEMM path returned nothing for one
attention projection, which after a long hunt turned out to be a toolchain packaging defect rather
than anything on the GPU (the builtins archive in Fedora 43's ROCm compiler-rt converts half
precision through the wrong register, and the native rocBLAS links it in), repaired here by patching
two helper functions and still unfixed in the package
([`logs/fp16-root-cause-2026-09-15/`](logs/fp16-root-cause-2026-09-15/)); and everything here is one
board. SDMA was on this list for weeks and is not any more:
substituting the navi12 microcode makes every transfer tried here complete, so the defect is in the
microcode the board ships and not in the silicon.
This took the combined work of several community projects, credited inline and in the references.

This is the ROCm/HIP companion to [akandr/bc250](https://github.com/akandr/bc250), which covers
the board itself and its (working) Vulkan setup; that background is not repeated here. For a long
time the Vulkan side was the only usable one, and these notes documented how far the ROCm/HIP
stack could be pushed before hitting a wall. The investigation is kept intact below, both because
the observations remain real on older kernels and configurations, and because they explain what
the working configuration actually changes.

Environment throughout: Fedora 43, ROCm 6.4.2 (rocBLAS 6.4.4 as shipped, plus a native gfx1013
rocBLAS built locally), LLVM/clang 19, Mesa 25.3 RADV for the Vulkan comparison, the community
40-CU unlock, the oberon governor around 1500 MHz. The historical observations were taken on
kernel 6.18.9-200.fc43 and most of the new benchmarks on 7.1.5-100.fc43. That difference turned
out not to matter: with the same patch set, both kernels measure identically, so the kernel
version is a detail of when the work was done, not a condition for it.

## Contents

- [A short primer: the AMD compute stack](#a-short-primer-the-amd-compute-stack)
- [The claim this repo tests](#the-claim-this-repo-tests)
- [A working configuration](#a-working-configuration)
- [What the working configuration measures](#what-the-working-configuration-measures)
  - [Everything that used to fail, rerun](#everything-that-used-to-fail-rerun)
  - [SGEMM throughput (native gfx1013 rocBLAS)](#sgemm-throughput-native-gfx1013-rocblas)
  - [Inference-path caveats: what the perplexity gate caught](#inference-path-caveats-what-the-perplexity-gate-caught)
  - [llama.cpp: ROCm vs Vulkan, same build, same boot configuration](#llamacpp-rocm-vs-vulkan-same-build-same-boot-configuration)
  - [A newer and larger model: Qwen3.8-27B](#a-newer-and-larger-model-qwen38-27b)
  - [Decode at context depth](#decode-at-context-depth)
  - [The allocation-reuse defect, a reproducer, and the flush that fixes it](#the-allocation-reuse-defect-a-reproducer-and-the-flush-that-fixes-it)
  - [PyTorch, and a note on allocation discipline](#pytorch-and-a-note-on-allocation-discipline)
  - [PyTorch built for gfx1013](#pytorch-built-for-gfx1013)
  - [Other things that work, checked once each](#other-things-that-work-checked-once-each)
  - [ROCm-only capabilities](#rocm-only-capabilities)
  - [What still fails, measured](#what-still-fails-measured)
- [Observation 1: occasional silent wrong results](#observation-1-occasional-silent-wrong-results)
- [Observation 2: the compute queue wedges under load](#observation-2-the-compute-queue-wedges-under-load)
- [Building a native gfx1013 rocBLAS](#building-a-native-gfx1013-rocblas)
- [Observation 3: the unlock, the fix, and the wedge looked entangled](#observation-3-the-unlock-the-fix-and-the-wedge-looked-entangled)
- [How far ROCm inference gets](#how-far-rocm-inference-gets)
- [ROCm vs Vulkan](#rocm-vs-vulkan)
- [Fedora 43 with ROCm 6.4.2](#fedora-43-with-rocm-642)
  - [Making it work](#making-it-work)
  - [llama.cpp inference](#llamacpp-inference)
  - [GPGPU](#gpgpu)
  - [Known defects](#known-defects)
  - [Limits](#limits)
  - [Reproducing](#reproducing)
  - [How this was arrived at](#how-this-was-arrived-at)
- [Fedora 44 and 45: ROCm 7.1.1 and 7.2.2](#fedora-44-and-45-rocm-711-and-722)
- [How the measurements here are controlled](#how-the-measurements-here-are-controlled)
- [Open questions](#open-questions)
- [Reproducing](#reproducing)
- [Files](#files)
- [References](#references)
- [Author and license](#author-and-license)

## A short primer: the AMD compute stack

This section lays out the vocabulary the rest of the document uses, working from an application down
to the silicon, simplified.

### The one-picture version

A program like llama.cpp can reach the same GPU by two completely separate software roads. One is
built for graphics, the other for general-purpose compute. When this investigation started only the
graphics road worked, so the document keeps contrasting them; most of the compute road
works now, and the [README](README.md) is the recipe for that:

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
here. llama.cpp's HIP backend uses this road, the harder of the two on this chip, and most of this
document is about why.

**KFD.** The kernel-side half of ROCm is the **KFD** (Kernel Fusion Driver), part of the `amdgpu`
module. It sets up the compute queues, doorbells, and per-process GPU memory maps that HIP programs
use. "The compute queue is broken" points at something in this path.

### How work actually reaches the GPU: queues

The driver hands work to the GPU through hardware **queues** (command rings the GPU pulls from, like
a to-do list). Two matter here:

- the **graphics / universal queue**, driven by the GFX engine, and
- the **compute queue(s)**, driven by the **MEC** (MicroEngine Compute), a small firmware processor
  on the GPU dedicated to compute dispatches.

The distinction that shapes this whole document: **ROCm/HIP sends its compute to the MEC compute
queue**, while Vulkan/RADV normally uses the graphics queue. Same shader cores at the bottom,
different route to reach them, which is how Mesa can route around a problem on the compute queue
and ROCm cannot. (An OpenCL path called **RustiCL**, part of Mesa, also goes by the graphics-queue
route, and was used as a control here. It no longer distinguishes the two, since under the working
configuration compute is correct on both; the section that used it explains what changed.)

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

Mesa's source, for this chip, carries the comment `GFX1013 is known to have broken compute queue`
and [disables the compute-only queue for it](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/33116),
routing compute through the graphics queue instead. The mechanism, checked against Mesa main in
August 2026 and unchanged when asked again on 21 September, the comment still sitting at
`ac_gpu_info.c:501`: in `src/amd/common/ac_gpu_info.c`, the code that fills in each IP block's queue count
returns early for `AMD_IP_COMPUTE` on this ASIC, so the compute IP is left reporting zero queues,
and `radv_compute_queue_enabled()` then sees `num_queues > 0` fail and reports no compute queue.
It sits in the shared AMD code and not in RADV itself, so it applies to Mesa's other AMD
drivers as well. ROCm has no equivalent escape hatch: its compute goes to the compute queue, so it
cannot side-step the problem the same way.

## The claim this repo tests

Getting ROCm working would open up the wider GPGPU ecosystem on the board (rocBLAS, PyTorch,
image generation, and so on). The stock answer is that it cannot: the compute queue is broken.
The notes below test that claim.

What the single label "broken compute queue" turned out to cover, on this board, is a set of
distinct problems with distinct causes, most of them in software:

- a TLB-invalidation defect that produced silent wrong results, fixed by correcting the PASID
  flush
- a wedge under sustained compute that belongs to the software-scheduling eviction path, avoided
  by not setting `amdgpu.sched_policy=2`, which was itself adopted here as a workaround for the
  first problem
- an allocation-reuse defect where a stale translation survives into a fresh mapping, fixed by
  rebuilding the runlist on map
- missing gfx1013 code objects in the shipped rocBLAS and PyTorch, fixed by building both
- defects in llama.cpp's HIP backend. Three were traced and patched early: a device-flag
  regression, a missing precision request, and a missing architecture-macro entry. A fourth had a working environment variable, not a patch, HIP graph instantiation at
  depth, and a fifth was open, an intermittently zeroed fp16 GEMM. It is easy to count that one twice,
  since "forced-MMQ wrongness on qwen3 dense models" is the same defect seen through a flag that
  does nothing. That fifth one is a Fedora 43 toolchain bug rather than a llama.cpp one, and is
  absent on Fedora 44. Since then the list has grown: the build carries thirteen patches, adding a
  flash-attention register spill, a matrix-vector kernel in two passes, a transposed concat, the
  linear-attention lane count, and a packed-fp16 prefill GEMM in five parts
  ([`patches/llamacpp/`](patches/llamacpp/), and [README.md](README.md) for what each is worth)

None of those required new hardware or firmware, and the residue that still looks like silicon is
small: the `hqd_destroy` preemption timeout itself, which is only reachable by asking for the
software scheduler and so does not arise in the working configuration, and a rare extreme-size
dispatch fault, seen twice in a day of boots and not at all in a later five-run repeat at the
largest size. This sentence named a third, an SDMA path that could not move a model into memory,
and called it the one clear-cut case; it was removed on 27 August because it is not one. The
navi12 microcode substitution fixes SDMA outright, as the summary above already said and as this
paragraph had not caught up with, and what looked like the clearest silicon defect on the list
turned out to be the wrong firmware. The
sections that follow give the current answer;
the observations after them are the investigation that led to it, kept as recorded, with notes
where later results corrected them, of which there are several.

## A working configuration

Three ingredients that were once reported as mutually exclusive can coexist, and together they
remove most of the failure modes documented below. Most of the work here was done on 7.1.5, which
is why it appears throughout, but the kernel version is not one of the ingredients: 6.18.9 with
the same patch set measures identically, and the evidence is below.

```
kernel 6.18.9 or later (measured on 6.18.9 and 7.1.5; both identical)
+ the 40-CU unlock                   amdgpu.bc250_cc_write_mode=3
+ the corrected PASID TLB flush      flush_pasid_uses_kiq = false (the patch from Observation 1)
+ hardware scheduling                do NOT set amdgpu.sched_policy=2
+ the runlist-rebuild flush         amdgpu.bc250_flush_by_runlist=3 (bitmask: 1 unmap, 2 map;
                                     value 3 needs two patch scripts, in order:
                                     apply_runlist_flush.py then
                                     apply_svmflush_generic.py; see the
                                     allocation-reuse section below)
+ SDMA fixed or avoided              the navi12 microcode substituted for the
                                     board's own (preferred, it fixes the
                                     transfers rather than avoiding them), or
                                     HSA_ENABLE_SDMA=0 in the environment for
                                     HIP processes; see the SDMA section below
```

Two findings make this possible. Both are corrections to observations below, and both are scoped to
the kernel version, so they were missed earlier.

**The unlock and the flush fix no longer conflict.** Earlier work found the corrected flush
forcing the board to 24 CU, where compute wedges (Observation 3). That no longer happens: the
board comes up at 40 CU with the corrected flush, verified on every test boot by the boot log (the
patched module prints its flush state at init, so a stale-initramfs mixup is excluded). A later
rebuild found the same on 6.18.9, 6.18.16 and 6.19.14 alike, so it is not a property of the newer
kernel; why the earlier boots behaved differently is unresolved, and the correction in
Observation 3 sets out both the measurements and the limits of what they explain.

**The flush and the scheduler interact; both changes are required.** `amdgpu.sched_policy=2` was
adopted here early, from the community freeze workaround, and every wedge measurement below was
taken under it. To untangle the two variables properly, a full two-by-two factorial was run on
7.1.5: `flush_pasid_uses_kiq` (false / true) crossed with `sched_policy` (default hardware
scheduling / 2), two boots per cell in a mirror-balanced order (A B C D D C B A), a fixed
measurement battery per boot (three 8.4M-thread correctness probes, SGEMM N=2048 x10 and N=4096
x50, dmesg counters, a health check), and everything else identical:

| cell | flush | scheduling | result (2 boots each) |
|---|---|---|---|
| A | false | hardware (default) | clean both boots (and a third pilot): probes 3/3 correct, both GEMMs complete correct, zero preemption timeouts, board responsive |
| B | false | 2 (software) | probes hang 3/3, both GEMMs wedge, 19 `cp queue preemption time out` per boot, queue degraded |
| C | true | hardware (default) | board lost mid-battery, both boots (the KIQ-flush freeze; recovered by power cycle) |
| D | true | 2 (software) | the historical configuration: probes mostly pass, sustained N=4096 wedges or faults on both boots |

The table reads as an interaction. Cell C is why `sched_policy=2` existed at all: under
hardware scheduling with the stock flush, the KIQ PASID flush freezes the board, so the software
scheduler was the rational mitigation. Cells B and D show the price: under software scheduling,
queue evictions preempt through `kgd_hqd_destroy`, which is where the wedge message is
printed (Observation 2), and sustained compute wedges with either flush setting. Only cell A,
both changes together, is clean: the corrected flush removes the freeze that made hardware
scheduling unsurvivable, and hardware scheduling removes the eviction path that made compute
wedge. Two boots per cell is thin for a board that varies this much between boots (the
flash-attention section is a caution on exactly that), so this is the pattern that held across
these boots, and not a law.

Cell A was originally reported as unreachable on 6.18, because the corrected flush appeared to
force 24 CU there (Observation 3). That has since been measured directly and it is not the case.
With the same patch set and boot arguments, kernel 6.18.9 is indistinguishable from 7.1.5 on every
check run here: the compute probe correct at all three sizes, the native rocBLAS SGEMM sweep clean
from N=256 to N=4096, a sustained N=4096 for 50 iterations clean, perplexity 8.9442 to the fourth
decimal, and zero faults in dmesg on both. The kernel version was never the ingredient that
mattered; the earlier readings came from a misapplied unlock patch and from `sched_policy=2` being
held fixed. Logs in [`logs/kernel-equivalence-2026-08-17/`](logs/kernel-equivalence-2026-08-17/).

**The scheduler policy is the determinant, and CU count is not.** The claim that a 24-CU board
wedges even on a trivial dispatch appeared in earlier write-ups here. Crossing CU count with
scheduler policy, all four cells on kernel 7.1.5 with the same module, one boot each:

| CUs | scheduling | compute probe | SGEMM N=256 | `preemption time out` in dmesg |
|---|---|---|---|---|
| 40 | hardware (default) | correct, 3 of 3 sizes | clean | 0 |
| 24 | hardware (default) | correct, 3 of 3 sizes | clean | 0 |
| 40 | `sched_policy=2` | hangs, both sizes tried | wedges | 10 |
| 24 | `sched_policy=2` | hangs at the larger size, smaller completes then hangs at exit | wedges | 9 |

Both hardware-scheduling rows are clean and both software-scheduling rows wedge, at either CU
count. At 24 CU with hardware scheduling the board even passes the full battery, perplexity
included, at 8.9442. The historical "24 CU wedges" observations were `sched_policy=2`
measurements that happened to be taken at 24 CU, and the CU count carried the blame. The same
policy applied to an otherwise clean 6.18.9 reproduces the failure there too, the probe hanging at
all three sizes and SGEMM wedging at N=256, so this is not specific to one kernel either
([`logs/kernel-equivalence-2026-08-17/`](logs/kernel-equivalence-2026-08-17/)).

This also reframes the kernel-7.1.5 test reported in Observation 2, which found both defects
persisting on the newer kernel: that boot carried `sched_policy=2` on its command line, because at
the time that was standard practice here. The failures it recorded were real, but they belong to
the software-scheduling path, not to the kernel version.

### Why this was missed

A short methodological accounting, since the earlier conclusions leaned toward hardware and were
wrong in their scope. Four things compounded:

- **A workaround became an unexamined constant.** `sched_policy=2` was adopted early as the freeze
  mitigation and then carried on every command line, including the newer-kernel test. Under it,
  evictions preempt through the same driver path that emits the wedge message. Every wedge
  measurement was taken inside the failure mode the mitigation itself selected.
- **A misapplied patch produced a fake constraint.** The 40-CU unlock of the time was applied into
  a `gfx10_kiq_*` function and not `gfx_v10_0_get_cu_info()`, which makes a module that loads
  while the board stays at 24 CU. Combined with the workaround above, that produced the reading
  that the correctness fix "cost 16 CUs" and that 24 CU wedges everything. Both halves are wrong:
  the fix and 40 CU coexist on every kernel retested, and at 24 CU with hardware scheduling the
  board passes the whole battery including perplexity. What looked like two interacting hardware
  blockers, each defeating one-variable experiments, was one bad patch and one fixed boot argument.
- **Intermittency degraded the knob sweep.** The 6.18 sweep sampled each knob a few times in a
  regime where the base failure rate drifts by boot and by session, so a false negative on any
  single knob (including `sched_policy=0`) was likely enough. It also does not record which flush
  its module carried, which limits what any of its rows can settle.
- **Throughput was mistaken for correctness in inference.** Token rates and clean exits do not
  prove the tokens are right (see the flash-attention note below). A seed-fixed text check now
  accompanies every inference claim here.

A fifth belongs on the list, since it is the one that produced the most confident wrong
conclusion: **the kernel version was changed at the same time as the patch set.** Every "this works
on 7.1.5" statement in earlier revisions of this document was really "this works with the corrected
flush, hardware scheduling and a correctly applied unlock", and the kernel came along for the ride.
Measuring 6.18.9 with the same patch set, and getting identical results on every check, is what
separated them.

Fairly attributed to the hardware or firmware: the underlying TLB-invalidation
oddities, the load-time host-aperture fault, the rare extreme-size dispatch fault, and the
`hqd_destroy` preemption timeout itself. What does not: the practical unusability, which was a
stack of driver-path and userspace choices that a different configuration avoids on any kernel
tested here.

## What the working configuration measures

**August 2026, the three-patch build.** These are the ratios for that build;
[README.md](README.md#llamacpp-rocm-against-vulkan) carries the current ones, measured on the
thirteen-patch build. The correctness results and the defect analysis below apply to both, and the
sections after this argue from the numbers here.

All numbers in this section are from kernel 7.1.5 at 40 CU with the corrected flush and hardware
scheduling, native gfx1013 code throughout, no `HSA_OVERRIDE`, one `llama-bench` invocation per
test. The historical rows use llama.cpp build 2da6686
([`logs/bench-2026-08/`](logs/bench-2026-08/)); the fixed-stack campaign uses master 7ba604f
with the patches from the caveats ([`logs/bench-fixed-2026-08/`](logs/bench-fixed-2026-08/)).

### Everything that used to fail, rerun

| workload | historical result | working configuration |
|---|---|---|
| SGEMM N=2048 x10 | protection fault | correct; the shipped sweep at this size runs twenty iterations, not ten, and reports `CORRECT` |
| SGEMM N=4096 x50 sustained | wedge, near-every-run | correct; the 20-iteration sweep measures 4.5 TFLOP/s at this size |
| SGEMM N=4096 x200 sustained | never survived | correct on the day; no capture here runs SGEMM two hundred times, the only x200 in the logs being the OpenCL probe at 1M threads |
| SGEMM N=8192 x20 sustained | wedge / occasional corruption | correct, 4.6 TFLOP/s |
| 10 rapid single-GEMM processes | queue degradation, stalls | clean, though the 10/10 is a count from the day and no log here records it |
| streaming-read probe, 1 and 2 GB | abort at 1 GB, wedge at 2 GB | `fails=0/10` at 2 GB, the size the shipped capture records |
| compute probe, 8.4M threads | failed on 4/4 boots (July, policy 2) | correct on every run since, see below |
| compute probe sweep 1M to 16.7M | wrong results / faults / hangs | correct on every run since, see below |
| HIP process exit | freeze risk | clean exits throughout |

The two probe rows read "17/17 in a counterbalanced A/B" and "30/30 across the two benchmark
boots" until 27 August, and neither denominator can be produced from what is shipped: the probe
runs live in `logs/factorial/`, `logs/kernel-equivalence-2026-08-17/`, `logs/bench-2026-08/` and
`logs/stock/`, not in the two campaign directories this section cites, and no directory here
holds a counterbalanced probe A/B at all. What is captured is stronger than either fraction and
easier to check. The probe reports a result 107 times across these logs and 105 of them are
correct. Ninety-two of those carry the banner naming their size, thirty-four at the 8.4M-thread
size with thirty-three correct; the remaining fifteen come through harnesses that print the result
without the banner. Both failures are explained and neither is on the working configuration: the 8.4M one is `total_wrong=525308` in
`logs/stock/compute_probe_stock_1500-1000.log`, which is the stock flush and the failure this row's
"before" column describes, and the other is the deliberate `HSA_OVERRIDE` trap in
[`logs/override-trap-2026-08-26/`](logs/override-trap-2026-08-26/), where every element is zero by
design.

The July side of that row is a day count too. "Failed on 4/4 boots" appears in no log here, and
what the shipped captures from the failing configuration actually show is one failure at that size:
`logs/stock/compute_probe_stock_1500-1000.log` runs the probe seven times and reports
`total_wrong=525308` at `nblocks=32768`, which is the 8.4M-thread size and the figure Observation 1
quotes. The "before" rests on a single captured failure and not by four boots, and
the four is what was counted at the time.

### SGEMM throughput (native gfx1013 rocBLAS)

Twenty iterations per size, every result checked against a CPU reference, all correct:

| N | median ms per GEMM | GFLOP/s |
|---|---|---|
| 512 | 0.2 | about 1340 |
| 1024 | 0.8 | about 2680 |
| 2048 | 5.7 | about 3010 |
| 4096 | 30.0 | about 4580 |
| 8192 | 236.0 | about 4660 |

About 61 percent of a 7.68 TFLOP/s FP32 peak at the governor's 1500 MHz cap, from an untuned
Tensile build. That peak is the clock and lane count, not a measurement. Measured with dependent FMA
chains and the clock verified at its cap, this part sustains 6.52 TFLOP/s, against which the same 4660
is 71 percent ([`logs/alu-rates-recheck-2026-09-25/`](logs/alu-rates-recheck-2026-09-25/)). An earlier
revision put the measured rate at 4.74 and concluded that almost none of the shortfall belonged to
Tensile; that rate was taken below the clock cap and the conclusion went with it.

That the first column is a median matters more than it looks, and the reason is worth carrying here
rather than only in the log. Re-running twenty iterations per size and taking the plain average
puts N=512 at 542 GFLOP/s against the 1340 above, which reads as a large regression and is not one:
a single iteration at N=512 takes 6.8 ms against about 0.16 ms warm, so one cold call costs more
than the other nineteen together and drags the mean down. Remove one cold call of the measured size
and N=512 comes to about 1645 and N=1024 to about 2488, either side of the published figures. The
three larger sizes are insensitive to this and reproduce within a percent. Anyone re-measuring this
table should take a median, or say which statistic they took
([`logs/defects-recheck-2026-08-22/`](logs/defects-recheck-2026-08-22/)).

![SGEMM throughput](figures/fig-sgemm-curve.png)

### Inference-path caveats: what the perplexity gate caught

Before the inference numbers, a warning: a seed-fixed generation check is essential here, because
token rate and a clean exit do not prove the tokens are right. Later work added a stronger gate,
wikitext perplexity compared against the Vulkan backend on the same model and text (11.21 for the
1.5B used here); reading generated text turned out to miss corruption that perplexity catches
immediately, and most of the findings below were only visible through it. The reverse holds too,
measured, not assumed: with the RDNA1 macro entry removed from the working build,
perplexity reads 8.9425 and looks entirely healthy while the same build asked to generate text
returns `The???????????????`. That figure was cited here for some time with no surviving log
behind it, which a later check caught, so the arm was measured again from the working tree: reverting
the one line gives 8.9425 and `The???????????????????????`, restoring it gives 8.9442 and `The
capital of France is Paris.`, reproducing the original to four decimals
([`logs/macro-remeasure-2026-08-18/`](logs/macro-remeasure-2026-08-18/)). Perplexity is computed
over batched
prefill and never exercises the decode kernel that garbles, so the two gates catch different
faults and both are needed. Each caveat below is a distinct llama.cpp or library defect that first
looked like board behavior.

**Flash attention garbled for months, and the cause was a two-character architecture list.** The
symptom: `-fa on`
produced garbage at full speed on nearly every boot sampled (ten of eleven in the surviving tally; the
campaign's per-boot record was not retained), with
the garble bytes stable within a boot but different across boots, while `-fa off` stayed correct.
That pattern reads as boot-dependent hardware marginality, plausibly memory training
on the board's bottom-binned GDDR6, and a one-line software fix (adding gfx1013 to ggml's
RDNA1 architecture macro) can look irrelevant if an unpatched build happens to come up correct on
one good boot. Both readings are wrong, and the split observation that decides it is this:
`-fa on` computed *correct* results in batched perplexity runs while garbling only in token-by-token
generation, which pointed at the decode-specific flash-attention kernel and not the hardware.
The macro was retested with better instruments and is the fix: `vendors/hip.h` defines RDNA1 for
`__gfx1010__` and `__gfx1012__` but not `__gfx1013__`, so the host-side code (which classifies by
compute-capability number) selects kernel configurations for an RDNA device while the device code
compiled without the RDNA1 define takes different paths, and the decode kernel reads wrong. The
garble varying by boot, and the single good boot, are consistent with the mismatch consuming
whatever happened to be in memory. With the one-line macro fix: coherent decode on three prompts
across three fresh reboots, nine runs in all, with fa-on batched perplexity 9.9148 and fa-off
9.8574, each bit-identical across boots
([`logs/historical-sources/`](logs/historical-sources/) keeps the six perplexity logs; the nine
decode outputs themselves were read at the time and not retained, so the bit-identical perplexity
across boots is the part of this that can still be produced). So neither the boot lottery nor the GDDR6 speculation explains the flash-attention garble. The same macro line also enables the hand-written RDNA1 integer-dot emulation,
which took the quantized matmul kernels from 285 GFLOPS to 2.64 TFLOPS (9.2x, and see the note
below on what backs that) and the default
prefill from 124 t/s to the 800 to 890 range at pp512 with no other change (892 on the boot
where the fix landed, 806 to 808 on three later runs across two boots; pp2048 is stable at 661 to
662 throughout, so the pp512 spread is run to run, not configuration); `-fa on` decode
measures 113.5 t/s tg64 alongside prefill and 117.6 to 118.8 in a decode-only run, the best
decode numbers recorded on this board.

On the 9.2x, checked 26 August: neither 285 GFLOPS nor 2.64 TFLOPS appears in any capture in this
repository, so that ratio is quoted from a measurement that was not kept. The macro's effect is
captured, on both arms and in one sitting, but as end-to-end rates instead of kernel throughput:
with gfx1013 out of the macro the 1.5B prompts at 86.4 t/s and generates
`The???????????????????????`, and with it restored the same build prompts at 319.8 and generates
`The capital of France is Paris.`
([`logs/macro-remeasure-2026-08-18/`](logs/macro-remeasure-2026-08-18/)). That is a 3.70x prefill
difference. It does not refute the 9.2x, which is a different quantity measured on the kernels
rather than the pipeline, but it is the only speed comparison for this change that has a log behind
it.

Two things about that macro line were pinned down on 17 September, and one of them narrows the claim
above. First, the emulation is the correct path and not a workaround for a missing feature flag:
LLVM's `FeatureISAVersion10_1_3` in `llvm/lib/Target/AMDGPU/AMDGPU.td` adds only
`FeatureBVHRayTracingInsts` and `FeatureMSAALoadInsts` to the common 10.1 set, while gfx1011 and
gfx1012 add `FeatureDot1Insts` and its siblings. gfx1013 therefore has no `v_dot4_i32_i8` at all,
which `hipcc --offload-arch=gfx1013` confirms directly: `__builtin_amdgcn_sdot4` fails to compile
with `needs target feature dot1-insts`. gfx1013 belongs on gfx1010's emulation path, exactly where
the macro puts it.

Second, the sentence above that credits the prefill gain to "the RDNA1 integer-dot emulation" is
too narrow, and the instruction counts say so. Compiling both arms of `ggml_cuda_dp4a` for gfx1013
with divergent inputs gives 8 vector-ALU instructions for the RDNA1 inline asm against 10 for the
generic `#else` branch: the compiler already contracts the byte-wise fallback into `v_mul_i32_i24`
with SDWA byte selects, so the emulation is about 20 percent cheaper per dot, not several times
cheaper. A 3.70x end-to-end difference cannot come from that alone. The macro flips three things on
the current tree, not one: `ggml_cuda_dp4a` in `common.cuh`, the flash-attention tile configuration
in `fattn-tile.cuh` (`amd_rdna` against a generic AMD table), and `nthreads_KQ_q` in
`fattn-vec.cuh` (2 against 4). Which of the three carries the prefill gain has not been separated,
so the honest statement is that the macro as a whole is worth 3.70x and the attribution inside it
is open ([`logs/gfx1013-dot-isa-2026-09-17/`](logs/gfx1013-dot-isa-2026-09-17/)).

**The default-batch GEMM crash is the missing-code-objects problem, not an fp16 defect.** `-fa off`
at the default batch size crashes llama.cpp against the **system** rocBLAS. Verbose logging
(`AMD_LOG_LEVEL=3`) gives the exact cause: `Cannot find CO in the bundle
/usr/lib64/librocblas.so.4.4 for ISA amdgcn-amd-amdhsa--gfx1013:xnack-`, that is
`hipErrorNoBinaryForGpu`. That log line is quoted from a session whose output was not kept. The conclusion it
supports, that the system rocBLAS carries no gfx1013 code objects, is independently captured, since
its gfx1013 Tensile entries are symlinks to gfx1010 and a probe of the file listing records that
([`logs/torch-probe-2026-08-19/`](logs/torch-probe-2026-08-19/)). The system rocBLAS has no gfx1013
code objects embedded at all (its
gfx1013 Tensile files are only symlinks to gfx1010 ones), so it cannot run a rocBLAS GEMM that
needs a compiled kernel. This is the same missing-code-objects situation the native build (PR
#8838) exists to fix; it is not specific to fp16. It looks fp16-specific only because llama.cpp
reaches rocBLAS on its large-batch dequant path, while a small micro-batch (`-ub 8`) uses ggml's
own gfx1013 kernels and never calls rocBLAS.

Two follow-ons at the library layer, both clean and reproducible: the native gfx1013 rocBLAS runs
`rocblas_gemm_ex` correctly (fp16, N=256 through 4096), and `HSA_OVERRIDE_GFX_VERSION=10.1.0`
(reporting the device as gfx1010 so the embedded gfx1010 object loads) did not rescue this
particular path, where the dispatch hung. Narrower than it was once written as: retested
under the working configuration, a direct SGEMM through the system library does complete under the
override at N=256 through 4096. What the override reliably breaks is code compiled for gfx1013,
which stops running altogether without saying so (the rocBLAS section has the measurement). Tying
the crash to a specific end-to-end llama failure was harder to pin down at the time; the runs that
muddied that A/B later traced to a test-harness mistake (the multi-boot note below), and the
library-layer cause above is unambiguous. The practical answer does not depend on any of this:
`-ub 8` avoids rocBLAS entirely, and is also the faster of the two default paths measured here
(about 189 t/s at pp512, versus about 124 with the native library or the quantized-kernel path at
the default batch). Neither of those two figures is captured, noted 28 August: no shipped benchmark
reports a prefill near either, and what survives of that configuration is its perplexity, 11.21,
which is exact against the log. The ordering they express, `-ub 8` ahead of the default batch on
the unpatched path, is what the section rests on and is visible in the era's own comparison; the
rates are recollections of it. Why large-batch prefill is slow at all turned out to be more interesting than
Tensile tuning: measured at the exact shapes llama.cpp uses, the untuned fallback fp16 GEMM
already sustains 2.6 to 4.2 TFLOP/s, which puts the GEMM itself at roughly a tenth of prefill wall
time, while ggml's quantized matmul kernels (the MMQ path that llama.cpp selects at batch) measure
at about 285 GFLOPS on this chip against 4.2 TFLOPS for the same shape in fp16. The mechanism is
visible in the source: those kernels are built around byte-wise integer dot products (`dp4a`) that
RDNA1 does not have in hardware, gfx1013 is additionally absent from the RDNA1 macro that selects
the hand-written emulation (so it gets the slowest generic fallback), and the path-selection logic
has no RDNA1 case that would prefer the BLAS route at large batch. Two resolutions came out of
that: the macro fix from the flash-attention caveat turns on the hand-written emulation and lifts
the quantized kernels to 2.64 TFLOPS, taking the unmodified default path to about 808 t/s at
pp512; and forcing the BLAS route (`-DGGML_CUDA_FORCE_CUBLAS=ON`, with the native rocBLAS and the
f32 compute type from the next caveat) measures faster still, 936 t/s at pp512 and 760 at pp2048,
perplexity-gated. The forced-BLAS build is the fastest correct prefill path, the default MMQ
path is close behind and slightly ahead on decode (113.5 versus 109.2 tg64, measured the same
way), and Tensile tuning, the original suspect, is demoted to a minor optimization.

How much of that paragraph is captured, checked 26 August. The 285 GFLOPS and 2.64 TFLOPS are the
pair discussed above, quoted from a measurement that was not kept. The forced-BLAS figures are in
the same position: 936, 760 and 109.2 appear in no capture here, and the one re-run that tried to
reproduce them could not, because `GGML_CUDA_FORCE_CUBLAS` is a compile-time `#ifdef` in this
version and setting it in the harness did nothing, so that arm measured the default path twice
([`logs/campaign-rerun-2026-08-18/`](logs/campaign-rerun-2026-08-18/)). Only the default path's
113.5 tg64 is captured. The comparison is therefore between one measured value and three
remembered ones, and "the forced-BLAS build is the fastest correct prefill path" is the part of
this section least supported by anything shipped here.

A campaign-scale follow-on: on the wider model set the system rocBLAS fails on more shapes than the first
crash showed, aborting with `CUBLAS_STATUS_INTERNAL_ERROR` on the fa-off long-context attention
GEMMs, on the Q8_0 dequant path, and on two more models' perplexity runs. The native gfx1013 build
resolves every one of those; the symlink workaround is not sufficient for real inference, the
native library is required.

**Batched compute corrupted at longer contexts, and the cause turned out to be software.** The
perplexity gate caught this; generated text can look plausible while it happens. The symptom: with
micro-batches of 32 and up, perplexity degraded roughly 17x at context 2048 and roughly 380x at
context 4096 (with large run-to-run variance), while micro-batch 8 stayed exactly correct at every
context, and every individual operator passed correctness tests against the CPU backend at the
exact production shapes and strides. The chase went through and eliminated buffer pools, virtual
memory mapping, batch sizes, and address-placement theories before landing on the actual cause,
which is one attention matmul's precision. llama.cpp requests fp32 precision for the KQ matmul
(its own comment: "this op tends to require high floating point range") but not for the KQV
matmul that aggregates the values. On this architecture, without the matrix instructions newer
GPUs use, the batched KQV runs through rocBLAS half-precision GEMM with fp16 accumulation, and
with this model family's large activations the accumulation error over thousands of keys becomes
catastrophic, growing with context. Data-dependent, and that is why synthetic operator tests pass
while real inference corrupts. One added line requesting fp32 precision on KQV (or the existing
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` environment variable) restores exact perplexity at every
context and micro-batch tested, pool and mapping settings irrelevant: 11.0634 against Vulkan's
11.0279 at context 4096, reproduced bit-identically.

Which of the two to use is worth being precise about, because the working configuration below
already sets that environment variable and therefore does not need the patch. Measured on a build
with the patch removed, at context 4096 over two chunks: with the variable it reads 11.0631, and
with flash attention on, which is what the configuration uses, it reads 11.0521, matching the soak
number exactly. Both figures were re-measured from the tree in August 2026 and reproduce to four
decimals, as does the patch itself, since restoring the line takes the failing arm back to
11.0631.

Without the variable and without the patch the same arm is catastrophically wrong, and how wrong
is not a stable number. This section quoted 276.29 until 26 August as though it were a
property of the defect. Re-measured, the same arm gives 4199.41, and repeating it with
`GGML_CUDA_NO_VMM` unset gives 624.13, so three observations span more than an order of magnitude
while every correct arm is bit-identical. What is reproducible is that the defect destroys the
output, not by how much ([`logs/kqv-remeasure-2026-08-18/`](logs/kqv-remeasure-2026-08-18/)). So
the patch matters for anyone who does not set the variable, and every number quoted in this
document is reproducible without it.

Two things about that variability do not sit comfortably with the mechanism above, and saying so
is better than leaving them implicit. The first is that the wrongness is bimodal, not noisy.
At context 1024 two of three unpatched runs return 8.1616 and 8.1616, agreeing to four decimals,
and the third returns 708.4671; at 2048 it is 10.0103 twice and then 1347.6541. Accumulated
rounding error over thousands of keys would be expected to vary continuously between runs, not to
be reproducible to four decimals and then jump by two orders of magnitude. Something discrete is
switching, and a kernel selected differently between invocations was the obvious candidate. It has
now been checked against a dispatch trace and it is not that. Logging every rocBLAS call across
eight unpatched runs shows the same 448 calls and the same eight signatures every time, identical
between correct and wrong runs, while the perplexities come back as 10.0103, 91.3845, 172.3724 and
1057.5494. The library is asked for exactly the same work and does not always return the same
answer, so the variation lives below its interface
([`logs/kqv-dispatch-2026-08-21/`](logs/kqv-dispatch-2026-08-21/)). That also cuts against the
upstream-precision reading, since an accumulation gap is deterministic: it should give a wrong
answer stably, not four different ones from identical calls. What the trace cannot say is whether
the divergence begins inside the GEMM or in data that differed before it, because it records call
shapes and not contents. An attempt to find that first divergent tensor with `llama-eval-callback`
produced six byte-identical dumps, twice, at two prompt lengths, and that was read here for most of
a day as the instrument suppressing the defect. It was not. The rate rises steeply with the amount
of work evaluated, one failure in six runs at a single perplexity chunk against five in six at two
chunks, and the eval-callback runs were a single forward pass, less work than either. Identical
dumps from a workload with a low per-run failure probability are an ordinary outcome and not
evidence of anything. A control had been run, but it established that the defect was live in the
build, not in the workload the instrument was actually running, which is the weaker of the
two and does not license the conclusion.

Five interventions were made while chasing that phantom, and they survive it, since each ran on the
perplexity workload where the defect is common and each shows the defect persisting: disabling HIP
graph capture, synchronising after every node, refusing every operator fusion, submitting one node
per graph launch, and reading every result back to the host. None removes it. Each probe carries a
counter proving it executed, a precaution added after one of them was nearly believed without it.
Refusing fusion does shift the correct answer slightly, 10.0204 against 10.0176 on the patched build
where the defect cannot occur, which was measured instead of assumed before scoring that arm
([`logs/kqv-divergence-2026-08-21/`](logs/kqv-divergence-2026-08-21/)). Graph capture is not the
part that matters either: disabling it leaves the defect fully
present, one of six runs correct against three of six with capture on, which at that sample size
is the same thing ([`logs/kqv-divergence-2026-08-21/`](logs/kqv-divergence-2026-08-21/)). Locating
the divergence now looks unreachable by this route, since any instrument fine enough to see it
appears coarse enough to suppress it. Twelve consecutive runs at one
context sharpen this: nine distinct values, but the correct 10.0103 appears three times and a wrong
115.8450 appears twice, both exact repeats, so the defect selects between a small number of
behaviours rather than drifting ([`logs/kqv-periodicity-2026-08-21/`](logs/kqv-periodicity-2026-08-21/)).
That run also refuted a pattern noticed in the ladder, where the catastrophic result fell on the
third invocation of every rung: there is no period, and the probability attached to that pattern
after noticing it was worth nothing. It also shows the defect strikes about two runs in three at
context 2048, pooling both runs, not the one in three that three samples suggested.

The second is that "this one is not the board" is an inference and not a measurement. It rests
on the mechanism being architecture-general, which is a good argument, but the defect has never
been run on hardware other than this one. Everything measured here is consistent with an upstream
precision gap and also consistent with a gfx1013 GEMM behaving badly, and the two have not been
separated. The patch is correct either way.

Reproducing the context ladder on 21 August strengthened
that and corrected one thing. Three unpatched runs at each of 1024, 2048 and 4096 give 8.1616,
8.1616, 708.4671; 10.0103, 10.0103, 1347.6541; and 224.8379, 211.8500, 8234.2048, against patched
values of 8.1702, 10.0204 and 11.0631. The defect therefore reaches context 1024, which an earlier write-up
had described as correct, and what grows with context is how often it strikes and how
badly, not whether it can. The patched ctx 4096 value reproduces the 18 August measurement
to four decimals ([`logs/kqv-ladder-2026-08-21/`](logs/kqv-ladder-2026-08-21/)). The likeliest
reading is an upstream precision gap on architectures that take the cuBLAS path for batched
attention, surfaced by a model with large activations, and not anything specific to this board,
with the caveat above that this has not been separated from a gfx1013 GEMM behaving badly and
cannot be from one board. The practical rule while it is open: batched work needs
the precision line or the environment variable; micro-batch 8 needs nothing.

The same environment variable turned out to matter for a second, unrelated reason. The qwen3
dense models read perplexity anywhere from 18 to 31 against Vulkan's 9.10 and 7.91 without it,
varying run to run, which looked at first like more of the same accumulation problem but is not:
it is the zeroed cuBLAS result described in the next caveat, and it is a dropped output rather
than a precision loss, which is also why the number moves around. What the two share is the
cure. On this chip the f32 compute type is free, prefill and decode rates
measured identical with and without it (no matrix cores means fp32 and fp16 GEMM run at similar
rates here), so there is no reason not to set it globally, and two separate defects make it
necessary rather than merely advisable.

**A dropped GEMM result, and it took a month to find: the layer-0 value projection came back all
zeros.** This one is fixed, and on Fedora 44 it does not exist, but it was the largest correctness
defect here and the way it misled is worth keeping.

The symptom: on qwen3-8B and qwen3-14B, wikitext perplexity read 18.1 and 24.7 against Vulkan's 9.10
and 7.91 unless `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` was set. In these models every layer's value
weight is stored F16 while everything else is quantized, so those 36 matmuls, and only those, went to
cuBLAS with fp16 compute; the rest of the graph went to the quantized kernels.

What failed was the position, not the tensor: **the first of those 36 calls in each graph
execution**, never any of the other 35, and never in the first execution of a process. Counted across
twenty captured runs, 0 of 15 first executions and 49 of 60 later ones on the 8B, 5 of 8 later ones on
the 14B, against 0 of 4 with the f32 compute type and 0 of 8 on the model whose value weights are
quantized and so never take the path.

**The cause was in the half-precision conversion helpers, not the kernels.** The system rocBLAS
imports them from libgcc_s, which are correct. The native build links its own copies from Fedora 43's
ROCm compiler-rt, which read the half value from `%edi` where ROCm clang passes it in `%xmm0`.
Converting `alpha` therefore returned whatever the last code left in a register, and when that was
zero, rocBLAS's `prob.k && *prob.alpha ? prob.k : 0` handed Tensile a K=0 problem whose output is all
zeros. Replacing the two helpers with F16C instructions removes it on both models
([`logs/fp16-root-cause-2026-09-15/`](logs/fp16-root-cause-2026-09-15/)).

That one mechanism accounts for everything the earlier theories could not: the positional rule, the
dependence on a running model, the suppression under tracing, and the clean standalone reproducers.
All of them come down to what previous code happened to leave in a register.

Almost every reading along the way was a correct observation pointing at the wrong layer. Graph
capture looked implicated and was ruled out twice: with capture disabled the zero count, the positions
and the perplexity are identical to the digit, 14.1344 either way
([`logs/fp16-mechanism-2026-08-23/`](logs/fp16-mechanism-2026-08-23/)). A batch-boundary model
predicted the zero count from the micro-batch size and got three of four exactly right, which felt
like confirmation and was not: five identical runs at ubatch 512 give two, three, three and four zeros
at drifting positions ([`logs/fp16-batch-boundary-2026-08-23/`](logs/fp16-batch-boundary-2026-08-23/)).
An MMQ reading rested on `GGML_CUDA_FORCE_MMQ` being a runtime switch; it is compile-time and was off,
so the arms that supposedly showed it were the default path and carried no information. On Fedora 43
the defect never returned the same wrong value twice, five distinct perplexities from 14.7398 to
21.4429 against an f32 arm that returned 9.0975 every time
([`logs/fp16-recheck-2026-08-25/`](logs/fp16-recheck-2026-08-25/)).

A second implementation agreed. PyTorch built for gfx1013 is an independent consumer of the same
rocBLAS, and 200 cycles of half-precision matmul at four model-shaped sizes produced no zeroed result,
which moved suspicion off the silicon
([`patches/torch_fp16_zero_cross.py`](patches/torch_fp16_zero_cross.py)).

Where it strikes is worth keeping, because it is the default path and not an opt-in one: the compute
type is derived from the operand types before the environment variable is consulted, so an F16 weight
selects fp16 compute with nothing set, and the variable is the workaround instead of the trigger.
That workaround is free. Measured on the model that actually takes the path, eight rounds alternated:
prefill 233.67 t/s with the variable against 234.51 without, decode 34.58 against 34.50, Welch t of
0.37 and 0.23 ([`logs/f32-cost-2026-08-21/`](logs/f32-cost-2026-08-21/)).

**A separate llama.cpp version warning.** During this work, inference on a current llama.cpp
master was found numerically wrong on this board at every setting, with superficially coherent
text, while an older build (2da6686) is exact. Re-measured from the tree in August 2026 as an
A/B/A within one boot: with the flag at upstream behaviour perplexity reads 168.5483 and 168.1212,
with it forced false it reads 11.1910, and restoring the change returns 11.1910 bit-identically.
An overnight bisect landed on a single commit: `c7d8722`, "ggml-cuda : restore prop.integrated on
HIP builds" ([PR #24233](https://github.com/ggml-org/llama.cpp/pull/24233), landed 2026-07-16). The
bisect's own output was not kept, noted 26 August, so which commits it tested cannot be checked
here. What is captured is the effect at that code line, not the search that found it: the
A/B/A above is in [`logs/integrated-remeasure-2026-08-18/`](logs/integrated-remeasure-2026-08-18/)
with all four values, and the line the commit restores is quoted from upstream source below. It
makes the backend treat this APU as an integrated GPU and use host-memory buffer paths, which are
exactly the territory this board handles badly. Forcing `integrated = false` restores exact
perplexity at micro-batch 8.

Checked against llama.cpp master in August 2026, the situation is clearer than it first looked, and
it has not changed on the HIP side:

```c
#if defined(GGML_USE_HIP)
        info.devices[id].integrated = prop.integrated;
#else
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
#endif
```

Upstream has already accepted this reasoning, just not for HIP. The issue their comment cites,
[#15034](https://github.com/ggml-org/llama.cpp/issues/15034), is "Broken/no Gemma 3n output on CUDA
(Nvidia Jetson Orin Nano)", which is the same failure on the same kind of device: an integrated GPU
where trusting the flag selects host-memory paths and the output comes out wrong. The counter-patch
here is not a special case for one board, it is the decision upstream already made for CUDA applied
to the branch that still trusts the flag. Until upstream changes it, builds after that commit need
the one-line counter-patch, or use a build from around
2da6686, and in either case verify with the perplexity gate and not by reading output.

### llama.cpp: ROCm vs Vulkan, same build, same boot configuration

![ROCm vs Vulkan](figures/fig-rocm-vs-vulkan.png)

Every HIP row below was measured under one configuration: llama.cpp master with the three
patches, flash attention on at the default micro-batch, the native gfx1013 rocBLAS on the library
path, `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, and `HSA_ENABLE_SDMA=0`, with prefill and decode in
the same `llama-bench` invocation. The Vulkan column is the same llama.cpp build on the same
boot. Each rate is one invocation, which for the noisier models carries more uncertainty than the
figure printed beside it suggests: the across-invocation spread on the 8B is about three times
`llama-bench`'s own error bar, so ratios here are reliable to about a few percent, not to
the second decimal. Every row passed a wikitext perplexity gate against Vulkan on the same command before its
rate was recorded; tokens/s:

| model | HIP pp512 | VK pp512 | HIP tg64 | VK tg64 | decode share |
|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 805.6 | 1842.2 | 113.5 | 211.0 | 54 percent |
| qwen3-8B Q8_0 | 241.0 | 401.1 | 39.2 | 39.1 | 100 percent |
| deepseek-r1-14B Q4_K_M | 95.4 | 199.0 | 20.3 | 34.5 | 59 percent |
| qwen3-14B Q4_K_M | 97.4 | 202.8 | 21.5 | 34.2 | 63 percent |
| qwen3.6-35B-A3B MoE IQ2_M | 287.6 | 455.4 | 34.3 | 86.5 | 40 percent |

One qualification on the 8B row, added after the fact. Its decode figure was for a while described
here as parity with Vulkan, on the strength of a single invocation pair reading 39.2 against 39.1.
That model's decode rate turns out to vary more across invocations than within them: eleven
independent measurements give mean 37.34 with sd 1.20. Against a Vulkan figure near 39.1 that puts
ROCm at about 95 percent rather than at parity, which is still far closer than any other model
here. The pairwise rows below are single invocations and carry the same caveat.

Notes on the spread: the 8B at Q8_0 decodes closest to Vulkan, the small model and the
Q4_K 14Bs sit between half and two thirds, and the MoE at 40 percent, so the decode gap is not
one number, it depends on quantization and architecture.

### A newer and larger model: Qwen3.8-27B

Added after the campaign above, both to check that the configuration holds on a model released
later than all of this work and to find where a 16 GiB board runs out. Qwen3.8-27B at UD-IQ3_XXS
is 11.09 GiB of weights for 27.3 billion parameters, the largest dense model tried here:

| backend | pp128 | pp512 | tg128 | perplexity (2 chunks, ctx 2048) |
|---|---|---|---|---|
| ROCm | 62.6 | 69.2 | 7.84 | 6.2487 |
| Vulkan | 93.1 | 97.9 | 17.18 | 6.2651 |

It works, unmodified, under the same recipe: no faults in dmesg, and the two backends agree on
perplexity to within 0.3 percent, which is the useful check since a backend that produced fast
nonsense would look identical in the rate columns. Vulkan keeps prefill by 1.4x and decode by
2.2x, so the decode share here, 46 percent, sits with the other Q4_K-class models and not with the
Q8_0 8B that comes closest.

Context ceiling on ROCm, prompt processing at increasing depth:

| context | prefill t/s |
|---|---|
| 4096 | 59.2 |
| 8192 | 49.1 |
| 16384 | 41.4 |
| 32768 | fails to run |

16384 tokens is the working ceiling for prompt processing on this model, and the failure at 32768 is
memory, not any of the defects documented here: it is `failed to create context with
model`, an allocation refusal before any kernel runs, because 11.09 GiB of weights plus a 32k KV
cache does not fit alongside the system in 16 GiB shared. Prefill decays gently to that point,
keeping 70 percent of its 4k rate at 16k.

Generation does not reach as far. Priming the cache to 16128 tokens and then decoding runs out of
memory on this model even though processing a 16384-token prompt is fine, so its decode ceiling is
8192, where it produces 7.0 t/s. The per-model table further down separates the two, and the
distinction matters: a context length a model can ingest is not necessarily one it can generate
from. Logs in [`logs/qwen38-2026-08-17/`](logs/qwen38-2026-08-17/).

One measurement note, since it cost time here. The first attempts at this model failed in two
different ways, once inside `ggml_cuda_mul_mat_cublas` and once at model load, and neither was a
defect: a second process still held the GPU. At 11.09 GiB on a 16 GiB shared board there is no
headroom for an overlapping run, and the failures it produces are loud enough to look like the
ones documented elsewhere in this file. Checked afterwards on an idle board, this model runs
whether or not `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` is set, at the same rate, so it carries no F16
weights on the path that variable protects.

That pattern has an explanation, and it is bandwidth. A streaming-read kernel measures 432 GB/s,
which is 402 GiB/s, as the achievable memory bandwidth on this board
([`patches/membw.cpp`](patches/membw.cpp), three shipped runs in
[`logs/membw-2026-08-19/`](logs/membw-2026-08-19/)). Every share below is computed against that
number, so its own reproducibility bounds them: the three runs agree to 0.12 percent, but the same
probe on 25 August, on a later kernel, returned 430.3 GB/s
([`logs/rocm-only-verify-2026-08-25/`](logs/rocm-only-verify-2026-08-25/)). The steadiness is
within a session; between sessions expect about 430 to 432, so treat the shares as good to a
percentage point and not to their last digit. Decode reads the weights once per token, so each rate
above implies a read bandwidth, and comparing that against the ceiling is more informative than
comparing backends:

| model | weights | ROCm implied read rate | share of ceiling | Vulkan implied | share |
|---|---|---|---|---|---|
| qwen3-8B Q8_0 | 8.24 GiB | 323 GiB/s | 80 percent | 323 GiB/s | 80 percent |
| deepseek-r1-14B Q4_K | 8.37 | 170 | 42 percent | 289 | 72 percent |
| qwen2.5-1.5B Q4_K | 1.04 | 118 | 29 percent | 220 | 55 percent |

Each row is the model's file size times its decode rate from the table above, against the 402
GiB/s ceiling. The 8B therefore sits near the memory wall on both backends, so they come close
there, not because ROCm suddenly got better. The 80 percent in that row derives from a
single decode invocation; on the pooled eleven it is nearer 76, which does not change the reading.
On the Q4_K models ROCm uses about 42 percent of the available bandwidth where Vulkan reaches
about 72, and that gap, not the arithmetic, is where the remaining decode difference lives.

The 8B ROCm row here read 299 GiB/s and 74 percent until it was rechecked, which did not
follow from the decode rate beside it; recomputed from the campaign's own numbers it is 323 and
80, close to Vulkan rather than merely in the same region. The 1.5B row moves the same way for
the same reason, from 123 to 118, because its 113.5 t/s is the mixed-invocation figure rather
than the faster decode-only one.

The 35B MoE is excluded from this table because it activates a subset of
its experts per token, so the file size is not the bytes read. The 1.5B prefill was reported at 936 t/s with
the forced-BLAS build, from a reading that was not kept (the caveat above); the table shows the
unmodified default path, which is measured. The 35B
MoE decodes faster than either dense 14B on both backends.

**The prefill gap is between two quantized-matmul implementations, not between libraries.** The
obvious guess is that ROCm dequantizes to fp16 and calls rocBLAS, so that rocBLAS bounds it. That
is not what happens: a full pp512 run on the small model under `ROCBLAS_LAYER=1` logs **zero**
rocBLAS calls at 797 t/s. llama.cpp's HIP backend selects its own quantized kernels here and never
enters a BLAS, which the quantized-kernel caveat above already establishes from the other
direction, by a reported 2.64 TFLOPS for those kernels after the RDNA1 macro fix, not itself
captured, and a measured 808 t/s out of them.

The two accounts agree numerically, which is the useful part. Taking the arithmetic prefill
implies (roughly twice the non-embedding parameters per token, about 1.55 billion of the model's
1.78) over the measured rate, ROCm's 807.9 t/s is about 2.5 TFLOP/s, against the 2.64 TFLOPS those
kernels are reported to benchmark at directly, so the agreement is between one measured rate and one
reported one. Vulkan's 1844.3
t/s is about 5.7 TFLOP/s, above the board's best
dense fp16 GEMM of 4.6, which a dequantize-then-GEMM path could not reach even in principle and
is consistent with Vulkan also multiplying against quantized weights. So both backends run the
same kind of kernel and Vulkan's is roughly twice as fast on RDNA1. Closing that is kernel work in
llama.cpp's HIP backend; the forced-BLAS build above is the other lever, reported at 936 t/s from a
reading that was not kept. The FLOP
figures count matmul arithmetic only and ignore attention, so treat them as a ratio instead of a
benchmark.

rocBLAS still bounds whatever does go through it, which is a separate matter that decides other
models. Its throughput on layer-shaped problems runs well below its square-matrix rate:

| shape (m x n x k) | what it is | GFLOP/s |
|---|---|---|
| 2048 x 2048 x 2048 | square reference | 4181 |
| 4096 x 4096 x 4096 | square reference | 4248 |
| 1536 x 512 x 1536 | attention projection | 2637 |
| 8960 x 512 x 1536 | feed-forward up | 2838 |
| 1536 x 512 x 8960 | feed-forward down | 3917 |

![rocBLAS GEMM throughput by shape](figures/fig-gemm-shapes.png)

None of those five numbers is captured, checked 26 August. They are whole numbers, which a search of the
logs cannot separate from version numbers and line counts, and searching for each returns
nothing; the single apparent hit for 2838 is a GDB process id. So this table and the figure under
it, which agree with each other exactly, agree about a measurement that was not kept. The shapes
themselves and the conclusion drawn from them are unaffected in kind, since the same ratio shows up
in the end-to-end prefill rates that are captured, but the numbers here should be read as a record
of what was measured, not as something a reader can check.

About a third of the square-matrix throughput goes on the tall-thin shapes, from an untuned
Tensile build. Which models pay it is worth checking instead of assuming, and the check is cheap:
under `ROCBLAS_LAYER=1` the Q8_0 8B calls `rocblas_sgemm` where the Q4_K 1.5B calls nothing at
all.

Two measurement notes, both learned by re-measuring rather than assuming. First, decode rate
depends on what else the same invocation ran: the 1.5B reads 113.5 t/s when prefill tests precede
it in one process and 117.6 to 118.8 t/s in a decode-only invocation, reproducibly, which is why
the table and the depth ladder below (a decode-only run) differ at depth zero. Second, the choice
of rocBLAS is not free even where the system library works: deepseek-r1-14B decodes at 21.7 t/s
against the system library and 20.3 against the native one, so the native library costs about
seven percent there while being the only one that runs the other models at all. An earlier
revision of this table mixed the two libraries across rows; these numbers do not.

A qwen3.5-9B file failed to load on both backends and on CPU identically (a GGUF metadata
mismatch, `qwen35.rope.dimension_sections` expected length 4, between an old conversion and this
llama.cpp revision; nothing to do with the board) and is excluded.

The gates behind those rows, rerun at eight chunks rather than two so the error bars are worth
quoting (same model, same command, same boot, wikitext, context 2048, flash attention on):

| model | ROCm/HIP | Vulkan |
|---|---|---|
| qwen3-8B Q8_0 | 7.3503 +/- 0.232 | 7.3792 +/- 0.233 |
| qwen3-14B Q4_K_M | 6.3970 +/- 0.193 | 6.4548 +/- 0.195 |
| deepseek-r1-14B Q4_K_M | 6.0013 +/- 0.173 | 6.0416 +/- 0.175 |
| qwen3.6-35B-A3B MoE IQ2_M | 5.1887 +/- 0.134 | 5.2041 +/- 0.134 |

Every pair agrees far inside one standard error
([`logs/gates-2026-08-14/`](logs/gates-2026-08-14/)). One detail is consistent enough to mention: the
HIP value is slightly lower than the Vulkan one on all four models, by 0.2 to 0.9 percent. With
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` the HIP path accumulates in fp32 where the Vulkan path does
not, so a small systematic accuracy edge is the expected direction, though four models on one
board is thin evidence for the claim.

The historical comparison from before the fixes, kept because the text below references its
era (qwen2.5-1.5B, `-fa off -ub 8` on build 2da6686, the conservative configuration that needed
no patches): pp512 182.2 versus 1844.2 (10 percent of Vulkan), pp2048 170.6 versus 1711.5,
tg128 106.8 versus 210.7 (51 percent). None of those six values was kept, as the paragraph after this
one sets out: they are recollections of that era and not readings. A side finding from it: the
small-microbatch
prefill was five times faster than the broken large-batch path (182 versus 38 t/s), so avoiding
the defective fp16 GEMM closed most of what looked like a 50x prefill deficit, leaving about
10x, and the fixes above closed most of the rest (to about 2x). Larger models in that era were
text-verified only: the MoE decoded at 31.6 and deepseek-r1-14B at 19.5 to 19.7; both now have
gated numbers in the table above, measurably better on the fixed stack.

None of the figures in this paragraph has a capture, noted 26 August. Searching the logs for
182.2, 170.6, 106.8, 31.6, 1844.2, 1711.5 and 210.7, excluding the log READMEs and the per-tensor
dumps, returns
nothing for any of them, and the two apparent matches are a GDB backtrace and unrelated SDMA probe
output. The build they were taken on, 2da6686 with `-fa off -ub 8`, is not the stack this
repository now runs, so they cannot be produced again either; re-running that command today would
measure something else. They are kept because the era they describe is referenced below and the
ratios are the point, but they are recollections of it, not readings a reader can check.

### Decode at context depth

Generation speed with the KV cache primed to the stated depth (tg64), qwen2.5-1.5B. The pre-fix
`-fa off` column (where the non-FA attention pays the full quadratic cost) is kept for contrast:

| depth | ROCm `-fa off` (pre-fix) | ROCm `-fa on` (fixed) | Vulkan (fa on) | fixed HIP as share of Vulkan |
|---|---|---|---|---|
| 0 | 106.8 | 117.6 | 211.0 | 56 percent |
| 4096 | 74.2 | 103.4 | 178.5 | 58 percent |
| 8192 | 54.6 | 96.1 | 163.8 | 59 percent |
| 16384 | 35.4 | 84.4 | 143.1 | 59 percent |
| 24576 | | 74.2 | 126.5 | 59 percent |
| 30720 | | 68.2 | 116.5 | 59 percent |

Provenance of that table. The Vulkan column is the 2026-08-12 campaign, `f_q15_tg.log`, and all six
values match it. The fixed column is `recipe_q15_ladder.log` of 13 August: depth 0 distinguishes the
two files, since the table says 117.6, the 13 August file reads 117.59 and the 12 August one reads
118.80. The two columns are a day apart instead of one boot. The 12 August run of the same ladder
does exist and agrees with the
13 August one to within 0.03 at every depth, worth knowing, though it is not what the table
quotes.

The pre-fix `-fa off` column has a weaker footing than either, noted 26 August when the search was
widened to one decimal place. Its four values are 106.8, 74.2, 54.6 and 35.4, and no
capture in this repository holds a `-fa off` ladder at these depths. Excluding the log READMEs and the per-tensor dumps,
54.6 appears in no file at all, and the apparent matches for the other three are a GDB
backtrace and unrelated SDMA probe output, not benchmarks. The column is kept because the
shape it shows is the point, that non-flash attention pays a quadratic cost the fix removes, and
that shape is independently visible in the fixed column beside it. But the four numbers themselves
should be read as recollections of the pre-fix era rather than as figures a reader can check, and
the `-fa off` arm has not been rerun since.

That column is worse still: 106.8, 74.2, 54.6 and 35.4 appear in no capture in this
repository. The file named for that arm, `logs/bench-fixed-2026-08/a_q15_faoff.log`, aborted with a
ROCm error and recorded no rates at all, and the pre-fix campaign's own 1.5B ladder
(`logs/bench-2026-08/hot_q15_d*.log`) gives 106.21, 103.41 and 97.66, which are different numbers.
Both campaigns were searched. The column is left in place because the contrast it draws is
described elsewhere from data that does survive, but it should be read as unsourced.

![Decode vs depth](figures/fig-decode-vs-depth.png)

How far the context can be pushed, which had not been measured here before, on the fixed stack
with the full recipe (decode of 32 tokens with the cache primed to just under the stated size):

| context | qwen2.5-1.5B | deepseek-r1-14B |
|---|---|---|
| 8192 | 96.3 t/s | 12.6 t/s |
| 16384 | 85.5 | 10.2 (see the note below) |
| 32768 | 67.7 | |
| 65536 | 47.7 | |
| 98304 | 35.6 | |
| 131072 | 28.3 (8 tokens, not 32) | |

The small model runs the full 131072-token context at 28.3 t/s, and 64k at 47.7. Two earlier
attempts at the longest row were reported here as failures; they were not. Each had been
given twenty and then forty minutes, and priming that many tokens takes about fifty-five, after
which it completes normally and decodes at a usable rate. Disabling graph capture changes nothing
there (29.8 against 28.3, within the spread), so the graph limit described further down does not
bite at this context
([`logs/context-2026-08-14/`](logs/context-2026-08-14/)).

The 14B entry needs care, because the obvious reading of it is wrong. It is not a memory ceiling
between 8k and 16k. The model runs a 16384-token context perfectly well, returning a wikitext perplexity of 4.5489, and it decodes at a
primed depth of 12000 at 10.2 t/s. What fails is narrower and has nothing to do with context size:
**HIP graph instantiation**. With graph capture enabled, a primed-depth decode aborts at depth
12000 (it is fine at 8000 and 10000) with `hipGraphInstantiate` failing inside
`ggml_cuda_graph_evaluate_and_capture`; with `GGML_CUDA_DISABLE_GRAPHS=1` the identical run
completes. The same failure appears at small micro-batches on the 8B, where a micro-batch of 128
aborts in the same call. So this is a graph-size limit and not a memory limit, and setting
`GGML_CUDA_DISABLE_GRAPHS=1` is the workaround when a configuration trips it.

**How much of that is actually backed?** Less than its specificity suggests, and it matters, because
two other pages disable graph capture on the strength of it.

Captured: a `llama-bench` primed-depth run of deepseek-r1-14B at a 16384-token context returning
`rc=134 FAILED` ([`logs/context-2026-08-14/`](logs/context-2026-08-14/)), and a perplexity run of the
same model at the same context returning 4.5489
([`logs/historical-sources/inv39/`](logs/historical-sources/inv39/)).

Not captured anywhere: the abort at depth 12000, the two passing depths of 8000 and 10000, the name
`hipGraphInstantiate`, the enclosing function, the micro-batch-128 failure on the 8B, and any run
of the failing configuration with the flag set. The two figures above are also not a before and
after: one is a primed-depth decode and the other a perplexity pass, which are different workloads,
so they cannot show a flag fixing a failure.

The model differs between statements as well. Here the entry is deepseek-r1-14B, which is what the
ladder column and both captures are.
[`logs/context-ceilings-2026-08-17/`](logs/context-ceilings-2026-08-17/) says the limit "fails on
that model" of qwen3-14B, and the harness that produced its rows prints the same attribution, but
nothing was measured on qwen3-14B to support it; the flag was set there prophylactically.

None of this shows the graph limit is not real. Disabling graph capture is cheap and the rows that
used it completed. What it shows is that a mechanism stated to the level of a specific API call and
three specific depths rests, in this repository, on one failing run and one unrelated success.

One run in roughly twelve at this depth died in a way nothing else here has: SIGBUS, on the host,
with no GPU fault logged. The backtrace puts the faulting store inside ROCr's own AQL queue,
`rocr::AMD::AqlQueue::StoreRelaxed`, reached from an ordinary `hipblasSetStream` by way of
`hipStreamQuery`, `submitMarker` and `dispatchBarrierPacket`. SIGBUS, not SIGSEGV points at
a mapping that exists but cannot be backed, which is an uncomfortable thing to see on a board that
needed a driver fix for stale translations, though nothing here connects the two beyond the
resemblance. A deliberate ten-run repeat of the same configuration did not reproduce it, so this is
an observation instead of a reproducer, and it is recorded with the full backtrace in case someone
else sees the same signature
([`logs/loose-ends-2026-08-18/graphflag/sigbus-backtrace.txt`](logs/loose-ends-2026-08-18/graphflag/sigbus-backtrace.txt)).

The workaround is not neutral for throughput, so check before reaching for it
globally. On the 8B at a primed depth of 16128 it changes decode by more than ten percent. The
direction depends on how the comparison is run, which is covered below: blocked arms make it look
15 percent slower and counterbalanced arms make it about 13 percent faster, and only the second
design can separate the flag from drift. At depth 0 the difference nearly vanishes either way, so
whatever it is, it belongs to decoding with a large cache resident. Set the flag where a
configuration needs it, not everywhere.

**Where each model actually runs out.** The ladder above stops where it was stopped, not where
the board refuses, so this is the same measurement pushed to failure on four models: prime the
cache to just under the stated context, then decode 8 tokens. Every cell below is that same
measurement, which matters because decode rate depends on how many tokens are generated (the
1.5B at 8192 reads 96.3 t/s over 32 tokens and 84.8 over 8):

| context | 1.5B Q4_K (1.04 GiB) | 8B Q8_0 (8.24 GiB) | 14B Q4_K (8.63 GiB) | 27B IQ3_XXS (11.09 GiB) |
|---|---|---|---|---|
| 8192 | 84.8 | 22.6 to 23.9 | 12.6 | 7.0 |
| 16384 | 73.6 | 16.2 to 18.7 | 7.3 | fails |
| 32768 | 62.7 | fails | fails | fails |
| 131072 | 27.6 | | | |
| 262144 | fails | | | |

Decode at depth ceilings at 131072 tokens on the 1.5B, 16384 on the 8B and the 14B, and
8192 on the 27B. Every failure is memory, and there are two distinct kinds. The 14B and the 27B at
32768 fail cleanly with `failed to create context`, an allocation refused before any kernel runs.
The others get further and then abort in the backend, with dmesg showing `amdgpu: SVM mapping
failed, exceeds resident system memory limit`. Neither is one of the defects documented here, and
dmesg recorded zero GPU faults across the whole campaign, which is a much weaker statement than it
reads and has needed correcting twice. The ROCr runtime reports memory access faults the kernel log
omits entirely, and one such fault appeared in twenty-eight repeats of the deepest 8B measurement
([`logs/deep-decode-faults-2026-08-20/`](logs/deep-decode-faults-2026-08-20/)). That run was then
described as having dmesg clean throughout, which was not an observation at all: the board rebooted,
and a dmesg check after a reboot reads the buffer of the boot that followed. The same instrument has
a second blind spot found later, on 21 August: its ring buffer wraps. A fault hunt flooding the log
with SVM messages pushed the boot-time 40-CU unlock line out of `dmesg` entirely, while the
persistent journal still held it, so a count of zero there can mean the evidence scrolled away
rather than never existed. Asked of the persistent
journal instead, the same period held five fatal GPU resets across the twenty boots retained on
20 August, a snapshot and not a reproducible rate
([`logs/journal-retro-2026-08-20/`](logs/journal-retro-2026-08-20/)). Logs in
[`logs/context-ceilings-2026-08-17/`](logs/context-ceilings-2026-08-17/).

One condition behind that table needs stating, because it was implicit and it matters. Every row
was taken with the model mmapped, which is `llama-bench`'s default. Loading without mmap lowers the
ceiling: the 8B runs a primed depth of 14336 and aborts at 16128, ten of ten, inside graph capture.
Sampling memory through both arms at that depth shows why, and the difference is not subtle. Without
mmap the run peaks at 14594 MiB used with 601 MiB available; with mmap it peaks at 11532 MiB with
3663 MiB available, because page-cache-backed weights can be reclaimed under pressure and anonymous
ones cannot. On a board sharing 14 GiB between host and GPU that margin decides it
between running and aborting. This was found by accident, when two arms of a variance experiment
used the non-mmap path and failed every time
([`logs/nommap-ceiling-2026-08-21/`](logs/nommap-ceiling-2026-08-21/)).

Prefill reaches further than decode on the largest model, which is worth separating because the
two get conflated easily. The 27B processes a 16384-token prompt at 41.4 t/s, but priming the
cache to that depth and then generating runs out of memory: generation needs the full cache
resident at once alongside 11.09 GiB of weights, where prompt processing does not have to hold as
much live simultaneously. The table recorded the 27B at 16384 as working on
the strength of the prefill number, which was the wrong measurement for a column about decode.

Three measurement notes. The 131072 row took 55 minutes to prime and reproduces the earlier
measurement closely (27.6 against 28.3 on a different boot), so that figure is stable. The 8B is
given as ranges because it is much
noisier than anything else measured here, and that turned out to be worth chasing instead of noting.
Ten consecutive runs at that row's depth in one boot, `d16128`, give 16.21 to 18.71, mean 17.66, a
coefficient of variation of 4.3 percent, with temperature between 56 and 68 C and free memory flat. Clock is
stated quantitatively, not as "reached 1500 MHz", because it is the most obvious
candidate and the sampled data rules it out and not merely failing to implicate it. Sampling
the shader clock once a second through all ten runs, 2722 samples in
[`logs/loose-ends-2026-08-18/variance/`](logs/loose-ends-2026-08-18/variance/) as `clk1.txt` to
`clk10.txt`, the time spent at the governor's 1500 MHz cap is 84.1 percent overall and between
83.2 and 85.1 percent in every individual run, the rest being the 1000 MHz idle state around the
ramp. A spread of 1.9 percentage
points in clock residency across runs whose rates differ by 15 percent means the clock is not what
moves the rate. The spread looks intrinsic rather than environmental.

A note on how precise any of these rates are, which applies to every single-invocation figure in
this document. `llama-bench` prints an error bar computed across its repetitions inside one
invocation. That is not the uncertainty of the number. For the 8B at `tg64`, the median reported
bar across every run on record is 0.43, while the spread across independent invocations is 1.20,
about three times larger. So a printed `39.20 +/- 0.17` is a good deal less precise than it looks,
and differences of a few percent between rates measured in different sessions should not be read
as changes.

That was learned by chasing one such difference for an afternoon. The 8B decode rate measured 36.30
on the current stack, with the same afternoon's other runs between 36.40 and 38.34, against 39.20
in the campaign of 2026-08-12, with prefill flat and
Vulkan flat, which looked like a decode-only regression on the ROCm side and was treated as one.
Two hypotheses were tested and both were wrong. The map-side runlist flush was ruled out by
toggling the parameter live. The debug instrumentation that the working tree had acquired since
the campaign, including `getenv()` calls in the memory pool's hot path, was ruled out by cloning a
clean tree at the campaign commit, applying only the three shipped patches, and measuring both
binaries in one boot: the clean build reads 36.40 and 36.66 against the instrumented build's 38.15
and 38.34, the opposite of what the hypothesis predicted. Pooling all eleven independent
invocations gives mean 37.34 and sd 1.20, which puts the campaign's 39.20 1.6 standard deviations
above the mean. There was no regression, and the small model reproduces the campaign closely in
both builds (807.60 and 807.40 against 806). Data in
[`logs/clean-build-2026-08-18/`](logs/clean-build-2026-08-18/).

The spread is also specific to decode at depth, not to this model or this board: prefill spreads are
0.3 percent on the 1.5B over 43 soak rounds and 0.5 to 1.1 percent on the three large models over
26 each.

The cause is not established, and the attempt to test the leading candidate produced a lesson
about controls instead of an answer. Memory bandwidth looked like the explanation, since decode at
depth on the 8B runs at about 80 percent of the board's measured bandwidth, the highest utilisation
of anything tested (the share is the model's file size times its
decode rate against the 402 GiB/s ceiling, derived in the table earlier in this section, and it is a
property of the model, not of the depth it was measured at), leaving little headroom to
absorb whatever else touches memory. A model at 29
percent utilisation was 6.7 times steadier, which fit. A third model at an intermediate 46 percent
came out the most variable of the three, at 11.7 percent, which fits nothing
([`logs/loose-ends-2026-08-18/bandwidth-variance/`](logs/loose-ends-2026-08-18/bandwidth-variance/),
ten runs, linked here from 26 August after the directory holding this result was found to be
reachable from nothing).

That third point turned out to be worthless, for an instructive reason. The 14B needs
`GGML_CUDA_DISABLE_GRAPHS=1` at a primed depth of 16128, since HIP graph instantiation fails on it
beyond 12000, and the other two models ran without that flag. The flag had been treated as inert.
Measured directly on the 8B, it is not inert: ten runs per arm at the same depth, disabling graph
capture appeared to cost about 15 percent of throughput and to multiply the spread by two and a
half. That measurement was itself blocked, not counterbalanced, and repeating it properly
reversed the sign, which is set out immediately below.

| 8B at depth 16128, blocked design | mean t/s | sd | coefficient of variation | range |
|---|---|---|---|---|
| graph capture enabled | 17.80 | 1.16 | 6.5 percent | 15.87 to 19.37 |
| graph capture disabled | 15.19 | 2.45 | 16.1 percent | 10.03 to 19.18 |

Those two arms ran as blocks, ten of one and then ten of the other, which is the same design flaw
this section is about. Repeating the comparison in ABBA order, so that anything drifting over the
run cancels, reverses the result:

| ABBA round | capture off | capture on | difference |
|---|---|---|---|
| 1 | 17.70 | 15.78 | +1.92 |
| 2 | 19.78 | 17.89 | +1.89 |
| 3 | 19.69 | 17.12 | +2.57 |

Mean difference +2.13 t/s, sd 0.38, the same sign in three rounds of three: disabling graph capture
is about 13 percent *faster* at this depth. The blocked figures above are left in place to show what
a design that presents the arms in blocks instead of interleaved produces: the opposite sign
([`logs/counterbalanced-2026-08-18/`](logs/counterbalanced-2026-08-18/)).

The flag is therefore not inert, which is what the three-model comparison needed to know. The 14B's
11.7 percent cannot be separated from what the flag alone produces, so that comparison is not
evidence in either direction.

The test that would settle the bandwidth question compares models at a depth where all of them
keep graph capture on, which means at or below 8192. It was run twice, and the two runs disagree,
which is itself the most useful thing to come out of it. Presented in blocks, ten runs of one
model before any of the next, the coefficients of variation come out at 2.8, 5.1 and 6.6 percent
for 29, 46 and 80 percent of the measured 432 GB/s bandwidth ceiling, a monotone ordering that looks like support.
Presented with the model order rotated each round, they come out at 0.7, 6.0 and 4.6 percent, and
the middle model is again the most variable.

The bandwidth explanation is not supported by the trustworthy design, and it is not refuted
either, since a three-point comparison at these sample sizes cannot resolve differences this
small. What is established is narrower: on this board, at these
depths, an effect of a few percent cannot be measured reliably by six to ten runs of an arm, and
two careful designs of the same comparison can return differences of opposite sign. That is a
limit on what any conclusion in this document about small differences can mean, and it applies to
every one of them
([`logs/counterbalanced-2026-08-18/`](logs/counterbalanced-2026-08-18/)).

One earlier reading, 12.94, sits 6.3 standard deviations below this boot's mean and is excluded
from the range above, for two reasons. It was taken interactively and never captured
to a log, so unlike every other figure here it cannot be re-examined; and it was measured minutes
before a run that exhausted memory and aborted, so the board was plausibly already in the state
that run then hit. Prior heavy work on its own does not reproduce it: measured deliberately, the
same model gives 18.00 and 18.51 on a clean board, 17.88 and 18.78 immediately after a 10.7 GiB MoE
has run, and 19.23 and 18.26 after dropping caches, all overlapping. Data in
[`logs/loose-ends-2026-08-18/variance/`](logs/loose-ends-2026-08-18/variance/) and
[`logs/context-ceilings-2026-08-17/8b-repeats/`](logs/context-ceilings-2026-08-17/8b-repeats/). And
a reading of 2.13 t/s for the 8B at 8192 was discarded: it ran immediately after a
three-and-a-half hour job that had exhausted memory, and re-measured on an idle board the same
configuration gives 22.8 and 22.6.

The share is essentially constant at depth: the fixed flash-attention path loses ground to
Vulkan at the same rate Vulkan loses ground to itself. This ladder was measured twice, on
different boots and (the second time) under the full library recipe, and reproduced to within a
percent at every depth except zero, where the decode-only versus mixed-invocation effect noted
above accounts for the spread.

Measured with `-fa on` before the garble was understood, this table shows ROCm nearly flat with
depth. The rates were plausible even then; only the outputs were garbage, and with the macro fix the
flash-attention path delivers both. Decode at depth 16384 went from 35 to 84 t/s, and ROCm's
share of Vulkan at depth from about a quarter to about three fifths. What was written here before
the fix, that a reliable flash-attention path was the most valuable missing piece for inference
at depth, held up in the most literal way.

### The allocation-reuse defect, a reproducer, and the flush that fixes it

The load-time aperture fault that gated large models above turned out to be one face of a deeper
defect. It now has two reproducers, a mechanism traced to within microseconds, and a fix that
holds across reboots and through an eight-hour soak. The story runs in two stages, because the
first fix worked for the obvious cases and left a residual that took a second pass to explain.

The mechanism: on this board the compute TLB invalidation that should follow `hipFree` does not
take effect, so when `hipMalloc` reuses a virtual address range, the GPU can keep translating
through the previous mapping. The mechanism was identified and a fix demonstrated by the
bc250-rocm-working project (GabriWar). His instrumentation found that
`gmc_v10_0_flush_gpu_tlb_pasid()` looks for the owning VMID in a register that on gfx10 under
hardware scheduling is never written, so the flush matches nothing: 20 of 20 flushes hit zero
VMIDs in his measurements, reproduced on separate hardware. He also confirmed the consequence
directly, finding that for 15 of 15 failing pages across 3 runs the physical address the GPU used
was exactly what that virtual address had pointed to in an earlier generation. His workaround
asks the firmware scheduler to rebuild the runlist on unmap, a cycle that does invalidate. That fix was
ported here as a runtime-switchable module parameter (`amdgpu.bc250_flush_by_runlist`,
[`scripts/apply_runlist_flush.py`](scripts/apply_runlist_flush.py)); at first it showed no effect,
because none of the probes here churned allocations.

The reproducer that changed that ([`patches/seq_probe.c`](patches/seq_probe.c)): a heavy dispatch
(a few seconds of arithmetic), `hipFree`, `hipMalloc` of any size, another dispatch. Generation
two then either takes a GPU page fault or silently drops a large prefix of its stores, roughly
524 thousand elements at the 8.4M size, close to the 525,308 of the original silent
corruption capture in Observation 1. Exact-multiple sizes failed the same way, single dispatches
did not fail here, and a light fill kernel does not trigger it, which is how earlier reuse probes missed
it. With the runlist flush enabled the same sequence runs clean. What this repository can show for
that is a verification boot with the parameter set from the kernel command line: three runs of the
reproducer clean and a flush-off control faulting
([`logs/bench-2026-08/runlist-verify/`](logs/bench-2026-08/runlist-verify/)), plus a fresh-boot
A/B/A on the SVM map side, three runs clean with the flush on and two faulting with it off
([`logs/svm-flush-2026-08/`](logs/svm-flush-2026-08/)).

An earlier and larger same-boot series, an interleaved A/B in which the flush-off arm faulted five
of five and the flush-on arm was clean five of five, was quoted here for weeks. Its raw logs were
never retained: the only surviving record is a line in the working notes this investigation was
assembled from, which is prose rather than an artifact. The claim is left out of the count above,
not restated, on the same principle applied to figures elsewhere here, that a result whose
evidence cannot be produced is not a result. Nothing else in this section depends on it, since the
two retained A/Bs point the same way.

The practical effects reach further than the reproducer. With the flush on, the 10.7 GiB MoE
loads went from zero of three to three of three (decoding at 31 to 32 t/s), the 14B loads three
of three, and `llama-perplexity`, which had failed on every prior attempt, completed: **HIP
perplexity 11.2071 +/- 0.675 against Vulkan 11.1859 +/- 0.676** on the same wikitext-2 chunks,
agreement well inside the error bars, which suggests the compute path is numerically sound end
to end.

For a while a residual remained: an operator-benchmark sweep with rapid alloc/dispatch/free
churn still faulted within seconds on every attempt even with the flush on, at a slightly
different case each time, which read as a workload-shaped race the unmap hook could not close.
Tracing closed it. Running the churn reproducer under ftrace with the amdgpu VM tracepoints and
freezing the buffer at the fault showed, in every captured instance, the same sequence: the
faulting pages are mapped, later unmapped, sit unmapped for seconds while other work runs, and
are then mapped again, and the GPU faults on them **44 to 122 microseconds after the new valid
PTE is written**. That timing rules out use-after-free through a stale mapping and points the
other way: while a range sits unmapped, in-flight work walking neighboring addresses lets the
translation cache hold the *invalid* entry, and on remap the hardware keeps using that
stale-invalid entry because the invalidation that should follow the map is the same broken PASID
sweep. The reason the unmap hook never helped is that this traffic does not go through the ioctl
path the hook covers at all: function profiling during a faulting run showed the allocations
flowing through the KFD SVM paths (`svm_range_validate_and_map`, `svm_range_unmap_from_gpus`),
which issue their own (ineffective) TLB flush. Extending the same runlist-rebuild to those two
sites, with the parameter widened to a runtime-writable bitmask (bit 1 unmap, bit 2 map),
resolves it: on one boot, with the bits toggled live, the churn sweep runs to completion with
zero faults with the hooks on (previously it aborted within about ten seconds), faults within
seconds with them off, and runs clean again when re-enabled. The cost is not measurable in
these tests: perplexity bit-identical, prefill, decode, and 14B load times unchanged.

The fix also holds under sustained load: an eight-hour soak alternating prefill, a perplexity
gate, and the churn sweep ran 43 rounds with zero faults and one distinct perplexity value (the
soak entry under "What still fails, measured" has the details).

The fix is not specific to this kernel, and neither is the defect. Every kernel in the ladder was
booted and given the same churn sweep:

| kernel | map-side flush | churn sweep |
|---|---|---|
| 7.1.5 | present | 84 runs, all completed, no faults |
| 6.19.14 | present | 2 runs, both completed, no faults |
| 7.1.2 | absent | 2 runs, both faulted |
| 7.0.13 | absent | 2 runs, both faulted |
| 6.18.16 | absent | 2 runs, neither completed nor faulted, both stalled to a 25 minute timeout |
| 6.18.9 | absent | 2 runs, both stalled the same way |
| 6.18.9 | present | 1 run, completed in 590 seconds, no faults |

The defect spans the whole range from the original work to the current kernel, and nothing
upstream has fixed it in the meantime. Worth noting for anyone reproducing: the symptom differs by
era. On the 7.x kernels the sweep dies with a GPU memory access fault within seconds; on the 6.x
ones it never finishes, with no fault logged at all. Someone grepping for a fault signature
on a 6.x kernel would conclude the board was fine.

The last row is why the stall can be attributed to the same defect and not left as a separate
unknown. A stall is only the absence of completion, so by itself it is equally consistent with a
sweep that needs longer on those kernels. Building the map-side hook into 6.18.9 and rebooting it,
with the same module source and the same boot arguments otherwise, turns a run that did not finish
in 1500 seconds into one that finishes in 590. A healthy sweep takes 590 seconds there, 591 on
7.1.5 and 592 on 6.19.14, so the timeout carried about 2.5 times the necessary headroom on the very
kernel that stalled. Per-kernel configuration and logs for every row, including the 6.18.9 pair,
are in [`logs/ladder-churn-2026-08-16/`](logs/ladder-churn-2026-08-16/).

One trap for anyone checking these configurations: an absent `amdgpu.bc250_cc_write_mode` on the
kernel command line does not mean the unlock is off. It is a module parameter, and
`/etc/modprobe.d/bc250-40cu.conf` sets it to 3 and is baked into every initramfs, so it applies
whenever the command line is silent (a command-line value does override it). The same trap is
described from the other direction further down, where it would invalidate an A/B that toggles only
the boot argument. The SIMD count is the check that cannot be fooled, and it has to be read across
all KFD topology nodes, not the first one, because node 0 is the CPU and reports 0:

    grep -h simd_count /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u

Which half does the work decides how large the change has to be.
Testing the bits separately on one boot: the churn sweep needs the **map** side
(map only, twice, ran to completion with no faults, the same as both bits set; unmap only faults
within seconds; neither bit segfaults in 17 seconds), while the older sequence reproducer is
satisfied by either bit alone (three of three with bits 1, 2, or 3, and three of three corrupt
with neither, reporting two bad generations;
[`logs/svm-flush-bits-2026-08-13/`](logs/svm-flush-bits-2026-08-13/)). PyTorch's varying-size churn
also passes with the
map bit alone, 300 iterations twice over, which is recorded in working notes and not in a log kept
here: the retained A/B covers both bits set and the unmap bit alone, not
the map bit alone. The map-side rebuild covers every case reproduced
here and the unmap-side one is redundant, which makes the minimal fix a single hook rather than
two.

That result holds across a reboot too, which is the stronger form of the test. On a fresh boot
that had never faulted, with the parameter arriving from the kernel command line, not a
runtime write, the churn sweep ran to completion twice (about ten minutes each, the suite
printing its own pass line); disabling the map side then produced a fault in 16 seconds and
again in 104 seconds; re-enabling it produced another full clean run. Judged by log content and not
exit status, that is three complete runs with the fix and zero without.

Across everything run this way, the tally is **84 churn sweeps with the map side enabled, all 84
completing with no faults, against 12 with it disabled, of which 2 completed and 10 faulted**. So
the disabled arm is not deterministic: it fails most of the time and occasionally survives, which is
not the same as faulting within seconds.

The figure aggregates across the day, not one run, so here is where it comes from. Enabled:
43 sweeps in the eight-hour soak, 26 in the large-model soak, 3 on the fresh verification boot,
5 in the same-boot series above, 2 from the map-only bit attribution, and the remainder from the
verification and reproduce runs. Disabled: 2 on the fresh boot, 5 in the same-boot series, 3 under
CPU load, 1 unmap-only and 1 with neither bit. The disabled arm sums exactly to the 12 quoted; the
enabled arm is reconstructible to 79 from the logs shipped here, with the balance in runs whose
per-sweep output was not kept separately.

One alternative explanation has now been tested and not merely acknowledged. If the runlist
rebuild helped only by slowing things down, then slowing the machine some other way should help
too. It does not: with the map side disabled and eight busy CPU threads loading the machine, the
sweep still faulted in two runs of three, the same rate as with the machine idle. So "any delay
hides it" is not supported, which is a point in favour of the rebuild doing something specific to
translations rather than merely perturbing timing. None of that proves the mechanism, and the
mechanism remains inferred.

What that evidence does and does not establish, stated plainly. The A/B/A is done both within a
boot (parameter toggled live, so boot-to-boot variance cannot explain it) and across a boot, and
the effect size is large. One board, two workloads. The result does not prove the
negative-caching mechanism directly; the mechanism is inferred from the timing and from which
kernel paths the traffic uses, and an alternative reading, that the extra rebuilds merely shift
timing enough to hide a race, is weakened by the CPU-load test above, since an unrelated slowdown
does not substitute for the rebuild, but it is not formally excluded. The intermediate attempt is worth
recording for the same reason: hooking the map ioctl alone changed nothing (three of three, one
of two, three of three), which is what sent the investigation to the function profile that found
the SVM paths.

The same extension resolves the other residual this document listed, PyTorch's allocation churn
(see the PyTorch section): the pattern that used to fault reliably now runs clean with the SVM
bits on and crashes with them off, on the same boot. The residual is not a workload-shaped
race after all, and not something the hook could not see. It was the hook watching the wrong
door.

### PyTorch, and a note on allocation discipline

The official `torch 2.9.1+rocm6.4` wheel with the native gfx1013 rocBLAS grafted in, matmul with
preallocated buffers, thirty iterations per size, all checked and correct:

| N | GFLOP/s |
|---|---|
| 1024 | about 1210 |
| 2048 | about 3050 |
| 4096 | about 4270 |
| 8192 | about 4550 |

One discipline still defines what PyTorch is for on this board, and one that used to has been
lifted. The lifted one is allocation. Loops that allocate and free GPU tensors every iteration
used to fault after 20 to 40 iterations, and the original unmap-only flush did not help, which
was read here as torch's caching allocator reusing addresses without ever unmapping. The SVM-side
flush above changes that: with it enabled, 300 iterations of deliberately varying matrix sizes
(sizes chosen at random from 512 to 5120, which defeats the allocator's block cache and forces
real map and unmap traffic) run clean, four times over in the retained log, while the same loop
with only the unmap bit set faults every time
([`logs/svm-flush-2026-08/torch_aba.log`](logs/svm-flush-2026-08/torch_aba.log)). The notes of the
day record the failures landing near iterations 100 and 200. The retained runs print no progress
line at all before aborting, which does not establish when they faulted: the harness prints every
fiftieth iteration and the process dies on SIGABRT, so an unflushed buffer looks exactly like an
early fault. What the retained log supports is that the unmap-only arm fails and the both-bits arm
does not; the iteration numbers are from the notes and are not reproducible here. The
same-size loop, it turns out, no longer faults either way, so the old 20-to-40 figure belongs to
the earlier configuration. Preallocated outputs (`torch.mm(a, b, out=c)`) remain the faster
pattern, but they are no longer a correctness requirement. Second, kernel coverage:
the official wheel ships no gfx1013 elementwise kernels, so only the rocBLAS-backed matmul path
runs on the GPU; tensor creation and activations must happen on the CPU, and a full autograd
training step fails on the missing kernels (`invalid device function`).

Running each operation in isolation, so that one failure does not mask the others, shows how
narrow the working surface is, and where the boundary actually falls. Two of the three columns are
captured and the middle one is not, noted 26 August: the stock-wheel abort is in
[`logs/torch-pristine-2026-08-20/pristine-probe.txt`](logs/torch-pristine-2026-08-20/pristine-probe.txt)
and the built-for-gfx1013 column in
[`logs/torch-probe-2026-08-19/probe.out`](logs/torch-probe-2026-08-19/probe.out), while the
per-operation errors for the grafted wheel come from a run whose output was not kept. The score it
produced, 1 of 11, is stated on the torch-pristine page; the per-row error strings are not
recoverable from anything shipped.

| operation | path | stock wheel | stock plus native gfx1013 kernels | built for gfx1013 |
|---|---|---|---|---|
| host to device and back | copy | correct | correct | correct |
| `matmul`, fp32 | library | aborts | `HIPBLAS_STATUS_INTERNAL_ERROR` | correct |
| `addmm`, fp32 | library | aborts | `HIPBLAS_STATUS_INTERNAL_ERROR` | correct |
| `matmul`, fp16 | library | aborts | `invalid device function` | correct |
| `randn` on device, `zeros`, `fill_` | torch kernel | aborts | `invalid device function` | correct |
| add, multiply, `relu`, `softmax`, `sum`, `.half()` | torch kernel | aborts | `invalid device function` | correct |

One of eleven, one of eleven, eleven of eleven. A stock wheel aborts at the first library-dispatched
operation instead of failing it, because it ships no Tensile library for gfx1013 or
for any gfx101x; only the host-to-device copy, which reaches no library, survives. Copying the
native gfx1013 Tensile kernels into it stops the abort without buying anything: the library calls
then fail rather than aborting, and the score stays at one. Building for the architecture is what
works.

This table took two attempts to get right, and the trap is easy to repeat. Both earlier readings were
taken on a virtualenv with 56 gfx1013 Tensile kernel files grafted into it, so neither described a
stock wheel. The second attempt was the more careful mistake: the wheel's `librocblas.so` was checked
and is genuinely stock, which seemed to settle it, but the kernels rocBLAS actually loads sit in a
directory beside the library and nobody looked there. Verifying the library file is not verifying the
library ([`logs/torch-pristine-2026-08-20/`](logs/torch-pristine-2026-08-20/)).

One loose end is left visible. That older grafted venv scores three of eleven
where a fresh wheel given the same 56 kernel files scores one, and the two match on torch version,
on `librocblas.so` by md5, on the kernel files by content, and on the entire listing of
`torch/lib`. Both figures are stable across alternated repeats and installing the missing numpy
changes nothing. The older environment carries a large number of extra packages from a source
build, one of which presumably matters. What makes the difference was not identified.

Two more traps around this table, both easy to fall into. Half-precision matmul does **not** go
through hipBLASLt: with `ROCBLAS_LAYER=1`
it logs `rocblas_gemm_ex` with an `f32_r` compute type, so it is rocBLAS like the fp32 path, and
it works once rocBLAS has the right code objects. And an fp16 matmul written the obvious way,
`a.half() @ b.half()` on tensors already on the device, fails on the *conversion* and not the
multiply, because `.half()` is a torch elementwise kernel. Building the operands as fp16 on the
host and copying them over, so no device conversion is needed, the multiply itself succeeds.
The same trap applies to checking the result: `.float()` on a device tensor is also a torch
kernel, so verification has to copy to the host first and convert there.

On a pristine wheel the failure is not `invalid device function` but an abort inside rocBLAS,
which prints the list of `TensileLibrary_lazy_*.dat` files it does have, none of them gfx1013,
and dumps core. That listing is the clearest single symptom of the problem.

Two further notes for anyone reproducing this. The wheel's arch list contains no gfx101x target
at all, so no `HSA_OVERRIDE_GFX_VERSION` setting rescues it: overriding to 10.1.0 still gives
`invalid device function` because there is no gfx1010 code object to load, and overriding to a
target the wheel does have, 10.3.0 or 11.0.0, loads code objects for a different architecture
generation and faults the GPU outright. Also, `torch.cuda.is_available()` returns true and the
device is named correctly as `AMD BC-250` with `gfx1013:xnack-`, which makes the wheel look
supported right up until the first kernel launch.

The missing code objects are the whole of the problem, which is worth establishing before
spending hours on a build. A single elementwise kernel compiled for gfx1013 and exposed through a
plain C entry point runs correctly on tensors that the stock wheel allocated, in the wheel's own
process, through its caching allocator: exact agreement with the CPU reference, zero error, at
both 1 million and 16.7 million elements over four launches. The wheel's own equivalent kernel,
called on those same tensors a few lines later in the same process, still fails with `invalid
device function`. Driver, runtime, allocator and hardware are all fine for this work; only the
shipped architecture coverage is not. The probe is in
[`patches/bc250_ext.hip`](patches/bc250_ext.hip) with its harness in
[`patches/torch_ctypes_test.py`](patches/torch_ctypes_test.py).

Two notes on that probe. Torch's own extension builder does not work on this system as shipped:
it detects `ROCM_HOME` as `/usr` and passes `-isystem /usr/include`, which breaks `#include_next`
and fails on `stdlib.h` and `math.h` before any GPU code is reached. Building the kernel directly
with `hipcc --offload-arch=gfx1013` avoids it entirely. And keep the verification on the CPU: an
innocent-looking assertion like `(out == 0).all()` is itself a torch kernel and fails on this
wheel, which reads as the probe having failed when it has not.

### PyTorch built for gfx1013

Building torch 2.9.1 from source with `PYTORCH_ROCM_ARCH=gfx1013` lifts the restriction
completely. The same eleven-case probe, on the same board, against the built wheel with the
native gfx1013 rocBLAS on `LD_LIBRARY_PATH`:

| operation | wheel, native rocBLAS grafted in | built for gfx1013 |
|---|---|---|
| host to device and back | correct | correct |
| `matmul`, `addmm`, fp32 | correct | correct |
| `matmul`, fp16 | correct | correct |
| `randn`, `zeros`, `fill_` | `invalid device function` | correct |
| add, multiply, `relu`, `softmax`, `sum`, `.half()` | `invalid device function` | correct |

Eleven of eleven, against four of eleven for the best the wheel can be made to do and three of
eleven for the wheel as shipped. Training works, which is the case that was previously
impossible: a three-layer network over 50 Adam steps runs entirely on the GPU, the loss falls from
2.30348 to 0.00048, and every step agrees with the same run on the CPU to 1.799e-05. The GPU run takes
0.26 and 0.25 seconds over two runs against 20.38 and 20.33 on the twelve CPU threads
([`logs/torch-train-2026-08-19/`](logs/torch-train-2026-08-19/)). Do not read those two as a
speedup: this torch has neither MKL nor MKLDNN, so its CPU path is about forty times slower than
numpy on scipy-openblas, and a CPU reference worth quoting uses the latter
([`logs/torch-rocblas-bench-2026-09-24/`](logs/torch-rocblas-bench-2026-09-24/)). Re-checked on 22 August against
the current configuration, kernel 7.1.8 with the navi12 microcode and `amdgpu.gpu_recovery=0`, the
GPU side is identical to every digit, 0.26 seconds with the same losses and the same 1.799e-05
agreement, while the CPU reference came in at 22.14 seconds, which is host-side variation
([`logs/torch-recheck-2026-08-22/`](logs/torch-recheck-2026-08-22/)). Both CPU figures here have a log behind them.

The final parameters differ from the CPU run by 9.3e-3, which is drift rather than a defect, and
the distinction is worth checking instead of assuming. One thing to know before trusting any of
this section, and how it was resolved. Checked on 25 August, the board no longer carried the build
that produced these numbers: the only Python environment on it held a stock `torch 2.9.1+rocm6.4`
whose architecture list has eleven entries and no `gfx1013`. An attempt to re-run the divergence
probe during a review pass is what surfaced it, aborting with `invalid device function`, which is
the stock wheel's signature and is documented elsewhere in this file as exactly what a wheel without
gfx1013 kernels does. As of 24 September the board carries one again, `2.9.1a0+gitd38164a` with
`torch.cuda.get_arch_list()` returning `['gfx1013']`, in `~/torchbench-venv`, built from the tree
that was left in place. The benchmarks in
[`logs/torch-rocblas-bench-2026-09-24/`](logs/torch-rocblas-bench-2026-09-24/) run against it. The
figures in this section are still the August ones and have not been re-measured on it.

Of the figures in the next two sentences only the
9.3e-3 has a surviving log: the rest were read from the probe output at the time and not retained,
so they are quoted from working notes, not from an artifact here, and the
probe is kept so the measurement can be repeated. A single forward and backward pass, with no
optimiser state, agrees to 1.1e-8 absolute and 3.4e-7 relative, which is fp32 rounding. Across the
run the difference grows monotonically, 8.7e-6 at step 1 to 9.3e-3 at step 50, which is what Adam
does with an initial difference since it renormalises the step size instead of damping it. The GPU
run repeated is bit-identical. Probes in
[`patches/pytorch/torch_train.py`](patches/pytorch/torch_train.py) and
[`torch_train_diverge.py`](patches/pytorch/torch_train_diverge.py).

One requirement carries over from the rest of this document: the GEMM paths need the native
gfx1013 rocBLAS. Built against the system rocBLAS, whose 56 gfx1013 entries are all symlinks to
the gfx1010 ones, every GEMM fails with `HIPBLAS_STATUS_INTERNAL_ERROR` while the elementwise
kernels are fine. Pointing `LD_LIBRARY_PATH` at the native build fixes all three GEMM cases,
fp16 included.

#### Building it against distribution ROCm

The build assumes AMD's installer layout under `/opt/rocm` throughout, and Fedora's ROCm packaging
differs in enough places to stop it seven times. None of the failures name their cause, and the
first does not fail at all:

| what the build assumes | what Fedora has | how it presents |
|---|---|---|
| `FindHIP.cmake` under `${ROCM_PATH}/lib/cmake/hip` | `lib64/cmake/hip` | no error, a CPU-only wheel |
| `torch_cpu` compiled with `-DUSE_ROCM` | it is not | upstream's own ROCm guard does not fire, CUDA-only symbols reach the compiler |
| edits to the tree persist | `build_amd.py` rewrites it every run | fixes silently reverted, including inside comments |
| hipBLASLt and Composable Kernel present | one packaged, one absent | configure aborts |
| ROCm clang at `${ROCM_PATH}/llvm/bin` | `/usr/lib64/rocm/llvm/bin` | every HIP object fails to compile |
| `rocm-core/rocm_version.h` present | absent | see below |
| that header included unconditionally | absent | one late compile error after most of the build |

The version header is the one to watch. With `rocm-core/rocm_version.h` missing, the build falls
back to `hip/hip_version.h`, whose patch field is a HIP build number instead of a ROCm patch
level. The arithmetic then yields `ROCM_VERSION=103884` from ROCm 6.4, which satisfies guards like
`ROCM_VERSION >= 70000` and enables code the installed ROCm does not have. It fails much later, at
an undeclared FP4 type in a sparse kernel, pointing at PyTorch's source rather than at the version
arithmetic. Systems with the rocm-core package never take this path.

Three source changes cover it, in
[`patches/pytorch/0001-fedora-rocm-build.patch`](patches/pytorch/0001-fedora-rocm-build.patch),
with the environment in
[`scripts/build_pytorch_gfx1013.sh`](scripts/build_pytorch_gfx1013.sh). Two working notes. Patch
the `cuda` sources, not the generated `hip` copies, since the latter are regenerated on
every build: the compiler names the generated file, which is the wrong file to edit. And write the
fixes so the rewriter leaves them alone; an edit mentioning the vendor token gets rewritten in
place, comments included, which turned one of these fixes into a self-contradictory sentence
before it was reworded. Both fixes were confirmed by re-running the rewriter and rebuilding from
the cleaned tree.

Two habits are worth carrying if reproducing this. Check `USE_ROCM:BOOL` in `build/CMakeCache.txt`
before letting a build run, since the first wheel here built, installed and imported cleanly while
containing no GPU code at all, and only `torch.cuda.get_arch_list()` returning an empty list
revealed it. And do not pipe the build through `tail`; the first attempt here did, which discarded
the `FAILED:` line and left only trailing compiler warnings to read.

### Other things that work, checked once each

Three capabilities that this document had asserted or assumed without ever measuring them on the
finished stack:

- **`llama-server`** serves correctly. Booted on the 1.5B with the working recipe it answers
  `/health` in 8 seconds and returns a coherent completion over HTTP at 114.6 tokens/s by its own
  timings, close to the 113 to 119 the command-line tools give for the same model. An earlier
  revision recorded 12 seconds and 97.9 tokens/s from a single run; both were re-measured and
  captured
  ("The capital of France is Paris. The capital of Italy is Rome..."). Everything else in this
  document was measured through the command-line tools, so the server path had never been tried.
- **ROCm and Vulkan run concurrently without interfering.** Two `llama-bench` processes on the
  same model at the same time, one on each backend, measured 806.1 t/s prefill on ROCm and
  1843.0 on Vulkan, each within a percent of what it does alone. Worth knowing on a board
  whose display is driven by the same silicon.
- **PyTorch and FP64 still behave after everything.** Re-measured and captured: the matmul path
  agrees with a CPU reference to a relative error of 1.89e-06 at N=1024 rising to 5.88e-06 at
  N=8192, which is ordinary fp32 accumulation over larger sums, and a 50-iteration sustained run
  at one size is bit-identical to its own first result while holding about 3.0 TFLOP/s. The FP64
  DGEMM probe returns 456.3 GFLOP/s with no wrong results
  ([`logs/once-checked-2026-08-17/`](logs/once-checked-2026-08-17/)). A previous revision quoted
  4.3e-07 here, which no test in this repository reproduces; the figures above replace it.

### ROCm-only capabilities

Things the Vulkan path cannot offer on this board, now usable:

- **Double precision.** rocBLAS DGEMM at N=2048 runs at about 456 GFLOP/s steady state, all
  results correct. That was long quoted here as 95 percent of a 480 GFLOP/s FP64 peak, a spec figure.
  Double precision now has a measured rate: `v_fma_f64` costs 80.8 cycles against `v_fma_f32`'s 5.32,
  one fifteenth of the fp32 instruction rate, and the chain sustains 0.43 TFLOP/s
  ([`logs/alu-rates-recheck-2026-09-25/`](logs/alu-rates-recheck-2026-09-25/)). The old denominator,
  480, is one sixteenth of 7.68 TFLOP/s, the FP32 rate the clock and
  lane count suggest, and this board's measured FP32 rate is 6.52 TFLOP/s
  ([`logs/alu-rates-2026-09-19/`](logs/alu-rates-2026-09-19/)). So 456 is 95 percent of a
  spec-derived ceiling, not of a measured one, and it is not comparable to the percentages of
  measured peak quoted elsewhere in this document. One sixteenth of the measured FP32 rate would be
  296, which DGEMM exceeds, so the ratio does not hold in that direction either. What is not in
  doubt is that the work runs and the results are correct. Vulkan compute has no practical
  double-precision path on this board, so for scientific workloads this capability is exclusive to
  ROCm.
- **PyTorch, including training.** There is no Vulkan PyTorch backend, so anything torch-shaped is
  ROCm-only here. Built for gfx1013 it is not limited to matmul offload: every operation tried
  runs, and a full training loop matches the CPU to 1.799e-05 per step on the losses (the section above).
- **Custom HIP C++ kernels.** Single-source GPU programming with the CUDA-style toolchain: the
  probes in this repo are just that, and a small Mandelbrot renderer
  ([`patches/mandelbrot.cpp`](patches/mandelbrot.cpp), FP64 iteration on the GPU) is included as a
  small example. An FP64 Jacobi stencil (2048x2048, five-point) runs two thousand back-to-back GPU
  sweeps with no fault, at 6.85 ms per sweep and 2.4 GFLOP/s when built with the unoptimised line
  its source documents, and at 0.19 ms per sweep and 87.7 GFLOP/s from the same source with `-O2`,
  a 36-fold difference to keep in mind before reading either number as a hardware figure.
  Both builds produce identical results. Two cautions on what the probe prints: its "effective
  GB/s" counts logical accesses, so the optimised build's 877 exceeds the board's measured 432 GB/s
  DRAM ceiling and should not be read as memory throughput, a stencil re-reading its neighbours
  from cache; and its convergence check does not pass at two thousand sweeps, which is the method and not the hardware, since the error falls monotonically with sweep count (0.95, 0.87, 0.65
  at 2 thousand, 20 thousand and 200 thousand) and Jacobi on a grid this wide needs far more
  ([`logs/once-checked-2026-08-17/`](logs/once-checked-2026-08-17/)).
- **Retrieval as a worked example.** Cosine-similarity search over one million 384-dimensional
  document embeddings, resident on the GPU, ran at 928 queries per second through the
  PyTorch matmul path, agreeing with a CPU reference to 2.1e-07. The top-k step ran on the CPU
  because this was measured on the stock wheel, whose only working GPU path is the
  library-dispatched matmul; on a torch built for gfx1013 that constraint is gone and the whole
  operation can stay on the GPU.

### What still fails, measured

- **Large-model loads faulted until the runlist flush.** The load-time host-to-device staging fault (the
  same aperture violation documented in the inference section) becomes likelier the more bytes are
  staged, and a faulted load degrades the boot, so subsequent loads tend to fault too until a
  reboot. On a clean boot the picture is much better than the size trend first suggested:

| model (file size) | plain | unified memory | mmap on | SDMA on | plain + runlist flush |
|---|---|---|---|---|---|
| qwen2.5-1.5B (1.0 GiB) | 3/3, and about 15/15 across the day | | | | |
| deepseek-r1-14B (8.4 GiB) | 3/3 clean boot; repeated faults on boots where an earlier process had faulted | 3/3 | 3/3 | 0/3 (hang) | 3/3, incl. on a previously faulted boot |
| qwen3.6-35B MoE (10.7 GiB) | 0/3 | 1/3 | | | 3/3 (and 1/1 on the verification boot) |

That table is a tally kept as the loads were run, and the per-attempt output behind it was not,
noted 27 August. Two of its cells, the MoE's 1/3 under unified memory and the 1/1 on the
verification boot, appear in no shipped log at all; the rest are countable only in the sense that
logs elsewhere here show loads of those models succeeding and failing, not that any file records
these denominators. Read it as what was observed across that day, not as something a reader
can recount.

  A related control: decode itself runs fine with SDMA enabled (105.7 t/s in a one-off tg64, which
  is quoted from a session whose output was not kept), but
  every model load with SDMA on hung to its timeout, so `HSA_ENABLE_SDMA=0` was required in
  practice for as long as this section describes. That workaround is no longer required, and the rest of this
  entry is the behaviour of the wrong microcode, not of the board: substituting navi12's
  SDMA firmware fixes it outright, as set out further down. The narrowing below stands, and stops
  one layer short of the cause. A dedicated retest on the fixed stack (after the campaign, with the
  native rocBLAS and
  all patches) confirmed it is untouched by everything that turned out to be software: with
  `HSA_ENABLE_SDMA=1` every load hung to its timeout (three small-model attempts, a perplexity
  run, and two 14B attempts at 900 seconds each), silently, with nothing in dmesg, while the same
  14B loads take 18 to 24 seconds with SDMA off.

  A bare `hipMemcpy` probe ([`patches/sdma_probe.c`](patches/sdma_probe.c), one watchdog per copy)
  then bracketed the boundary, and it is not about bulk at all. The threshold is exact: with SDMA
  enabled a copy of **16384 bytes completes and 16385 bytes never returns**, with nothing logged.

  Runtime tracing (`AMD_LOG_LEVEL=4`) shows what changes at that byte, and it is a path switch
  inside ROCclr rather than anything about buffer sizes. At 16384 the log reads `Unpinned write
  path`, then `memcpy stg buf`, then `Blit staging H2D copy`, followed by an ordinary kernel
  dispatch: the host copy goes into a staging buffer and a **blit compute kernel** moves it, so the
  SDMA engine is never involved. At 16385 the same call instead reports `HSA Async Copy staged
  H2D`, queries the copy engines (`free_engine mask 0x3`) and issues `HSA Async Copy on
  copy_engine=0x1` with a completion signal. That signal never fires.

  Varying everything else around it leaves the boundary alone
  ([`patches/sdma_angles.c`](patches/sdma_angles.c)):

| variant | result |
|---|---|
| pageable host to device, 16384 | completes |
| pageable host to device, 16385 | hangs |
| **pinned** host to device, 16385 and 1 MiB | hangs |
| device to host, 16385 | hangs |
| asynchronous on an explicit stream, 16385 | hangs |
| device to device, 16385 and 1 MiB | completes |
| `hipMemset`, 16385 and 1 MiB | completes |

  Pinning the host memory does not help, which rules out the bounce-buffer reading an earlier
  revision of this section gave: pinned memory needs no staging and still hangs, because above the
  threshold it takes the same async-copy path. The device-to-device and memset rows are not
  counter-examples either, since tracing shows both are serviced by blit compute kernels
  (`grid=[10240, 1, 1]`) and never reach SDMA.

  The workaround works below ROCclr. With `HSA_ENABLE_SDMA=0` the runtime
  makes the identical `HSA Async Copy on copy_engine=0x1` call for a 1 MiB copy, with identical
  engine masks, and it completes. The failing component is not the ROCclr path selection; it is
  what services that copy underneath it.

  The path switch can also be moved, which confirms the reading from the other side. ROCclr exposes
  `GPU_FORCE_BLIT_COPY_SIZE`, a size in kilobytes below which it keeps using the blit kernel. With
  SDMA left enabled, a 1024 KB copy hangs at every setting up to 1023 and completes at 1024 and
  above, so the boundary tracks the knob exactly. Set large enough it makes the whole stack work
  with SDMA on: a model that otherwise never finishes loading runs at 119.3 t/s and returns
  perplexity 8.9442, matching the reference. Measured decode-only against `HSA_ENABLE_SDMA=0` over
  three repetitions each, the two are equivalent (117.6 and 118.0 against 118.1 and 119.7). None of
  those six figures was kept, noted 26 August: the directory ships the knob boundary table and the
  traces, and no benchmark or perplexity output from that run. The boundary is what this is cited
  for and it is captured; the throughputs are not.

  It is specifically that knob, and not staging generally. ROCclr also exposes
  `GPU_STAGING_BUFFER_SIZE`, which sets the size of the staging buffer the blit path copies
  through, and it might plausibly move the same boundary. It does not: a 1 MiB copy hangs at 4, at
  64 and at 1024, exactly as it does unset, while the blit-copy knob moves the boundary in the same
  session. So what decides the outcome is which path the runtime picks, not how much staging memory
  that path is given, which is the distinction the tracing already implied and this tests directly.

  `HSA_ENABLE_SDMA=0` remains the recommendation anyway, because it has no ceiling to get wrong:
  the knob only helps for copies smaller than whatever value is set, and a single larger one
  falls back to SDMA and hangs. The knob's value here is as evidence, since forcing the blit path
  by a second, independent mechanism produces the same result
  ([`logs/sdma-interrupt-2026-08-17/blit-knob/`](logs/sdma-interrupt-2026-08-17/blit-knob/)). For
  contrast, the same probe with SDMA disabled walks 4 KiB to 2 GiB without
  complaint. At 2 GiB, in the sweep this repository ships
  ([`logs/sdma-sizes-2026-08-19/blit-recheck-2026-08-25/`](logs/sdma-sizes-2026-08-19/blit-recheck-2026-08-25/),
  ten repeats per size, best of), it reaches 149.59 GB/s from pageable host memory and 12.96 ms
  from pinned, which is about 165.7 GB/s. The figure quoted elsewhere in this document is the
  pageable one. The roughly 150 GB/s cited for large transfers is the plain H2D column, 150.89,
  151.05 and 149.59 at 512 MiB, 1 GiB and 2 GiB, and pinned is the faster path, not that
  number's source. The engines are present in the KFD topology, two of them with eight
  queues each and firmware 52,
  and the kernel logs no ring-test failure at init; work handed to them never completes.
  This paragraph concluded for weeks that the defect was board-genuine. It was not: substituting
  the navi12 SDMA microcode fixes it outright, and everything above describes the behaviour of the
  wrong firmware, not of the silicon. The firmware version reported here, 52, is the clue
  that was sitting in plain sight, and it is the same number the blob's own header carries:
  reading both files on 27 August, the board's `cyan_skillfish2_sdma.bin` declares
  `ucode_version` 0x34, which is 52, against navi12's 0x2c. They are the same SDMA 5.0 format and
  the same 33792 bytes, so the version field is the only header difference besides the checksum.
  The payloads are not close, though: 17988 of those 33792 bytes differ, everything from the
  offset the header gives as the start of the microcode. The substitution does not talk the
  board out of a version check, it hands it a different program of the same generation, and why
  the board's own build fails remains open
  ([`logs/sdma-firmware-2026-08-19/identity-2026-08-26/`](logs/sdma-firmware-2026-08-19/identity-2026-08-26/)).

  Those two halves have now been joined up, using the SDMA trap instrumentation from
  [GabriWar/bc250-rocm-working](https://github.com/GabriWar/bc250-rocm-working)
  (`patches/bc250-sdma-trap-instrumentation.patch` in that repository, not this one), which logs
  inside the trap IRQ handler where
  the interrupt vector has already been dispatched, so a line appearing means the interrupt really
  arrived. Built into a 6.18.16 module and booted, with the threshold reproducing there exactly:

  - **The interrupt path works.** 31 SDMA trap interrupts were dispatched during boot, all on
    instance 0, against a logging budget of 64 that was never exhausted.
  - **Neither side of the threshold produces one.** The count is 31 before running the probe and
    31 after. The 16384-byte copy completes without an interrupt, which is what the staging
    reading predicts since the engine is never involved, and the 16385-byte copy hangs without one.
  - **The user queue is created.** During the hang the process holds a KFD queue of type 1, the
    SDMA type, at 1 MiB. The work is submitted and nothing comes back.
  - **The hypothesis their patch singles out does not hold here.** It proposes that the console
    firmware leaves interrupt-handler rings 1 and 2 alive, so an interrupt routed there vanishes
    silently. On this board both are cleanly zeroed, `base=0x00000000 cntl=0x00000000`, and the two
    "Fence fallback timer expired" lines their patch predicts on every boot do not appear at all.

  So this is not "SDMA is broken". The engine runs, its interrupt reaches the driver for
  kernel-submitted work, the handler rings are configured as Linux expects, and the user queue is
  created. Putting that together with the trace above, the failing step is narrow: ROCclr asks the
  HSA runtime for an async copy on a copy engine, the runtime routes it to SDMA, and the completion
  signal never fires and no trap interrupt is dispatched.

  The queue itself never moves either, which narrows it further, though establishing that took two
  tries. Sampling the KFD queue descriptors during a hanging copy, the SDMA queue is present
  throughout and its descriptor is byte-identical in every sample. The first attempt at a control
  compared this against sampling during a real compute workload, which changes substantially
  between samples, and that comparison is not valid: below the copy threshold ROCclr uses a blit
  compute kernel and creates no SDMA queue at all, so the two arms were never watching the same
  queue, and no matched control exists on this board because no SDMA copy ever works. What does
  establish the point is decoding the descriptor rather than counting changed lines. Against
  `struct v10_sdma_mqd` from the kernel headers, the ring is fully configured, base address,
  doorbell offset and read-pointer writeback address all programmed, and the write pointer is
  zero. Nothing was ever placed in the ring, which points at the submission end, not at the
  engine doing the work and failing to signal it
  ([`logs/sdma-onebyte-2026-08-18/`](logs/sdma-onebyte-2026-08-18/); the superseded line-counting
  comparison is kept in [`logs/loose-ends-2026-08-18/sdma-mqd/`](logs/loose-ends-2026-08-18/sdma-mqd/)).
  Everything on either side of that step
  works, including the same request when the runtime services it without SDMA. That is one board
  and one instrumented boot, and it does not name a cause; it removes one hypothesis and says where
  the next one has to look
  ([`logs/sdma-interrupt-2026-08-17/`](logs/sdma-interrupt-2026-08-17/), traces in its
  `angles/` subdirectory).

  When a large model does load it runs at full speed, so this reads as a load-time problem, not a
  runtime one. The last column is what helped: the load fault looks like the allocation-reuse
  defect (the section above), and with the runlist flush enabled the large-model loads that had
  been failing went through in these trials, including the MoE that never loaded plain. The
  residual noted at the time has since been traced to the SVM paths and closed by the map-side
  extension of the same flush.
- **A rare extreme-size dispatch fault.** Across the day's boots the 16.7M-thread probe faulted
  once and the 8.4M probe once (a fresh boot's first run); the two dedicated benchmark boots ran
  the full sweep clean, though that 30/30 is a count from the day rather than one the shipped
  directories reproduce. Later retesting adds six more clean runs at 16.7M threads, five of
  them consecutive, and three at 8.4M, with no faults logged. So it is rare enough that it has not
  been reproduced deliberately since, but it was seen twice and is not called fixed.
- **One hard crash.** In roughly sixty heavy runs, one power-cut-level crash (a large dispatch on
  an already heavily used boot). Rarer, still possible.
- **Flash attention failed on most boots, until the macro fix.** A dedicated reboot campaign
  measured garbled `-fa on` output on nearly every boot sampled, warm, cold and verification boots
  alike, while the sustained-compute wedge was absent on every boot. The per-boot record was not
  retained, and the count once given here, sixteen of seventeen, cannot be traced to anything that
  survives: the contemporaneous working note from 11 August tallies ten of eleven boots at that
  point, and no later tally exists in the logs, in the notes, or on the board. Ten of eleven is what
  can be produced, the larger figure is not, and the conclusion does not turn on which is right. That
  rate, and the per-boot garble patterns, are what make the boot-lottery reading tempting. It does not
  hold: with gfx1013 added to the RDNA1 macro the same test is coherent on every boot tried
  (nine prompts over three fresh reboots), and the perplexity numbers are bit-identical across
  boots. A caution preserved from the same campaign: its first pass also reported model loads
  hanging on every boot, and that turned out to be a test-harness mistake, not the board. A newer
  llama.cpp CLI ignores `-no-cnv`, drops into its interactive console, and spins on a closed
  stdin; every "hung" load had in fact completed and generated (all fifteen original logs contain
  finished generations at full speed). With the flag that build actually needs (`--single-turn`)
  the same loads complete in seconds: twelve of twelve across three fresh reboots in the surviving
  note, which is again less than the twenty-five of twenty-five once claimed here and is again what
  can actually be produced.
- **What is reliable, by contrast.** A 40-minute soak of back-to-back heavy compute and GEMM ran
  439 iterations with zero wrong results and zero faults, model loads with the corrected
  invocation completed in seconds on the same boots that fail flash attention, and ROCm compute
  and Vulkan ran concurrently without interfering. That iteration count comes from the working
  notes of 10 August; the soak's own output was not kept, so it is quoted instead of produced. The
  same conclusion has since been reached from soaks whose logs are here, eight hours on the small
  model and 8.1 hours rotating three large ones, both linked above. So sustained ROCm *compute* on a
  booted-and-working board was dependable even while attention looked boot-dependent; with the
  macro fix, attention joined it.
- **A second soak, on the large models.** The soak above runs one small model; this one rotates
  qwen3-8B, qwen3-14B and the 35B MoE, each round doing a prefill benchmark, an eight-chunk
  perplexity gate and, once per rotation, the allocation-churn sweep. Across 78 rounds in 8.1
  hours, 26 per model, there is **exactly one distinct perplexity value per model** (7.3503,
  6.3970 and 5.1887, all 26 runs of each identical), prefill spread is 1.1 percent on the 8B, 0.5 on
  the 14B and 1.1 on the MoE across the whole run, all 26
  churn sweeps complete with no faults, and the temperature peaks at 94C against a 77C mean. Since
  each round loads a multi-gigabyte model from scratch, this
  also exercises the large-model load path, which used to be the least reliable part of this
  system. ([`logs/soak-large-2026-08-14/`](logs/soak-large-2026-08-14/))
- **A third soak, on the stack as it now stands.** The two soaks above predate the corrected patch
  scripts, the native PyTorch build and the current llama.cpp patch set, so nothing had run for
  hours on the recommended configuration. Forty-two rounds over eight hours four
  minutes, each one a prefill benchmark, a perplexity gate, an allocation-churn sweep, and every
  third round a PyTorch training loop. Nothing moved: **one distinct perplexity value across all 42
  rounds, 8.9442**, prefill 661.27 to 663.61 t/s (a 0.35 percent spread), 42 of 42 churn sweeps
  clean, zero GPU faults, 62 to 71 C. The fourteen PyTorch training runs returned an identical
  final loss of 0.00048 every time, so the training path is deterministic across eight hours that
  also included hundreds of large allocations. The perplexity figure is the same one the earlier
  soak returned across its 43 rounds and the same one a single gate returns today, so it is stable
  across builds, boots and months, not within one run
  ([`logs/loose-ends-2026-08-18/soak/`](logs/loose-ends-2026-08-18/soak/)).
- **An eight-hour soak on the finished stack.** Forty-three rounds, each one a twenty-repetition
  pp2048 hammer, a wikitext perplexity gate, and an allocation-churn sweep (the workload the
  map-side flush fixes), on the full recipe with the flush parameter set from the kernel command
  line. Every round passed: 43 perplexity runs returning one distinct value, 11.0521, bit
  identical from the first round to the last; prefill flat at 661.5 to 663.6 t/s with no drift;
  43 churn sweeps completed with zero faults, where the same sweep aborts within about ten
  seconds if the map side is disabled. Thermals over the 8.2 hours: 94C peak, 75C mean, the
  shader clock pinned at 1500 MHz under load with one sample at 1444, so the governor is holding instead of throttling, under the 100C cut-off but with thin headroom.
  ([`logs/soak-2026-08-13/`](logs/soak-2026-08-13/))
- **Batched compute** corrupted until the KQV precision fix (the precision caveat), including at context 1024; with the fix, exact at every context tested. Solved in software, pending upstream.
- **Prefill**: about 808 to 892 t/s at pp512 (boot variance) on the default path once the RDNA1
  macro enables the real integer-dot emulation (was 124 without it), and a reported 936, not kept,
  gated-correct on the
  forced-BLAS build with the f32 compute type, the fastest correct path measured. The remaining
  gap to Vulkan is what is left after those, not Tensile tuning.
- All of it is one board, as ever.

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

One qualification was attached here originally and no longer holds. It read that the patched runs
above came from a boot where the module happened to come up at 40 CU, that the same patch more often
left the board at 24 CU where even a trivial dispatch wedges, and that the correct flush and a
working compute queue could therefore not be had together. All three parts are now known to be
artifacts: the 24-CU boots came from a misapplied unlock patch, and 24 CU is not a wedged state at
all once `sched_policy=2` is off.
[Observation 3](#observation-3-the-unlock-the-fix-and-the-wedge-looked-entangled) sets out both
mistakes with their controls.

### A useful contrast: the graphics queue runs the same compute cleanly

The clearest single test runs the identical kernel on the graphics queue instead of the compute
queue, by porting it to OpenCL ([`patches/ocl_compute_probe.c`](patches/ocl_compute_probe.c)) and
running under **RustiCL** (`RUSTICL_ENABLE=radeonsi`), which dispatches through the
graphics/universal queue as RADV does:

| size (threads) | HIP, stock configuration | RustiCL (graphics queue) | HIP, working configuration |
|----------------|--------------------------|--------------------------|----------------------------|
| 1,048,576 | ok | 0 wrong, 2.0 ms | ok |
| 4,194,304 | ok | 0 wrong, 46 ms | ok |
| 8,388,608 | 525,308 wrong / wedge | 0 wrong, 92 ms | 0 wrong |
| 16,777,216 | wrong / hang | 0 wrong, 184 ms | 0 wrong, 5 runs of 5 |

The graphics queue was correct and fast at every size, including a sustained
many-small-dispatch pattern (1M threads times 200 sequential launches), with no wedges
([`logs/rusticl_graphics_queue_ok.log`](logs/rusticl_graphics_queue_ok.log),
[`logs/rusticl_sustained_ok.log`](logs/rusticl_sustained_ok.log)).

The fourth column is what this contrast looks like now, and it changes the conclusion. When the
first two columns were measured, the reading was that the shader hardware, memory and ALUs are
fine and the fault lives specifically in the MEC compute-queue path, which would explain why
Mesa's route-through-graphics fix works and why ROCm, unable to do that, was stuck. Retested under
the working configuration, HIP on the MEC compute queue is correct at both of the sizes that used
to fail, including five consecutive runs at 16.7M threads with no faults logged. The two queues
are not distinguishable by this test any more.

What the contrast does establish is narrower than it looked: the graphics queue was already clean
when the compute queue was misconfigured, which located the problem above the shader cores rather
than in them. It does not show a MEC-specific hardware defect, because the compute-queue column
was measured with the stock flush and `sched_policy=2`, both of which the working configuration
changes. Mesa's decision to route around the compute queue remains sound for a driver that has to
work on unpatched systems.

**Where this lands now:** under the working configuration (corrected flush at 40 CU, hardware
scheduling) the silent wrong results were not observed at all: every shipped probe
run on this configuration reports `ALL CORRECT`, 105 of the 107 results in these logs and
thirty-three of the thirty-four at the old failing size of 8.4M threads, the two exceptions being a
stock-flush run and a deliberate override trap and not this configuration, with two isolated
faults elsewhere in the day as the residual.
Two fractions stood here until 27 August, 17/17 in a counterbalanced A/B and 30/30 across two
benchmark boots, and neither denominator can be produced from what is shipped; the counts above
can be. The 40-CU
qualification above does not apply under this configuration on either kernel measured, 6.18.9 or
7.1.5 (see Observation 3).

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
to preempt a queue and it never yields. So it is a preemption that does not complete, not a signal
going missing; that preemption is a routine queue eviction, traced
in [What the wedge appears to be](#what-the-wedge-appears-to-be-a-queue-eviction-whose-preemption-times-out)
below. Either way it resembles the "compute-only queue doesn't work properly" that Mesa documented
and chose to route around.

No driver knob removed it in these tests. Tried without effect: `amdgpu.sched_policy` 0, 1, and 2
(1, HWS without over-subscription, was worse, a stuck dispatch there escalates to a full GPU reset
rather than a per-queue eviction), CWSR on and off, `HSA_ENABLE_INTERRUPT` 0 and 1, `amdgpu.mcbp`
(mid-command-buffer preemption) on and off, `amdgpu.queue_preemption_timeout_ms` raised from the
9000 default to 90000, transparent hugepages set to `never`, HIP graphs on and off,
flash-attention on and off, 24 versus 40 CU, and native-versus-override builds. Two firmware
versions were compared by extracting the older Cyan Skillfish set (release 21.40) from the
linux-firmware git history and force-loading it in place of the current 21.50 (different MEC/CP
microcode, same RLC); the wedge behaved the same on both. Pacing dispatches with idle gaps between
them looked promising on single runs but did not hold up: over ten runs on a fresh boot it wedged
about as often as back-to-back. The list below records what did not help, and is not a claim that
nothing can; the per-knob results are in
[`logs/wedge_knob_sweep.txt`](logs/wedge_knob_sweep.txt).

A newer kernel alone did not fix it. That was first apparent from source: the compute-queue reset and
preemption functions in this path (`gfx_v10_0_kiq_reset_hw_queue`, `gfx_v10_0_reset_kcq`) are
byte-for-byte identical between 6.18.9 and current mainline, and mainline carries no gfx1013-specific
preemption code. It also holds when actually booted. Fedora's kernel 7.1.5 (about a year newer than
6.18.9, and newer than the ROCm 7.1 / kernel 7.0 stack others report on in
[ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313)), with the same amdgpu rebuilt to carry
only the 40-CU unlock and the stock flush, comes up at 40 CU and reproduces both problems: the bare
compute probe is correct at 1M threads but, across four fresh boots, failed every time at 8M, twice
with silent wrong results (about 2.3M and 3.3M dropped-store elements of 8.4M) and twice with a GPU
memory-access fault, and wedges at 16M with `cp queue preemption time out`; native gfx1013 rocBLAS is
correct at N=1024 and N=2048 (about 226 and 1400 GFLOP/s) but faults at N=4096 with a
`GCVM_L2_PROTECTION_FAULT`, the same fault class others flag in that thread. Logs:
[`logs/kernel-7.1.5/`](logs/kernel-7.1.5/). `HSA_XNACK=1` is also a dead end here: the chip reports
`gfx1013:xnack-` and stays that way even with `amdgpu.noretry=0`, so retry-fault memory coherence
(which would replace the eviction path below) is not available in hardware.

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
points stood out. Almost every GEMM that completed was numerically exact, and within a single
long-lived process (one allocation, many dispatches, [`patches/sgemm_iter.cpp`](patches/sgemm_iter.cpp))
the failures were wedges, not wrong answers.
The exception was at larger sizes: a single N=8192 GEMM once returned a wrong result in an earlier
run (checksum mismatch, not captured in the logs here), so "structured kernels never corrupt" would
be too strong; wrong results are rarer with them, not absent. And the wedge
looked intermittent, not a clean size threshold: N=1024 hung once and then ran fine on
retry. That intermittency is why "just keep dispatches small" seems unlikely to be made reliable.
On a fresh boot the queue tends to tolerate roughly one large sustained dispatch and then wedge on
the next, whether or not the dispatches are paced.

### What the wedge appears to be: a queue eviction whose preemption times out

Tracing the driver gives a more specific picture than "stuck mid-flight", and it lines up with the
intermittency. On this scheduler (`sched_policy=2`, the non-HWS path) the `cp queue preemption time
out` message is printed only by `kgd_hqd_destroy`, which is reached only when a compute queue is
being destroyed or evicted. The timeout is not a dispatch failing on its own; it is a queue
*eviction* whose MEC preemption does not complete.

A function-tracer stack trace on the eviction entry point
([`logs/ftrace/wedge_eviction_stack.txt`](logs/ftrace/wedge_eviction_stack.txt)) shows what triggers
those evictions, captured on a wedging run while the process was hung mid-dispatch (not exiting):

```
evict_process_queues_nocpsch  <-  kgd2kfd_quiesce_mm  <-  svm_range_evict
  <-  svm_range_cpu_invalidate_pagetables  <-  __mmu_notifier_invalidate_range_start
  <-  __split_huge_pmd  <-  __x64_sys_munmap
```

The trigger is the process's own `munmap`. The ROCm/HIP/Tensile runtime churns its address space
(a couple hundred `munmap` calls over a run, from code-object and module management, not
application `malloc`, and roughly constant instead of scaling per GEMM). Each unmap that overlaps
a KFD SVM range fires an MMU notifier, and KFD responds by quiescing, that is evicting, all of the
process's compute queues. Evicting a compute queue means preempting it on the MEC, and on this
board that preemption intermittently times out when a dispatch is in flight. Process exit is a
second trigger for the same eviction path (via a userptr invalidate instead of `munmap`), which
is consistent with the HIP exit-freeze.

If that reading is right, the eviction itself is ordinary KFD behaviour that happens on any ROCm
system; the board-specific part is only that the MEC preemption for it sometimes never completes.
It would also explain the pattern above: larger dispatches spend longer in flight, so an eviction is
likelier to land while one is running; pacing does not help because the runtime keeps churning
memory regardless; and no scheduler or timeout knob helps because the failing step is the MEC
preemption, below all of them. This is one board's trace and an inference from it, not a proof, but
it is more specific than the earlier guess and it is reproducible with the recipe in the log file.

When a large dispatch fails as a page fault, not as a wedge or wrong results, amdgpu
decodes it. In one captured instance the faulting client is **TCP** (the shader's vector-L1 /
vector-memory path), the page is mapped and the page-table walk succeeds (`MAPPING_ERROR: 0x0`,
`WALKER_ERROR: 0x0`), and the access is rejected on **permission** (`PERMISSION_FAULTS: 0x3`) on a
**read** (`RW: 0x0`). So this is a permission rejection on a mapped page, not a missing one. That
is a single decode, and the rest is interpretation rather than measurement. A permission fault on
a mapped page, on a read, matches the known pattern of a buffer unmapped while a shader still
references it (a use-after-free on the GPU side), which matches the runtime `munmap` churn and
queue eviction of Observation 2 seen from the memory controller. Read that way, the same eviction
that usually times out the MEC preemption and wedges the queue can instead let an in-flight read
land on a just-revoked page and fault; the fault address being in the process's SVM range fits. A
stale translation left by the PASID flush (Observation 1) could also leave a wrong permission, so
one trace does not cleanly separate the two. Offered as a hypothesis, from one board. The full
decode is in
[`logs/deep-dive-2026-07-28/l2_fault_decode.log`](logs/deep-dive-2026-07-28/l2_fault_decode.log),
and the same `GCVM_L2_PROTECTION_FAULT` is the fault class
others report from an image-bandwidth test in
[ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313). A bare HIP streaming-read kernel
([`patches/bw_probe.cpp`](patches/bw_probe.cpp), no arithmetic, no rocBLAS) reproduces the
size-dependent failure on its own.

**Where this lands now:** the wedge described in this section belongs to the software-scheduling
path, and that is now measured directly and not inferred. Every observation above was taken
under `amdgpu.sched_policy=2` (including the kernel-7.1.5 test, whose command line carried it), and
the message itself is printed from the nocpsch eviction path this section traced. With hardware
scheduling the same workloads run clean up to N=8192 sustained, on 7.1.5 and on 6.18.9 alike.
Adding `sched_policy=2` back to an otherwise clean 6.18.9 restores the failure exactly: the compute
probe hangs at every size, SGEMM wedges at N=256, and 16 `cp queue preemption time out` messages
appear. Crossing the setting with CU count separates the two cleanly on this board, since both
hardware-scheduling cells are clean and both policy-2 cells wedge at 24 and 40 CU
([A working configuration](#a-working-configuration)).

That leaves the 6.18 knob sweep, which recorded `sched_policy=0` as wedging just like policy 2 and
was read here as showing that the scheduler alone was not enough on that kernel. It cannot carry
that weight, for a reason worth being explicit about: the sweep does not record which flush
was in the module it ran against. It was taken at 40 CU on 6.18.9, which at the time meant the
stock flush, since the corrected one was then believed to force 24 CU. So its `sched_policy=0`
sample was most likely a stock-flush measurement and not the configuration that works, but the log
does not say so outright and the inference is mine. Read as a negative result about hardware
scheduling on its own, it should be treated as inconclusive, not as evidence either way.
The eviction analysis here still describes what happens under policy 2, but the "firmware or
silicon limit" conclusion was too broad. The MEC preempted reliably
whenever the firmware scheduler asked; what failed was the driver-initiated `hqd_destroy`
preemption path.

## Building a native gfx1013 rocBLAS

A long-standing workaround for the missing gfx1013 matrix kernels is to build for **gfx1010** and
run with `HSA_OVERRIDE_GFX_VERSION=10.1.0`, since gfx1010 and gfx1013 share an ISA. That is still the
wrong tool, but the reason is worth restating, because retesting it under the working configuration
gave a different and more dangerous answer than the original one.

Originally the override failed loudly: anything using scratch or private addressing hit
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, and only a small scratch-free SGEMM survived. Retested
now, rocBLAS GEMMs through the system library run clean under the override at N=256, 1024 and 4096
with no wrong results, so the "only a tiny SGEMM" limit has lifted, which is presumably why the
override keeps getting recommended.

The trap is what happens to code you compiled yourself. The bare compute probe, built for gfx1013,
run under the override on the same boot:

| configuration | result | kernel time |
|---|---|---|
| no override | all 4,194,304 elements correct | 1933.0 ms |
| `HSA_OVERRIDE_GFX_VERSION=10.1.0` | all 4,194,304 elements wrong, every one zero | 0.1 ms |

Both rows were rerun on 26 August and are captured in
[`logs/override-trap-2026-08-26/`](logs/override-trap-2026-08-26/); until then this table had no log
behind it. The override arm reproduces exactly, `wrong=4194304/4194304` with the first mismatch
reading `got 0, want -2109779631`, and the probe's banner shows why: it reports `gfx1010:xnack-`
under the override where it otherwise reports `gfx1013:xnack-`. The first row is the rerun, 1933.0 ms.

A dispatch that returns in 0.1 ms where the real thing takes two seconds did not run. Telling the
runtime the device is gfx1010 means the gfx1013 code object no longer matches, the launch quietly
does nothing, the output buffer keeps whatever it held, and there is no error anywhere. The
override can make a prebuilt library appear to work while silently zeroing every kernel of your own,
which is worse than the aperture violation it replaced: that one at least stopped the program. The
native build below avoids the whole question.

Following the approach of
[ROCm/rocm-libraries PR #8838](https://github.com/ROCm/rocm-libraries/pull/8838), rocBLAS was
instead built natively for gfx1013 on Fedora's system ROCm. That meant working through a chain of
Fedora-specific issues: system ROCm lives in `/usr` instead of `/opt/rocm`; `amdclang++`,
`msgpack-cxx`, and `roctracer` were missing; and gfx1013 had to be added to Tensile's `SupportedISA`
and `AsmCaps` and to the Tensile and rocBLAS C++ architecture enums. The full worked recipe is in
[`scripts/build_rocblas_gfx1013.sh`](scripts/build_rocblas_gfx1013.sh).

The result is a real `librocblas.so.4.4` (about 37 MB) with 56 gfx1013 Tensile libraries and
genuine gfx1013 code objects (`Kernels.so-000-gfx1013.hsaco` reports ELF machine "AMD GPU" with
gfx1013 flags, native rather than an override). It runs on the board with no `HSA_OVERRIDE` and
computes correct GEMMs, per the table above. The main value is that it removes the override from
the picture: where a native rocBLAS GEMM works it is correct, and where it does not it is the
wedge, not an aperture mismatch. At the time of the original investigation it was not enough for
reliable inference, because the wedge still applied to the large fused matmuls that inference
leans on; under the working configuration that limit is gone and this library is the one behind
the SGEMM and PyTorch numbers above.

Two related notes. Fedora's own system rocBLAS "supports gfx1013" only by symlinking its gfx1013
Tensile files to the gfx1010 ones, so a stock install is silently running the gfx1010 override,
which is a fair part of why "rocBLAS works on gfx1013" reports coexist with real workloads
failing. And the same wedge reached a mainstream framework: the official `torch 2.9.1+rocm6.4`
wheel detects the board but ships no gfx1013 (or gfx1010) kernels, so a matmul fails immediately
out of the box; with a native gfx1013 rocBLAS grafted in, PyTorch matmuls were correct up to
N=4096 single, but a sustained loop (N=4096 repeated) wedged the compute queue the same way the
probes and rocBLAS did ([`logs/deep-dive-2026-07-28/`](logs/deep-dive-2026-07-28/)). Under the
working configuration the sustained loop is clean (the PyTorch table above); what remains on the
torch side is the wheel's missing gfx1013 elementwise kernels and the allocation discipline, both
described there.

## Observation 3: the unlock, the fix, and the wedge looked entangled

Two facts appeared to collide here, and the correction below reports that neither of them holds up
on retest. First, without the 40-CU unlock the board runs at 24 CU, and at 24 CU a trivial compute
dispatch appeared to wedge: `compute_probe` returned correct results at 40 CU and hung at 24. So
the community
**40-CU unlock** ([duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock))
looked like a prerequisite for any ROCm compute here, not just inference, which is
counterintuitive (more CUs, more stable) and seemed to tie the wedge to the harvested-CU / WGP-mask
configuration.

Second, in the kernel tree used here the 40-CU unlock's register write lives inside
`gfx_v10_0_kiq_reset_hw_queue()`, a function that only runs when a KIQ hardware queue is reset. On
the stock driver, the KIQ-fence bug triggers such a reset during boot, which incidentally fires the
unlock, so the board comes up at 40 CU.

Together those appeared to undercut the correctness change on this board, and the reasoning ran:
`flush_pasid_uses_kiq = false` removes the KIQ activity that was triggering the reset, so the unlock
does not fire, the board comes up at 24 CU, and at 24 CU compute wedges. Every step of that except
the first is now known to be wrong, for the reasons in the correction below, but it is recorded as
it was believed. Testing it at the time with the native rocBLAS GEMM, on the patched module at 24 CU
every size wedged, including N=256
([`logs/rocblas/sgemm_sweep_patched_24cu.log`](logs/rocblas/sgemm_sweep_patched_24cu.log)). An
earlier session did once boot the patched module at 40 CU, which is where the correct patched
`compute_probe` results (Observation 1) and the prefill pass below came from, but that
patched-and-40-CU state did not reproduce on later boots.

So with that tree the available states appeared to be the correct TLB flush at 24 CU (where
compute wedges) or the working 40-CU configuration with the buggy flush (wrong results and
freeze), but not both. The correction below shows that conclusion was too broad. A controlled
rebuild reproduced the entanglement: with `flush_pasid_uses_kiq = false` and the unlock present
only in the reset path, the board comes up at 24 CU and the bare probe wedges, the predicted
state.

The obvious escape, moving the unlock write out of the reset path into normal init, did not work in
attempts here, but in an informative way. Placed at the end of `gfx_v10_0_hw_init` (after RLC init,
CU harvesting, and CP resume) the register writes do run, but the board still reports 24 CU: the
40-CU state seemed to need the queue-reset context around `kiq_reset_hw_queue`, not just the
register values. Trying to fire that reset deliberately at the end of init, or from userspace via
`amdgpu_gpu_recover`, was either permission-gated or hung the board.

**None of the three premises above reproduce.** Rebuilding
the current patch set and booting it, with the correctness fix genuinely enabled, all three states
hold at once. And 24 CU is not a wedged state either: booted with the unlock disabled
(`simd_count 48`) and the corrected flush active, the bare compute probe is correct at 4096, 16384
and 32768 blocks, and the native rocBLAS SGEMM runs ten clean iterations at every size from 256 to
4096, the N=256 case being the one this section says wedged. No preemption timeouts appear in
dmesg. Measured on five kernels spanning the whole range from the earlier work to the
current one (6.18.9, 6.18.16, 6.19.14, 7.0.13, 7.1.2), each module-only, `dracut`-installed, with
the boot arguments verified before booting:

| check | result with `flush_pasid_uses_kiq = false` |
|---|---|
| CU count | `simd_count 80`, that is 40 CU, on all five kernels |
| sustained GEMM | N=4096 at about 30 ms per iteration, no wrong results, no preemption timeouts anywhere. Within this ladder it is 20 iterations on 6.18.9, 6.18.16 and 6.19.14 and 30 on 7.0.13 and 7.1.2. A 50-iteration sustained run does exist for 6.18.9, in [`logs/kernel-equivalence-2026-08-17/`](logs/kernel-equivalence-2026-08-17/) four days later, and the old wording merged the two campaigns |
| bare compute probe at 8.4M threads | correct three times out of three (the size that returned 525,308 wrong results with the flush left on) |

Three controls make that readable. The module parameter only exists in the patched module, so its
presence identifies which module booted. The parameter does real work: the same kernel booted with the
flush left on core-dumps the probe, core-dumps the sustained GEMM, and froze the board, which is
the failure this document describes. And the CU measurement is sensitive, not cosmetic:
disabling the unlock drops `simd_count` to 48 and slows the same GEMM to 48.7 ms per iteration, a
ratio of 1.62 against the 40-CU number, close to the 40 to 24 ratio expected.

The correct flush and 40-CU compute are not mutually exclusive on this part, and two things made it
look as though they were. The first is a misapplied patch. The companion project's own notes from May warn that targeting
`gfx_v10_0_get_cu_info` naively can land the register write in a `gfx10_kiq_*` function instead,
because a forward declaration appears earlier in the file, and that the symptom of getting it
wrong is a module that loads while the board stays at 24 CU. Which is the state described
here, and it explains why the write appeared to live in KIQ context: in that build it did, by
accident. The second is described below: the scheduler setting that was held fixed across the
comparison.

**Why it looked that way, established 2026-08-15.** The earlier measurements were right and they
reproduce exactly; the attribution had an invisible constant in it. Those 24-CU sweeps ran with
`amdgpu.sched_policy=2`, which this document recommended at the time as protection against the
HIP-exit freeze. Restoring that argument reproduces the published result: at 24 CU with
the corrected flush, N=256 wedges on a timeout with `cp queue preemption time out`, exactly as the
archived log records. The control that was never run then is the one that matters: **the same
policy at 40 CU wedges identically**, same size, same two preemption timeouts.

| CU count | scheduler | N=256 |
|---|---|---|
| 24 | `sched_policy=2` | wedge, 2 preemption timeouts |
| 40 | `sched_policy=2` | wedge, 2 preemption timeouts |
| 24 | hardware scheduling | clean, and clean to N=4096 |
| 40 | hardware scheduling | clean, and clean to N=4096 |

The wedge was never about the CU count. It belongs to the software scheduler, which the two-by-two
factorial later pinned independently with the same signature. The original experiment varied the
CU count while holding the scheduler fixed across both arms, so the thing doing the damage could
not be seen. That also dissolves the "more CUs, more stable" paradox the text above flags as
counterintuitive: it was not a hardware peculiarity, it was a boot argument. The explanation
offered above, that the unlock's register write lived inside `gfx_v10_0_kiq_reset_hw_queue()` and
so depended on KIQ activity the flush fix removes, does not match the community patch's history:
every released version of it, from the first in May 2026, writes those registers from
`gfx_v10_0_get_cu_info()`, and none mentions the reset path at all. So either the tree used then
carried a hand-placed variant that was not kept, or the coupling had a different cause. Stating
that honestly seems better than keeping a tidy mechanism the artifacts do not support.

One practical trap found while testing this, which would silently invalidate any experiment that
varies the unlock from the kernel command line: `/etc/modprobe.d/bc250-40cu.conf` carries
`options amdgpu bc250_cc_write_mode=3` and is baked into every initramfs, so the unlock fires
whether or not the command line mentions it. An A/B that toggles the boot argument alone changes
nothing, and would read as the unlock being insensitive to configuration.

The direct "is the wedge a regression" experiment was attempted two ways, both inconclusive for
frustrating reasons. A stock older kernel (Fedora 6.6.14) does not bring this board up at all:
amdgpu's display code faults during KMS init
([`logs/older-kernel-6.6-display-oops.log`](logs/older-kernel-6.6-display-oops.log)), and on the
kernels tested, BC-250 support appears only from about kernel 6.18 (Fedora's 6.18.9 amdgpu exposes
`bc250_cc_write_mode`; its 6.17.1 does not). Reverting the one named TLB regression on 6.18 lands
back in the 24-CU-wedges-everything state above. So whether the wedge itself is a regression or a
hardware limit is unresolved here. The graphics-queue contrast was once read as leaning toward a
hardware or firmware cause, but it no longer supports that: HIP on the compute queue is correct at
the same sizes under the working configuration, so the two queues no longer differ on that test.

**Where this lands now:** the entanglement was not real, and it was not a property of kernel 6.18.
The unlock's register writes run during ordinary driver init and the board comes up at 40 CU with
`flush_pasid_uses_kiq = false`, on every boot, with the module's own init log line confirming both
states together. This is now measured on 6.18.9 as well as on 7.1.5, with the same patch set, so
the "correct flush at 24 CU, or working 40 CU with the buggy flush, but not both" trade this
section documents was an artifact and not a kernel-specific behaviour. Two causes account for
it: the unlock patch of the time was applied into a `gfx10_kiq_*` function instead of
`gfx_v10_0_get_cu_info()`, which produces exactly the reported symptom of a module that loads
while the board stays at 24 CU, and `sched_policy=2` was held fixed across both arms of the
comparison, which wedges compute at 24 CU and 40 CU alike. The wedge that made 24 CU look fatal
belongs to the scheduler policy, not to the CU count: at 24 CU with hardware scheduling the board
passes the whole battery including perplexity, and at 40 CU with `sched_policy=2` it wedges on a
trivial dispatch. The factorial is in [A working configuration](#a-working-configuration).

## How far ROCm inference gets

**The state before the ten patches that followed the first three.** On the thirteen-patch build ROCm prefills at 0.96 to 1.29 of Vulkan, not a third of it, and decodes at 0.81 to 1.01. The current figures are in
[README.md](README.md#llamacpp-rocm-against-vulkan). What follows is kept because the sections after it
argue from these numbers.

This section records the inference attempts from the 6.18-era investigation; the working numbers
now live in [What the working configuration measures](#what-the-working-configuration-measures).
Its lasting value is the fault analysis: the load-time aperture violation documented here later
turned out to be the allocation-reuse defect, which the runlist flush resolves.

With the patched module, llama.cpp's HIP backend got further than before, though not to a usable state at the time.

A completing prompt-processing pass was achievable with a native gfx1013 build and rocBLAS kept
out of the hot path:

```bash
cmake -B build-hip -S . -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 \
  -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja -C build-hip llama-bench

HSA_ENABLE_SDMA=0 GGML_CUDA_FORCE_MMQ=1 \
  ./build-hip/bin/llama-bench -m qwen2.5-1.5b-q4km.gguf -ngl 999 -fa 1 -p 128 -n 0 -mmp 0
# qwen2.5-1.5B Q4_K_M, pp128 about 35 tok/s, RC=0
```

`-fa 1` (flash attention) routes around rocBLAS via ggml's own gfx1013 kernels, and
`HSA_ENABLE_SDMA=0` was required when this was run, though it is not any more: the navi12
microcode substitution removed the need, and the recipe now offers it as the alternative to
substituting the firmware and not as a requirement. The `GGML_CUDA_FORCE_MMQ=1` in that command line
does nothing at all: it is a compile-time option in this version, so setting it in the environment
is inert. The command is
kept as it was actually run rather than tidied, since the point of this section is the historical
record. Two further warnings applied in hindsight: this
recipe and every rate in this section ran with flash attention on, which at the time garbled on
most boots (later traced to the RDNA1 macro entry, see the caveats; fixed since). These runs were
judged by completion and return code only, so their output correctness is unknown; the recipe
([`scripts/native_fa.sh`](scripts/native_fa.sh)) is kept for the record, and the working
inference configuration above is the one to use.

Token generation was more workable than earlier runs suggested, and what gated it was mostly not
the compute queue. On a later stack (`llama.cpp` b9265, native gfx1013, 40 CU), decode ran and
generated tokens (around 40 tok/s, correctness unknown per the warning above) in most of the runs
where the model finished loading, so the earlier read of decode as hard-blocked was too strong.
Two things gate it, and a compute-kernel scratch fault is not the main one. Logs:
[`logs/inference/`](logs/inference/).

First, an intermittent aperture violation. It does not fire on most decode-reaching runs: the
single-boot campaign below reached decode six times and faulted on none. It does recur across
sessions, though, and every instance captured has the same shape. Under `AMD_LOG_LEVEL=3` the
fault is not inside a compute kernel: on the runs that fault, the abort happens before any ggml
kernel dispatches at all, on a runtime host-to-device copy (`__amd_rocclr_copyBuffer`) whose host
source is past the GPU's legal aperture
([`logs/inference/decode_copybuffer_aperture_violation.txt`](logs/inference/decode_copybuffer_aperture_violation.txt),
zero compute kernels dispatched
before the abort). That points at this board's UMA host-buffer mapping (only `hipHostMalloc`'d
memory is GPU-legal here), a memory-mapping fault, not a scratch/private-memory problem in
`mul_mat_vec_q`. An earlier trace on a different llama.cpp build had read as the latter; the two
builds may differ, so this is stated for the current one.
[`logs/inference/decode_aperture_violation.txt`](logs/inference/decode_aperture_violation.txt) is
that earlier trace. `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, which removes that host copy on a UMA
APU, did not reliably remove the fault. That one is a day observation like the load table below:
the variable is read by ggml, but no capture here records a run with it set, so "not reliably"
is the strength of it.

Second, and far more limiting in practice, the flaky and slow model load. In one single-boot run
the model failed to finish loading 15 of 21 attempts, and it got worse with cumulative GPU use;
the runs that did load then decoded cleanly, with no aperture fault in that run
([`logs/inference/decode_campaign_stats.txt`](logs/inference/decode_campaign_stats.txt), collected
by [`scripts/decode_stats.sh`](scripts/decode_stats.sh)). This is HSA-signal-level (the
`hipEventSynchronize` overhead this board is known for), not the kernel fence timeout: a module
rebuilt with a fence-fallback timer at 2 ms did not speed the load. So in practice the wall for
decode on that stack was getting the model loaded, not a decode-kernel fault.

For scale at the time: on 6.18 even where ROCm prefill completed it was roughly 36 times slower
than Vulkan on the same model and did not survive repetition. That ratio is long gone. Under the
working configuration the same comparison is about 2.3 times, 807.9 against 1844.3 t/s at pp512,
and the remaining difference is between two quantized-kernel implementations and not anything
failing. Decode, sustained generation, and the compute path under them work as well
(the tables above).

## ROCm vs Vulkan

**Also the pre-thirteen-patch state**, as the section above says.

Vulkan appears here only as the baseline the ROCm path is measured against; the full Vulkan
characterization of the board (many models, context scaling, memory ceilings) lives in
[akandr/bc250](https://github.com/akandr/bc250) and is not repeated here. The current side-by-side
is in [What the working configuration measures](#what-the-working-configuration-measures); the
summary after the fixes is that Vulkan keeps prefill, but by how much depends on the model, and
the pattern is useful to know: across the six models gated here the ratio runs from 2.29x on the
1.5B down to 1.41x on the 27B, with the two largest models the closest. The often-quoted "about
2x" is roughly the median and hides a real trend in ROCm's favour as models grow. It was 10x
before the fixes, and the small model was reported at 936 t/s at pp512 with the forced-BLAS build, a
figure not kept, against
808 on the default path.

ROCm's decode share likewise depends on the model: closest on the 8B at Q8_0, roughly three fifths
on the Q4_K models at any depth, 46 percent on the 27B and 40 percent on the MoE. And ROCm alone
offers FP64, PyTorch including training, and custom HIP kernels. One Vulkan-side note: a community
patch set that re-enables the dedicated compute queues RADV normally hides on this chip
([bc250-gfx1013-fix](https://github.com/DryhoppedIPA/bc250-gfx1013-fix), see References) was
built and verified here; it runs llama.cpp correctly and a few percent faster
(tg128 210.6 to 217.4 on the small model), with its larger async-compute gains aimed at
graphics workloads. Neither figure was kept, noted 26 August, and no directory here holds
that build or its output, so "verified here" describes something done and not something
a reader can check. It is stated as a recollection, which matters more than usual because it
is a claim about someone else's work.

The historical 6.18-era comparison, kept for the record and
corrected against its own log on 26 August: on qwen2.5-1.5B Q4_K_M, Vulkan ran
pp128/pp256/pp512/tg128 at 1275.72, 1597.66, 1845.60 and 211.01 t/s, each a mean of three repeats
with the standard deviation the log prints beside it (0.71, 1.98, 0.41 and 0.02). ROCm produced no
throughput at all in that run: all three of its cells failed, `rc=124` at pp128, `rc=139` at pp256
and `rc=137` at pp512, which is a timeout and two killed processes
([`logs/inference/bench_rocm_vs_vulkan.log`](logs/inference/bench_rocm_vs_vulkan.log)). One figure here is worth pinning: pp128 at
35.37 t/s is captured in
[`logs/inference/rocm_prefill_works.log`](logs/inference/rocm_prefill_works.log), a separate run of
the same native gfx1013 build, while the ROCm arm of the log cited here timed out. Two runs of one
configuration, one completing and one not, is useful to know in itself and is lost by reporting them
as a single result.

## Fedora 43 with ROCm 6.4.2

The [README](README.md) is the manual for the Fedora 44 configuration. Fedora 43 with ROCm 6.4.2
remains bootable on this board and measures the same speed, so its recipe, measurements and caveats
are kept here in full for anyone running it. The one substantive difference is the fp16 toolchain
defect described below, which Fedora 44 does not have.

The BC-250 is a cheap ex-mining blade carrying an RDNA1-class APU (gfx1013, Cyan Skillfish) with
around 14 GiB of usable shared memory, 24 compute units by default and 40 with the community
unlock. Vulkan has worked on it for a while. ROCm largely did not, and this documents getting it
to.

Most of the ROCm compute stack works once a corrected TLB flush, hardware scheduling, the 40-CU
unlock and a flush-on-map-and-unmap workaround are in place. This page is the recipe and the measurements. The investigation that produced them is in [INVESTIGATION.md](INVESTIGATION.md).

One caveat belongs up here and not in the defect table. Under sustained load the board can still
take itself down: a rare GPU page fault escalates through a preemption timeout to a GPU reset, and a
reset takes the machine with it, twice out of two deliberate attempts from a completely idle GPU
([`logs/gpu-reset-fatal-2026-08-21/`](logs/gpu-reset-fatal-2026-08-21/)). Not in the way that
phrase suggests, though: captured over netconsole, the driver logs `GPU reset succeeded, trying
to resume`, and the host then stalls on a clocksource watchdog
([`logs/reset-netconsole-2026-08-23/`](logs/reset-netconsole-2026-08-23/)). The success message
turns out to mean little, since on this chip the reset path has nothing to call and returns success
without touching the hardware ([`logs/reset-smu-gc-2026-09-14/`](logs/reset-smu-gc-2026-09-14/)). The deliberate-reset
directory establishes that a reset is fatal here but captures nothing of what the kernel was doing,
since on those runs the board went away before anything reached the journal. The fault behind it
appeared five times across the twenty boots the journal held, on both kernels tested
and on the configuration recommended below. `amdgpu.gpu_recovery=0` stops the driver requesting the
reset, confirmed by calling that path directly, and is discussed in step 2. Long runs complete far
more often than not, and an
eight-hour soak returned a bit-identical correctness gate on 253 consecutive rounds before ending
that way, but this is not hardware to leave running unattended on work that matters
([`logs/journal-retro-2026-08-20/`](logs/journal-retro-2026-08-20/)).

Everything here is one board, one software stack. Measurements are reproducible and the logs are
included; explanations are working theories. Corrections welcome.

Environment: Fedora 43, ROCm 6.4.2, LLVM/clang 19, Mesa 25.3 RADV for the Vulkan comparison, the
oberon governor at 1500 MHz, llama.cpp master 7ba604f. Two LLVMs are in play and the logs show
both: clang 19 is ROCm's own, at `/usr/lib64/rocm/llvm/bin/clang++`, and compiles everything HIP
here, while the `LLVM 21.1.8` in the captured device strings is Mesa's radeonsi reporting itself.
Every item in that list is checked against the machine in
[`logs/env-versions-2026-08-27/`](logs/env-versions-2026-08-27/), which exists because the clang
version was the one thing here no capture carried. Most measurements here were taken with the
board's own SDMA microcode and `HSA_ENABLE_SDMA=0`, before the navi12 substitution in step 3 below
was known; that substitution has since been re-checked against throughput, the correctness gates,
Vulkan, allocation churn and decode at depth, and changes none of them. The kernel version is not an
ingredient:
6.18.9, 6.18.16, 6.19.14, 7.1.5 and 7.1.8 measure identically with the same patch set, the last of
those checked after a report that the patches work on newer kernels. Below 6.18 the board does not
come up at all.

Fedora 44 with ROCm 7.1.1 on the same kernel runs correctly with two fixes, a native gfx1013 rocBLAS
7.1.1 built from a script and a corrected gfx10 VGPR count in comgr, and is faster: qwen2.5-1.5B pp512
996 against 810 t/s and tg64 146 against 114, qwen3-8B pp512 307 against 243, alternated across boots.
Without them it runs no model ([`logs/fedora44-working-2026-09-15/`](logs/fedora44-working-2026-09-15/),
[`logs/fedora44-rocm711-2026-09-15/`](logs/fedora44-rocm711-2026-09-15/)).

**That speed claim does not hold.** The 1.23 and 1.28 ratios above are an artefact: the Fedora 44 GPU clock policy was not the one Fedora 43 ran under, because the upgrade had
replaced the governor configuration. With Fedora 43's restored, the two systems measure the same to
within 2 percent on every model, and so does Fedora 45. The sentence is left in place because this
section is a snapshot; the corrected comparison is in
[Fedora 44 and 45](#fedora-44-and-45-rocm-711-and-722) and in
[README.md](README.md#fedora-43-44-and-45-measure-the-same). The correctness half of the sentence,
that the two fixes are required and that without them it runs no model, still holds.

### Making it work

**1. Patch and build the driver module.** Two scripts, in this order, each taking the amdkfd
directory of a kernel tree:

    python3 scripts/apply_runlist_flush.py     <tree>/drivers/gpu/drm/amd/amdkfd
    python3 scripts/apply_svmflush_generic.py  <tree>/drivers/gpu/drm/amd/amdkfd

The first is a hand-port of GabriWar's runlist-rebuild flush, made runtime-switchable; the second
extends it to the SVM map side, which is this repository's part. Those two produce
`bc250_flush_by_runlist` and nothing else, which matters because the
check below expects three. The other two come from elsewhere:
`bc250_flush_pasid_kiq` is added to `gmc_v10_0.c` by the embedded patcher in
[`scripts/ladder_prep_rung.sh`](scripts/ladder_prep_rung.sh) (the `FLUSHPARAM` step, which turns
`flush_pasid_uses_kiq` into a module parameter defaulting to the stock behaviour), and
[`patches/amdgpu-flush-pasid-mmio.patch`](patches/amdgpu-flush-pasid-mmio.patch) is the same change
hardcoded rather than parameterised, which works but leaves nothing to set at boot.
`bc250_cc_write_mode` comes from the community 40-CU unlock referenced under
[References](#references) and is not produced by anything here. The module this repository
measures is the three together, and `ladder_prep_rung.sh` is the only place they are applied as one
set.

Then build and install module-only with
[`scripts/build_patched_amdgpu.sh`](scripts/build_patched_amdgpu.sh), which rebuilds the
initramfs. That last part is not optional: the running module comes from the initramfs, and
forgetting it is the most common way to spend a day measuring a module you are not running.

Porting to a kernel this has not been built against is mostly mechanical, with two traps worth
knowing. Some `kernel-devel` packages do not ship `amdgpu_trace.h`, and the module build then dies
on a trace include until the headers are copied across from the source tree. And the 40-CU hunk has
to go inside the *definition* of `gfx_v10_0_get_cu_info`, not the forward declaration that appears
earlier in the same file: putting it in the wrong place compiles cleanly and boots at 24 CU, which
is why the check below matters
([`logs/kernel-718-2026-08-19/`](logs/kernel-718-2026-08-19/)).

After rebooting, confirm the patched module is the one loaded. The three parameters below exist
only in it, so if they are absent the stock module is running whatever is in `/lib/modules`:

    ls /sys/module/amdgpu/parameters/ | grep bc250
    # expect: bc250_cc_write_mode  bc250_flush_by_runlist  bc250_flush_pasid_kiq

Checking the file's checksum proves nothing here, since the module that matters is the one baked
into the initramfs.

**2. Boot with these arguments, and without `sched_policy`:**

    amdgpu.bc250_cc_write_mode=3 amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=3
    ttm.pages_limit=4194304

The last one is not an amdgpu setting, and it is worth naming plainly because the large-model
figures below depend on it. TTM defaults its page limit to
half of system memory (`ttm_device.c` computes `num_pages / 2`), which on this board's 14.8 GiB is
about 7.4 GiB. The 14B and the 10.7 GiB MoE both allocate more than that, and the context ceilings
were measured peaking at 14594 MiB. The value above raises the limit to 16 GiB. That the default
actually blocks those loads has not been measured here, only read from the driver, so treat it as
a parameter this board was configured with, not as a demonstrated requirement.

Consider adding `amdgpu.gpu_recovery=0`, understanding what it does and does not buy: measured
against a real fault, it prevents the reset and keeps the host alive, but the GPU stays unusable
until reboot and still looks healthy to enumeration
([`logs/fault-usability-2026-08-24/`](logs/fault-usability-2026-08-24/)). A GPU reset does not
appear to be survivable on this
board, so the rare fault described under Known defects takes the whole machine down and not just the
process. The driver source suggests why: neither MODE1 nor MODE2 has an implementation for
this chip, both report success anyway, and the register state captured after a "reset" is the
state from before it. Upstream also lists this chip as having recovery disabled by default, but
that list is unreachable on devices without RAS support, so the parameter has to be set
by hand. A PCI function-level reset, tried as an alternative, keeps the host alive but loses the GPU
until power-off ([`logs/reset-smu-gc-2026-09-14/`](logs/reset-smu-gc-2026-09-14/)). Two small driver
changes, in [`patches/amdgpu/`](patches/amdgpu/), were tested as runtime-switchable equivalents:
making the recovery default reachable refused every
KFD reset request at the driver default and kept the host up, four trials of four, and making the
unimplemented resets fail, not report success kept the host up and the GPU computing after a
deliberate reset, four of four, though the reboot that followed needed a power cycle each time. A
gfx9-style per-queue reset ported to gfx10 ran and did not recover the queue
([`logs/reset-honest-2026-09-15/`](logs/reset-honest-2026-09-15/)). Rebinding the driver hangs the
host the same way, even on a healthy GPU, and s2idle suspend does not return on this board at all
([`logs/rebind-recovery-2026-09-15/`](logs/rebind-recovery-2026-09-15/),
[`logs/suspend-recovery-2026-09-15/`](logs/suspend-recovery-2026-09-15/)). That parameter stops the driver requesting a reset on the path those faults take.
The line above leaves it out because it means a wedged GPU stays wedged until reboot instead of
being reset, which is a trade and not a pure gain. What it leaves behind is not a guess: a
natural fault was caught under the parameter, and the sentence above describes that event. It costs
nothing measurable in throughput.

Do **not** set `amdgpu.sched_policy=2`. It was once recommended here as a freeze mitigation and is
the single thing most likely to make a correctly patched board look broken: it wedges sustained
compute at any CU count.

These can also be set through `/etc/modprobe.d`, which is what the 40-CU unlock guide does, and the
two mechanisms can disagree without saying so. If both are present, check what the module actually
received rather than what you asked for:

    cat /sys/module/amdgpu/parameters/bc250_cc_write_mode   # 3 for 40 CU
    cat /sys/module/amdgpu/parameters/bc250_flush_by_runlist
    cat /sys/module/amdgpu/parameters/sched_policy          # 0, not 2

A `modprobe.d` file that accumulated several conflicting `options amdgpu` lines over time is easy
to end up with and hard to notice, since only the last one takes effect.

**3. Fix SDMA, or disable it.** The board's own SDMA microcode never completes a transfer above
16384 bytes. Substituting the navi12 microcode fixes it outright, and both blobs ship with
`linux-firmware`:

    sudo cp /lib/firmware/amdgpu/navi12_sdma.bin.xz  /lib/firmware/amdgpu/cyan_skillfish2_sdma.bin.xz
    sudo cp /lib/firmware/amdgpu/navi12_sdma1.bin.xz /lib/firmware/amdgpu/cyan_skillfish2_sdma1.bin.xz
    sudo dracut -f

`dracut` is required, since the initramfs carries this firmware too. Back up the originals first;
reverting is a copy back, and the backup is the only way to get them again short of reinstalling
`linux-firmware`.

To check the substitution took, compare the two files rather than trusting the copy:

    md5sum /lib/firmware/amdgpu/cyan_skillfish2_sdma.bin.xz \
           /lib/firmware/amdgpu/navi12_sdma.bin.xz          # same digest once substituted

They are different blobs before the copy and identical after it, so a match is the check and a
mismatch means the copy or the `dracut` did not happen. The two are the same SDMA 5.0 format and
size, and differ in `ucode_version`, `0x34` on the board's own against `0x2c` on navi12's, and in
17988 of their 33792 bytes
([`logs/sdma-firmware-2026-08-19/identity-2026-08-26/`](logs/sdma-firmware-2026-08-19/identity-2026-08-26/)). Credit for this goes to
[GabriWar](https://github.com/GabriWar/bc250-rocm-working).

If you would rather not touch firmware, `export HSA_ENABLE_SDMA=0` avoids the path instead. Both
work, and for llama.cpp inference neither is measurably faster. Which is better depends on
transfer size: with SDMA working it is about 13 percent faster just above the 16384-byte threshold
and 10 percent at 256 KiB, indistinguishable below it (ROCclr uses a blit kernel there regardless),
and four times slower at 16 MiB, where the blit path sustains about 110 GB/s against SDMA's 30
([`logs/sdma-sizes-2026-08-19/`](logs/sdma-sizes-2026-08-19/)).

**Set one environment variable for HIP processes:**

    export GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32

This is required for correctness on models carrying F16 weights.

**4. Build a native gfx1013 rocBLAS** with
[`scripts/build_rocblas_gfx1013.sh`](scripts/build_rocblas_gfx1013.sh) and put it on
`LD_LIBRARY_PATH`. Expect this step to need work: the script is a worked example, not an
installer, and it stops partway so that gfx1013 can be added by hand to Tensile's ISA tables and to
the Tensile and rocBLAS C++ enums. An attempt to repeat the build here did not complete, and the
surviving copy of those hand edits does not compile, so treat this as the least reproducible part
of the recipe
([`logs/rocblas-rebuild-attempt-2026-08-20/`](logs/rocblas-rebuild-attempt-2026-08-20/)). The system
rocBLAS has no gfx1013 code objects, only symlinks to the gfx1010
ones, and GEMMs abort against it. Small quantized models avoid rocBLAS entirely, so this step is
easy to think you got away with skipping.

With Fedora 43's ROCm 6.4.2 toolchain the resulting library links broken half-precision conversion
helpers from the compiler-rt builtins archive, which zeroes fp16 GEMMs at random
([`logs/fp16-root-cause-2026-09-15/`](logs/fp16-root-cause-2026-09-15/)). Either keep step 3's
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, or repair a copy of the built library and point
`LD_LIBRARY_PATH` at it:

    python3 scripts/fix_half_helpers.py librocblas.so.4 fixed/librocblas.so.4

The script only touches helpers that match the broken variant and needs a CPU with F16C. The
llama.cpp HIP backend carries the same pair; either run the script on `libggml-hip.so` too, or build
llama.cpp with `-DCMAKE_HIP_FLAGS=-mf16c`, which removes the calls at no measured cost
([`logs/fp16-mf16c-2026-09-15/`](logs/fp16-mf16c-2026-09-15/)).

Confirm the native library is the one actually loaded instead of assuming `LD_LIBRARY_PATH` won:
start a run, then read the process map. Anything built with `RPATH $ORIGIN` and a bundled copy,
which is how the PyTorch wheel ships, will load its own regardless of the path.

    grep -o '/[^ ]*librocblas[^ ]*' /proc/<pid>/maps | sort -u

**5. For llama.cpp, apply the three patches** in [`patches/llamacpp/`](patches/llamacpp/): the
`prop.integrated` counter-patch, the KQV precision request, and the gfx1013 entry in the RDNA1
macro. Everything here is measured at master 7ba604f (2026-08-09); rechecked against upstream
master ee4c505 on 2026-08-19, 174 commits later, all three sites are unchanged and all three
patches still apply cleanly. Without the first, every number the board produces is wrong while
looking plausible.

Upstream has since caught up with the first. Master from 8 September 2026 onward (PR #28604)
hard-codes the same decision, so on current master only the second and third are needed, and both
still are: rechecked at master `bfdc321` on 14 September, removing the macro entry garbles
generation and cuts prefill from 793 to 139 t/s, and removing the KQV request makes perplexity
read 88 to 89 on three runs of four where 9.9850 is correct. `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`
happens to mask the KQV defect as well. Master also renamed `--no-mmap` to `--load-mode none`, and
its perplexity values differ from 7ba604f's in the third decimal on both backends
([`logs/llamacpp-master-recheck-2026-09-14/`](logs/llamacpp-master-recheck-2026-09-14/)).

**6. Verify.** Run [`reproduce.sh`](reproduce.sh) from the repository root; it reports the CU count,
which should read 40. Then gate real work on **both** perplexity and generated text, because they
catch different faults: with the RDNA1 macro entry missing, perplexity reads a healthy-looking
8.9425 against a correct 8.9442 while generation returns `The???????????????????????`.

### llama.cpp inference

![ROCm against Vulkan across six models](figures/fig-rocm-vs-vulkan.png)

Tokens per second, same build on both backends, every row gated on perplexity under a matching
configuration. The build is verified: every log behind this table reports `build: 7ba604f (1)`. This used
to say "same build and boot", which the timestamps do not support. The Vulkan figures were taken on
12 August around 20:15 and four of the five HIP rows on 13 August around 09:25, thirteen hours
apart, so whether the board stayed up between them is not recorded either way. The last column is
HIP decode as a share of Vulkan decode on the same row, so
100 percent means the two backends tie. The column is not a bandwidth figure: this document also carries a
share-of-the-402-GiB/s-ceiling column elsewhere, and the two are different quantities.

| model | HIP pp512 | VK pp512 | HIP tg64 | VK tg64 | decode share |
|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 805.6 | 1842.2 | 113.5 | 211.0 | 54 percent |
| qwen3-8B Q8_0 | 241.0 | 401.1 | 39.2 | 39.1 | 100 percent |
| deepseek-r1-14B Q4_K_M | 95.4 | 199.0 | 20.3 | 34.5 | 59 percent |
| qwen3-14B Q4_K_M | 97.4 | 202.8 | 21.5 | 34.2 | 63 percent |
| qwen3.6-35B-A3B MoE IQ2_M | 287.6 | 455.4 | 34.3 | 86.5 | 40 percent |

The figure carries a sixth model the table does not, qwen3.8-27B UD-IQ3_XXS at 69.23 against 97.94
prefill and 7.84 against 17.18 decode, measured on 17 August and not in the 12 August campaign
and at tg128 rather than tg64 ([`logs/qwen38-2026-08-17/`](logs/qwen38-2026-08-17/)). It is kept
out of the table because the table is that campaign.

Decode is competitive, and closest on the 8B, where both backends are working against the same
memory-bandwidth limit at roughly three quarters to four fifths of the measured ceiling. Prefill
trails Vulkan by 1.4x to 2.3x, narrowing as models grow. That gap is between llama.cpp's quantized
matmul kernels and Vulkan's, not a library deficiency: tracing shows zero rocBLAS calls on the
ROCm side at pp512.

Each rate above is a single `llama-bench` invocation, and that matters more than it looks.
`llama-bench` prints an error bar computed across repetitions *within* one invocation, which for
the 8B understates the spread across invocations by about threefold. Pooling every independent
measurement of 8B decode on record gives mean 37.34 with sd 1.20 over eleven runs, against the
39.2 in the table, so the row above is a good sample, not a tight figure. An earlier
revision of this page called that row parity with Vulkan; on the pooled numbers ROCm is nearer 95
percent of Vulkan there, which is still much closer than on any other model tested. Read every
rate here to a few percent and not to the second decimal.

**Correctness.** Wikitext perplexity, same model, command and boot, context 2048 over eight
chunks, flash attention on. The eight runs behind this table were taken within a fifteen-minute
window, so "same boot" holds here and is checkable from the timestamps.

One thing this table is not: the gate that preceded the rates above. This is a stronger
re-gate, eight chunks on both backends, taken after every rate in the throughput table. The gates contemporaneous with the rates are the
two-chunk HIP runs of 12 August in
[`logs/bench-fixed-2026-08/`](logs/bench-fixed-2026-08/), and for the four models re-measured on
13 August the nearest matching-configuration gate is about thirteen hours earlier, not in
the same session. Every model is gated; the ordering was tidier in the telling than in the run.
Perplexity depends on how much text is evaluated, so figures here are
only comparable at the same chunk count; elsewhere in this repository the 8B is often gated over
two chunks instead, which gives 9.0975 instead of 7.3503:

| model | ROCm/HIP | Vulkan |
|---|---|---|
| qwen3-8B Q8_0 | 7.3503 +/- 0.232 | 7.3792 +/- 0.233 |
| qwen3-14B Q4_K_M | 6.3970 +/- 0.193 | 6.4548 +/- 0.195 |
| deepseek-r1-14B Q4_K_M | 6.0013 +/- 0.173 | 6.0416 +/- 0.175 |
| qwen3.6-35B-A3B MoE IQ2_M | 5.1887 +/- 0.134 | 5.2041 +/- 0.134 |

Every pair agrees far inside one standard error, and all eight values reproduce exactly when the
campaign is re-run on the configuration this page now recommends, after the microcode substitution,
the kernel change and `amdgpu.gpu_recovery=0`. The throughput table above reproduces too, eighteen
of its twenty cells within 3 percent and most within one; the two that move further are decode on
the 8B and on the MoE, both inside the spread already documented for them
([`logs/campaign-current-2026-08-22/`](logs/campaign-current-2026-08-22/)). Four endurance soaks
each returned a bit-identical gate value on every round:
8h ([`logs/soak-2026-08-13/`](logs/soak-2026-08-13/)), 8.1h
([`logs/soak-large-2026-08-14/`](logs/soak-large-2026-08-14/)), 8h04m
([`logs/loose-ends-2026-08-18/soak/`](logs/loose-ends-2026-08-18/soak/), whose log runs 07:32:28 to
15:36:32), and a fourth that ran 253 rounds with SDMA alternating
([`logs/soak-crash-2026-08-20/`](logs/soak-crash-2026-08-20/)). Read the
fault counts more carefully than the gate values. The first three counted faults from dmesg alone,
which misses the class the runtime reports; and dmesg cannot see a fault from a boot that ended in
a reset, since after a reboot it reports the boot that follows. The fourth soak ended in exactly
that way, and its kernel trail survives only in the persistent journal. The bit-identical gate values are
unaffected, since those are read from the benchmark's own output.

That agreement is not an artifact of the one text it was established on. Repeated on a different
slice of wikitext and on concatenated C++ source, a large distribution shift, the two backends stay
within 0.06 to 0.72 percent of each other on both models tested. Measuring decode through a second
instrument agrees too: `llama-cli` reports 115.00 t/s where `llama-bench` reports 115.40 on the same
model and configuration
([`logs/corpus-instrument-2026-08-18/`](logs/corpus-instrument-2026-08-18/)).

**Context ceilings for decode at depth**, tokens per second, `fails` meaning the board refuses:

| context | 1.5B Q4_K (1.04 GiB) | 8B Q8_0 (8.24 GiB) | 14B Q4_K (8.63 GiB) | 27B IQ3_XXS (11.09 GiB) |
|---|---|---|---|---|
| 8192 | 84.8 | 22.6 to 23.9 | 12.6 | 7.0 |
| 16384 | 73.6 | 16.2 to 18.7 | 7.3 | fails |
| 32768 | 62.7 | fails | fails | fails |
| 131072 | 27.6 | | | |
| 262144 | fails | | | |

Every failure is memory rather than a defect, and dmesg recorded zero GPU faults across the
campaign. That last clause turned out to mean almost nothing, and it took two corrections to see
why. One: the ROCr runtime reports memory access faults the kernel log does not carry, and
twenty-eight repeats of the deepest 8B measurement produced one such fault visible only in the process
output ([`logs/deep-decode-faults-2026-08-20/`](logs/deep-decode-faults-2026-08-20/)). The second
is worse, because it was a mistake, not a limitation. That run was described here as having
dmesg clean while the board rebooted minutes later. A dmesg check after a reboot reports the boot
that followed, so it could not have seen the fault: the buffer was empty because it was new. The
persistent journal, asked the same question, found five fatal GPU resets across the twenty boots it
held at the time, a count that later sweeps cannot reproduce because the journal rotates
([`logs/journal-retro-2026-08-20/`](logs/journal-retro-2026-08-20/)). Read fault counts here
as "nothing the instrument could see", and check which instrument was used. Checking which
instrument turned out not to be enough either: most harnesses here count with a pattern that
matches neither fault signature the current stack produces, which the one captured journal of a
real fault shows directly
([`logs/fault-usability-2026-08-24/`](logs/fault-usability-2026-08-24/)). Prefill reaches
further than decode on the largest model (the 27B
processes a 16384-token prompt at 41.4 t/s but cannot generate at that depth), because generation
needs the whole cache resident alongside the weights.

### GPGPU

![SGEMM throughput against problem size](figures/fig-sgemm-curve.png)

**rocBLAS SGEMM**, native gfx1013 build, twenty iterations per size, every result checked against a
CPU reference:

| N | median ms per GEMM | GFLOP/s |
|---|---|---|
| 512 | 0.2 | about 1340 |
| 1024 | 0.8 | about 2680 |
| 2048 | 5.7 | about 3010 |
| 4096 | 30.0 | about 4580 |
| 8192 | 236.0 | about 4660 |

Every figure in this section was re-measured on the current configuration and holds:
bandwidth to 0.09 percent across three runs, DGEMM at 456 GFLOP/s, and the context ceilings on the
1.5B to the second decimal
([`logs/frontpage-verify-2026-08-22/`](logs/frontpage-verify-2026-08-22/)). The same claims were run
again later, after a week of instrumented driver builds going on and off this board, and that run
supports less than the first: DGEMM and the streaming reads reproduce, the
custom-kernel line is not in its capture at all, and the bandwidth came out 430.3 GB/s rather
than the 432 quoted here, which is 0.4 percent low and wider than the spread within either
session ([`logs/rocm-only-verify-2026-08-25/`](logs/rocm-only-verify-2026-08-25/)). The 432 rests
on the three shipped runs of [`logs/membw-2026-08-19/`](logs/membw-2026-08-19/); expect about 430
to 432 from a fresh run and not a repeat to one decimal.

About 61 percent of a 7.68 TFLOP/s FP32 peak, from an untuned Tensile build. That denominator is
what the clock and lane count suggest, not what the machine does: measured later with dependent FMA
chains and the clock verified at its cap, this part sustains 6.52 TFLOP/s, against which the same
4660 is 71 percent ([`logs/alu-rates-recheck-2026-09-25/`](logs/alu-rates-recheck-2026-09-25/)). So there is real headroom in rocBLAS here, about 30 percent. Re-measured on the
current configuration, the three large sizes reproduce within a couple of percent; the small ones
need the first call excluded, since one cold GEMM at N=512 costs 6.8 ms against about 0.16 ms warm
and swamps a twenty-iteration average
([`logs/defects-recheck-2026-08-22/`](logs/defects-recheck-2026-08-22/)). FP64 DGEMM reaches 456 GFLOP/s, quoted here as about 95 percent of its rate
peak, which is the same spec-derived denominator, 7.68 divided by sixteen. Measured streaming-read memory bandwidth is 432 GB/s (402
GiB/s), reproducing to 0.12 percent across three runs
([`logs/membw-2026-08-19/`](logs/membw-2026-08-19/)).

**PyTorch** works when built from source for gfx1013 (`PYTORCH_ROCM_ARCH=gfx1013`): 11 of 11
operations in the probe including fp16 matmul, and a 50-step training loop whose losses track a
CPU reference to within 1.799e-05 at every step, returning an identical final loss of 0.00048 on
all fourteen runs of an eight-hour soak. Accumulated parameter difference after those fifty steps
is 9.312e-03, past the 1e-3 threshold the script itself checks, so its built-in verdict reads as
disagreement; that is drift between two backends, not a defect, and the per-step loss
agreement is why ([`logs/torch-train-2026-08-19/`](logs/torch-train-2026-08-19/)).

The stock wheel is not usable on this board at all. Installed fresh, `torch 2.9.1+rocm6.4` aborts
at the first library-dispatched operation, shipping no Tensile library for gfx1013 or any gfx101x.
Copying the native kernels into it stops the abort and reaches only 1 of 11, and it cannot be done
through `LD_LIBRARY_PATH` in any case, since `torch/lib` carries `RPATH $ORIGIN` and the bundled
library wins. Build from source. Build notes and the distribution-ROCm fixes are in
[`patches/pytorch/`](patches/pytorch/); two earlier figures for the stock wheel, and why both were
wrong, are in [`logs/torch-pristine-2026-08-20/`](logs/torch-pristine-2026-08-20/).

### Known defects

| defect | fix or workaround | status |
|---|---|---|
| PASID TLB flush covers nothing under hardware scheduling: silent wrong results, KIQ freeze | `amdgpu.bc250_flush_pasid_kiq=0` | fixed here, not upstream |
| Software-scheduler eviction path wedges sustained compute | do not set `amdgpu.sched_policy=2` | understood; 2x2 factorial at both CU counts |
| Allocation reuse on the KFD SVM paths faults after free and realloc | `amdgpu.bc250_flush_by_runlist=3` | fixed here; costs less than run-to-run noise. Lighter replacements tested and rejected: MMIO and SDMA invalidation of the assigned VMID (the request latches, the ACK never sets), rewriting its page-table base, and a rebuild filtered to one PASID ([`logs/tlb-alt-2026-09-15/`](logs/tlb-alt-2026-09-15/)) |
| rocBLAS ships no gfx1013 code objects | native build (PR #8838 approach) | fixed by building. Whether the pull request has landed is a question for the pull request, not for this table |
| PyTorch ships no gfx1013 code objects | build with `PYTORCH_ROCM_ARCH=gfx1013` | fixed by building |
| llama.cpp `prop.integrated` regression produces plausible-looking wrong output | [`patches/llamacpp/0001-hip-integrated-false.patch`](patches/llamacpp/0001-hip-integrated-false.patch) | bisected to c7d8722; the bisect's own output was not kept, so what is captured is the effect at that code line ([A/B/A](logs/integrated-remeasure-2026-08-18/)) instead of the search |
| llama.cpp KQV fp16 accumulation corrupts batched attention, worse with context but present at 1024 | [`patches/llamacpp/0002-kqv-f32-precision.patch`](patches/llamacpp/0002-kqv-f32-precision.patch) | fixed here, not upstream |
| gfx1013 missing from llama.cpp's RDNA1 macro: garbled generation, and quantized matmul much slower | [`patches/llamacpp/0003-gfx1013-rdna1-macro.patch`](patches/llamacpp/0003-gfx1013-rdna1-macro.patch) | fixed here, not upstream. The garbled output is captured on both arms ([`logs/macro-remeasure-2026-08-18/`](logs/macro-remeasure-2026-08-18/)); the speed figure is not, see INVESTIGATION.md |
| llama.cpp flash-attention tile kernel spills 569 registers on RDNA1 (no `v_dot2_f32_f16`, so `ggml_cuda_mad` unpacks to float), 7x slower than Vulkan | [`patches/llamacpp/0004-rdna1-fattn-tile-no-spill.patch`](patches/llamacpp/0004-rdna1-fattn-tile-no-spill.patch) | fixed here, not upstream: an RDNA1 tile table for D=128, 256 and 512 and a dispatch check for head ratios without a power-of-two divisor; kernel at parity with Vulkan, prefill +17 percent at pp512 and +40 at pp2048 ([`logs/rdna1-fattn-spill-2026-09-17/`](logs/rdna1-fattn-spill-2026-09-17/)) |
| llama.cpp matrix-vector kernel launches RDNA1 with the generic table (four warps and a barrier per row) and emulates every int8 dot | [`patches/llamacpp/0005-rdna1-mmvq-table-and-sums.patch`](patches/llamacpp/0005-rdna1-mmvq-table-and-sums.patch) | fixed here, not upstream: a type-aware RDNA1 entry with a long-K row, `v_sad_u8` for the activation sums, and a float-activation q4_K kernel; decode 1.5B 113 to 168, 14B 20 to 29, MoE 33 to 54 ([`logs/rdna1-mmvq-2026-09-18/`](logs/rdna1-mmvq-2026-09-18/)) |
| HIP graph instantiation fails past a primed depth of 12000 on the 14B | `GGML_CUDA_DISABLE_GRAPHS=1` | workaround is not throughput-neutral, see below. The depth, the failing call and the model are stated more precisely in this repository than its captures support; what is captured is one failing primed-depth run on deepseek-r1-14B and no run of that configuration with the flag set (see INVESTIGATION.md) |
| fp16 cuBLAS path returns an all-zero layer-0 value projection ([root cause](logs/fp16-root-cause-2026-09-15/)) | `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, or repair the libraries with [`scripts/fix_half_helpers.py`](scripts/fix_half_helpers.py) | **root cause found, fixed locally, not upstream.** Not a gfx1013 kernel defect: Fedora 43's ROCm compiler-rt builtins archive (`rocm-clang-runtime-devel-19-14.rocm6.4.2`) carries `__extendhfsf2`/`__truncsfhf2` built for an integer-register convention, while ROCm clang 19 passes half values in `%xmm0`. The native rocBLAS links them in, so converting alpha returns whatever was left in a register; when that is zero, rocBLAS hands Tensile a K=0 problem and the GEMM returns zeros. Replacing the two helpers with F16C instructions in a copy of the library gives 9.1117 on qwen3-8B four runs of four (f32 9.0975) and 7.7645 on qwen3-14B against 16.7385 unrepaired. The llama.cpp HIP backend built with the same toolchain carries the same pair, which makes its f16 CLAMP fail against the CPU (3 of 6 op tests pass). With both repaired libraries on the path the default fp16 path needs no environment variable (9.1117 and 7.7645 again). llama.cpp built with `-mf16c` has no helper calls and gives the same results at the same speed ([`logs/fp16-mf16c-2026-09-15/`](logs/fp16-mf16c-2026-09-15/)); a rocBLAS rebuilt with it, or with a correct archive, was not tried. History of the hunt, including the gfx1010 result and the trace that hid it, in [INVESTIGATION.md](INVESTIGATION.md) and [`logs/fp16-scalar-2026-09-15/`](logs/fp16-scalar-2026-09-15/) |
| SDMA never completes a copy above 16384 bytes ([evidence](logs/sdma-firmware-2026-08-19/)) | substitute the navi12 microcode, or `HSA_ENABLE_SDMA=0` | **fixed.** It was the wrong microcode, not the board: with navi12's, every size from 4 KiB to 2 GiB completes and the gates stay bit-identical |
| A GPU reset reports success without resetting anything, then hangs the host reinitialising live hardware ([evidence](logs/reset-smu-gc-2026-09-14/)) | `amdgpu.gpu_recovery=0`, **confirmed**: calling the KFD reset path directly returns harmlessly with the parameter set and kills the board without it ([`logs/kfd-reset-probe-2026-08-22/`](logs/kfd-reset-probe-2026-08-22/)) | **open, and the most serious defect here.** Asking the driver to reset the device kills the board even with the GPU idle, twice out of two. The parameter stops the driver asking, which converts a hung machine into one you can reboot deliberately; it does not keep the GPU usable, and recovery is a reboot. Do not reset the device by hand. Where the resume stalls, what has been ruled out and why the stopping point moves between identical runs are in [INVESTIGATION.md](INVESTIGATION.md) |

Evidence for every row is under [`logs/`](logs/), and each defect is worked through in
[INVESTIGATION.md](INVESTIGATION.md); the three open ones are linked directly above.

Two notes on the workarounds. `GGML_CUDA_DISABLE_GRAPHS=1` is not throughput-neutral: on the 8B at
depth 16128, measured in ABBA order, disabling capture is about 13 percent faster. And
`HSA_ENABLE_SDMA=0` costs nothing measurable for inference, nor does fixing SDMA properly, which
moves 8B decode by 0.1 percent and leaves Vulkan unchanged. SDMA is not neutral in general though:
it is faster in a band of medium transfers and four times slower at 16 MiB. Both figures were
measured more than once and the history of getting them wrong is in
[INVESTIGATION.md](INVESTIGATION.md).

### Limits

- One board, one stack. Nothing here says another BC-250 behaves the same way.
- Decode at depth varies run to run by as much as 15 percent on some models and under 1 percent on
  others, in one boot, with the clock pinned (residency at 1500 MHz is 83 to 85 percent in every
  run) and temperature and memory flat. The cause is not established. A memory-bandwidth
  explanation was the working theory and is not supported: measured with model order rotated, the
  coefficients of variation are 0.7, 6.0 and 4.6 percent at 29, 46 and 80 percent of the measured 432 GB/s bandwidth
  ceiling, so variability does not rise with utilisation.
- Effects of a few percent are hard to establish on this board at all. Two careful designs of the
  same graph-capture comparison returned differences of opposite sign, and only the counterbalanced
  one is trustworthy. Treat any small difference here, including ones stated above, as needing a
  counterbalanced repeat before it means anything.
- The wedge and freeze behaviour that dominated earlier work is routed around by this
  configuration rather than repaired, and its root cause is not known.
- One defect this document called board-genuine for weeks turned out to be the wrong microcode.
  That is worth holding on to when reading the remaining two: "we could not fix it" is not
  evidence about hardware.
- Every measurement in this repository taken before the microcode substitution used
  `HSA_ENABLE_SDMA=0`, because
  until then SDMA could not complete a transfer at all. That makes it a constant behind almost
  everything here, not a setting that was chosen, and now that the microcode substitution
  makes the other value usable, the standard this document argues for says to re-test it rather
  than assume it stayed neutral. Re-tested and unchanged so far: throughput, the correctness gates
  on two models, Vulkan, the allocation-churn sweep and decode at depth, and an eight-hour soak
  alternating it round by round, which returned `8.9442` on all 253 completed rounds (and returns
  it again today, [`logs/gate-verify-2026-08-25/`](logs/gate-verify-2026-08-25/))
  ([`logs/soak-crash-2026-08-20/`](logs/soak-crash-2026-08-20/)). That soak ended in a board reset
  on round 254, described in the defect table above; roughly half its rounds had SDMA enabled, so
  the crash is not evidence against the substitution, and the same failure appears on boots
  predating the microcode change.
- The context ceilings above were measured with the model mmapped, which is `llama-bench`'s
  default. Loading without mmap costs usable context: the 8B decodes at a primed depth of 14336
  but aborts at 16128, ten attempts out of ten, because the weights then sit in anonymous memory
  that cannot be reclaimed. Measured through both arms, the no-mmap run peaks at 14594 MiB with
  601 MiB left while the mmap run peaks at 11532 MiB with 3663 MiB left
  ([`logs/nommap-ceiling-2026-08-21/`](logs/nommap-ceiling-2026-08-21/)).
- Numbers quoted from a single benchmark invocation carry more uncertainty than the printed error
  bar suggests, as noted above.
- The PyTorch figures cannot currently be re-measured on this board. The environment that produced
  them, a source build for gfx1013, is no longer installed: the board now carries a
  stock wheel whose architecture list does not include this chip, so reproducing that section means
  rebuilding first. The build tree and the probes are still present. (No longer true as of 24
  September: a gfx1013 build is installed again in `~/torchbench-venv`. This snapshot is left as it
  stood, as the heading says.)
- The board boots with `mitigations=off`, which is worth naming since it flatters every CPU-side
  comparison here. It does not
  affect the GPU figures, but every CPU-side number quoted for comparison was taken with CPU
  speculative-execution mitigations disabled, which flatters the host: the PyTorch CPU reference of
  about 20 seconds against the GPU's 0.26 is the case where this matters most.
- What a fault under `amdgpu.gpu_recovery=0` leaves behind has now been measured, and it is worse than
  the parameter's name suggests. A natural fault arrived after
  about 190 rounds: the usual chain, page fault to preemption failure to runlist rebuild `-62`, and
  no reset at all, exactly as the parameter promises. The host stayed up. But the GPU did not come
  back. Every GPU process for the next hour died, seven of them, the last segfaulting inside
  `libamdhip64`, while the driver logged 1193 preemption failures. Worse, the wedge is not visible
  to any ordinary check: the device stays on the bus, the runtime still enumerates it and still
  reports 14 GiB free. Only actually running something fails
  ([`logs/fault-usability-2026-08-24/`](logs/fault-usability-2026-08-24/)). The parameter is still
  worth setting, since a live host is recoverable by a reboot when you choose and a hung one is not,
  but it buys a clean shutdown and not continued service. An ordinary reboot restores the board
  fully, so the wedge is a state, not damage. That is one fault, observed once; whether every
  fault leaves the machine in the same state is untested.

### Reproducing

[`reproduce.sh`](reproduce.sh) builds and runs the probes: rocBLAS code objects and SGEMM against
the system library, the override and the native build; compute correctness through the graphics
queue and the compute queue; and a CU-count check. It gates on the module and scheduler
configuration and explains what is wrong instead of producing misleading output. Run from a clean
copy of this repository on the board it passes every stage, and the output is kept in
[`logs/reproduce-verify-2026-08-19/`](logs/reproduce-verify-2026-08-19/) so a reader can see what
passing looks like before running it. It was re-run against the configuration this page now
recommends, kernel 7.1.8 with the navi12 microcode and `amdgpu.gpu_recovery=0`, and every stage
still passes ([`logs/reproduce-verify-2026-08-22/`](logs/reproduce-verify-2026-08-22/)), and again
after a week in which a dozen instrumented driver builds went on and off this board, with the same
result ([`logs/reproduce-verify-2026-08-25/`](logs/reproduce-verify-2026-08-25/)).

The measurement harnesses are in [`scripts/`](scripts/), one per experiment, each with a header
saying what question it was written to answer. Raw output is in [`logs/`](logs/), one directory per
run with a README naming the harness that produced it.

### How this was arrived at

[INVESTIGATION.md](INVESTIGATION.md) is the full account: how each defect was found and what the measurements were. One control matters more than any single result and is worth stating here: compare conditions that differ in one way only. Most of the difficulty on this board has come from conditions differing in more than one, usually because a workaround adopted early had quietly become part of the apparatus. Everything that
survived came from an intervention on a single variable: a module parameter toggled live within one
boot, a two-by-two factorial, one source line reverted and restored, a git bisect, a byte-level
bracket.

A second class of error showed up later and is worth separating from the first, because no amount
of care about experimental design catches it. Several claims here rested on an instrument that
could not have detected what it was being used to rule out. Fault counts came from `dmesg` after
the board had rebooted, when it necessarily reports the boot that followed the crash. A search for a
quoted figure matched it inside an unrelated longer number, and matched others only in prose written
by the author, which made the check circular. Each time the
instrument returned nothing and the nothing was reported as evidence. The cheap defence is to ask,
before believing a negative result, what a positive one would have looked like and whether the
instrument could have produced it.

## Fedora 44 and 45: ROCm 7.1.1 and 7.2.2

Fedora 44 was tried on 15 September 2026 for one reason: its ROCm toolchain does not carry the broken
half-precision helpers behind the zeroed fp16 GEMM ([`logs/fp16-root-cause-2026-09-15/`](logs/fp16-root-cause-2026-09-15/)).
It became the board's default the same day.

**How it was installed without risking Fedora 43.** The btrfs `root` subvolume was snapshotted to
`root-f44` and the copy upgraded with `dnf --installroot --releasever=44 distro-sync`, excluding
kernels. The copy then boots through its own boot entry on the unchanged 7.1.8 kernel and amdgpu
module, while Fedora 43 stays one boot entry away. Swap had to be off for the snapshot, since btrfs
refuses to snapshot a subvolume holding an active swapfile, and the copy's SDMA microcode had to be
substituted again, since `linux-firmware` came back stock. After a relabel with `restorecon` and a
permissive boot with zero AVC denials, including during GPU work, the entry runs enforcing.

**Out of the box it runs no model** ([`logs/fedora44-rocm711-2026-09-15/`](logs/fedora44-rocm711-2026-09-15/)).
Two separate problems:

- The system rocBLAS 7.1.1 has no gfx1013 files. Symlinked to gfx1010, every GEMM fails with
  `CUBLAS_STATUS_INTERNAL_ERROR`, f32 included. The fix is a native build.
  [`scripts/apply_gfx1013_rocblas711.py`](scripts/apply_gfx1013_rocblas711.py) adds gfx1013 to rocBLAS
  and Tensile at `rocm-libraries` tag `rocm-7.1.1`, in the same nine places as PR #8838 and Fedora's
  own Tensile patches for newer RDNA parts. Every edit is anchored and verified, which matters
  because the hand edits behind the Fedora 43 build were lost and never reproduced. The build needed
  three environment fixes, all in the build chroot: `rocm-cmake` and `rocminfo` installed, a msgpack
  CMake shim defining the target name Tensile 4.44 expects, and `/dev/shm` mounted. Without
  `/dev/shm`, joblib runs Tensile's parallel map in-process, where `OverwriteGlobalParameters` clears
  the global dictionary and refills it from itself, empty. It took about 50 minutes.
- llama.cpp's flash attention stops on `GGML_ASSERT(max_blocks_per_sm > 0)`. HIP 7.1 reads each device's
  VGPR budget from comgr's ISA metadata table instead of hard-coding it as 6.4.2 did, and in ROCm 7.0
  to 7.1.1 every gfx10 row says 256 total VGPRs instead of 1024. The occupancy calculation then
  truncates to zero for the tile kernel. ROCm/llvm-project commit 4f5ae331f659, first in rocm-7.2.0,
  corrects the 13 rows; [`scripts/fix_comgr_gfx10_vgprs.py`](scripts/fix_comgr_gfx10_vgprs.py) makes
  the same 13-byte change in a copy of `libamd_comgr.so.3`. A clamp in llama.cpp was proposed upstream
  (PR #27787) and declined as treating the symptom, which the table bug explains.

**With both fixes it is correct; the speed gain was an artefact.** The paragraphs below were written from
runs taken before 16 September, when the GPU clock policy on the Fedora 44 side turned out to be different
from Fedora 43's. The upgrade had reinstalled `oberon-governor` and replaced `/etc/oberon-config.yaml` with
the package default, which allows 2000 MHz at a fixed 1000 mV; the board overheats there, the governor logs
`GPU overheated, throttling`, and the clock oscillates between 1000 and 2000 MHz. Re-measured with Fedora
43's configuration restored, every Fedora 44 figure lands within 2 percent of its Fedora 43 counterpart:
ROCm 1.5B prefill 792.9 against 805.6, qwen3-14B 96.8 against 97.4, Vulkan 1.5B 1849.1 against 1842.2,
SGEMM N=4096 30.1 ms against 30.0, DGEMM 456.2 GFLOP/s against 456. ROCm 7.1.1 is not faster than 6.4.2 on
this board, and the reasons to run Fedora 44 are the correct fp16 toolchain and the scripted rocBLAS build.
The clock also explains the Vulkan readings above: with it pinned, qwen3-14B Vulkan prefill reads 204.65
+/- 0.02 rather than swinging between 145 and 269
([`logs/fedora44-benchmarks-2026-09-15/clock-corrected/`](logs/fedora44-benchmarks-2026-09-15/clock-corrected/)).

The original text of that comparison follows, since the correctness results in it stand.

**With both fixes it is correct and faster** ([`logs/fedora44-working-2026-09-15/`](logs/fedora44-working-2026-09-15/)).
Every gate matches the corrected Fedora 43 stack exactly: 8.9442, 9.1117 on the default fp16 path
including under the allocator setting that used to force the zeroed GEMM, 9.0975 in f32, 7.7645 on the
14B, 8.1702 with flash attention off, and 12801 of 12801 op tests. Alternated across four boots,
qwen2.5-1.5B prefill rises from 810 to 996 t/s and decode from 114 to 146, and qwen3-8B prefill from
243 to 307 with decode essentially level, 39.4 to 40.4. A Fedora 43 boot with SELinux permissive
matches Fedora 43 enforcing, so the mode is not the cause.

**Where the speed comes from is split, not isolated to one component**
([`logs/fedora44-validation-2026-09-15/`](logs/fedora44-validation-2026-09-15/)). GPU-bound prefill,
2048 tokens in one ubatch, is 16 percent faster (868 against 746 t/s), and prefill with flash attention
off 31 percent, so device code runs faster. That points at clang 20 against 19 or the HIP 7.1 launch
path. Host overhead also fell sharply: a whole 256-token decode run, load included, takes 2.6 s wall,
2.3 s user and 0.4 s system on Fedora 44, against 7.7 s, 5.1 s and 4.5 s on Fedora 43. HIP graph
capture is not the difference, since disabling it leaves the decode gap at 26 percent. The two
compilers cannot be swapped between the two runtimes, so the split between compiler and runtime is
not measured.

**Vulkan moved too, and the obvious explanation is wrong.** On Fedora 44 the 1.5B prefills at 2417 t/s
against 1844 on Fedora 43, while qwen3-14B falls from about 199 to about 148 and the 8B becomes bimodal,
its prefill samples clustering at 267, 317 and 373 to 385. Mesa went from 25.3.4 to 26.1.8 between the two
systems, so the driver was the suspect. It is not: Fedora 43's Mesa 25.3.4, extracted from its rpm and
loaded on Fedora 44 through `VK_ICD_FILENAMES` with the two libraries Fedora 44 dropped copied beside it,
measures the same as Fedora 44's own driver on all four models, bimodal prefill included, and the two
gates agree to the fourth decimal. Nor is it the llama.cpp binary: the Fedora 43 Vulkan build and the
Fedora 44 one give the same figures on Fedora 44 against either driver. The driver configuration files are
identical apart from game entries, and the kernel, module and parameters are the same by construction.
What does explain it is the GPU clock policy, found the next morning: the Fedora 44 governor configuration had been
replaced by the upgrade, and with Fedora 43's restored the Vulkan figures match Fedora 43 on every model
([`logs/fedora44-validation-2026-09-15/`](logs/fedora44-validation-2026-09-15/)).

**The kernel side is unchanged, and still needed.** The allocation-churn A/B/A on Fedora 44: with
`bc250_flush_by_runlist=3` three runs of the MUL_MAT perf sweep and the sequence reproducer are clean;
at 1 both sweep runs die on `Memory access fault by GPU` with ten kernel fault lines; back at 3, three
runs clean again and the gate reads 8.9442.

**The default configuration, measured.** With the two fixes installed ahead of the system libraries
through `ld.so.conf` and no environment variables, a six-model campaign in one boot
([`logs/fedora44-benchmarks-2026-09-15/`](logs/fedora44-benchmarks-2026-09-15/)) gives every ROCm gate within
one standard error of Vulkan, and the MoE and deepseek-r1 ROCm gates identical to Fedora 43 to four
decimals. SDMA with the navi12 microcode gives the same gates as SDMA off, and memory bandwidth is
unchanged at 433.9 GB/s, as a property of the board should be.

That campaign also read ROCm prefill 23 to 29 percent above Fedora 43 on five of six models, and large
rocBLAS GEMMs 26 to 33 percent faster (SGEMM N=4096 23.2 ms against 30.0, DGEMM 605.7 GFLOP/s against
456). **None of that is real.** It is the same oscillating governor found the next morning, reaching
2000 MHz at a fixed voltage before overheating. Re-measured with the clock pinned, the two releases are
within half a percent of each other on every GEMM line and the prefill gain disappears
([`logs/fedora44-benchmarks-2026-09-15/clock-corrected/`](logs/fedora44-benchmarks-2026-09-15/clock-corrected/)).
Bandwidth was the clue that should have been taken at the time: it did not move, because it is not
clock-bound the way the GEMMs are.

**Decode at depth needs one depth per invocation.** The campaign's ladder put every depth in one
`llama-bench` invocation with mmap on, and its depth-0 entries read 82 t/s on ROCm and 157 on Vulkan against
146 and 241 in the throughput table. Re-run without mmap it was still noisy on Vulkan and at 30720 on ROCm.
One depth per invocation gives ROCm a smooth 146, 134, 124, 108, 95, 87 from 0 to 30720 tokens, 24 to 30
percent above Fedora 43 at every depth, while Vulkan on Mesa 26.1.8 still spreads from 133 to 205 t/s within
one invocation at 4096 and falls below ROCm at 30720. Ten consecutive invocations at depth 0 are tight on
both backends, so the instability is specific to depth on Vulkan and to multi-depth invocations on both.

**One test per `llama-bench` invocation, for decode as well.** Checking an odd 14B decode reading, 26.0 t/s
in one run against 21.6 in the campaign, turned up the same artefact as the depth ladder in a milder form.
With `-p 512 -n 64` in one invocation, which is how every throughput table in this repository was
measured, decode on larger models reads low and scatters: qwen3-14B 21.6 on ROCm and 33.9 on Vulkan, and
deepseek-r1-14B 21.2 +/- 4.9, against 26.7, 37.6 and 26.4 with decode in its own invocation, nine samples
each over three alternated rounds ([`logs/fedora44-benchmarks-2026-09-15/`](logs/fedora44-benchmarks-2026-09-15/)).
The 1.5B and all prefill figures are unaffected. The Fedora 43 tables were measured the combined way, so
their large-model decode figures are probably low by a similar margin; they are left as measured and the
Fedora 43 against Fedora 44 comparison uses the combined measurement on both sides. The mechanism, whether
something left over from the prefill test (allocations, graph state, memory pressure) slows the decode test
after it, was not isolated.

**Build variants.** On Fedora 44, forcing cuBLAS for every quantized matmul costs 43 percent of 8B prefill and 41
percent of MoE prefill and moves both short gates, so llama.cpp's own selection stays. Current master with
the two remaining patches matches 7ba604f on prefill and decodes qwen3-14B about 6 percent faster, 27.3 to
28.7 against 25.6 to 27.1 over four alternated pairs.

**Where the host-CPU difference comes from.** The one difference that survived the clock correction is host
CPU time, and it is the event-wait path. For the same 256-token decode at the same rate, ROCm 6.4.2 makes
127743 ioctls against 7.1.1's 4379, of which `AMDKFD_IOC_WAIT_EVENTS` accounts for 11387 against 1037 over a
shorter run; kernel time falls from 4.4 s to 0.4 s. The memory-management ioctls are the same in both, so it
is specifically how the runtime waits for completion signals. `ROC_ACTIVE_WAIT_TIMEOUT` at 0 and 100000 and
`HSA_ENABLE_INTERRUPT=0` change none of it on 6.4.2, the last only moving the cost from system to user time
([`logs/fedora44-hostoverhead-2026-09-16/`](logs/fedora44-hostoverhead-2026-09-16/)).

**The deep-context crash is the KFD system-memory budget, and a newer runtime does not fix it.** The limit
is 63/64 of RAM minus 1.5 GiB, which the driver reports as 13422 MiB on this board and which read
`13412M out of 13422M` at the moment of failure. `ttm.pages_limit` raises a different counter and XNACK is
refused for every GC 10.1.x part, so neither applies. `amdgpu.no_system_mem_limit=1` removes the crash but
the run then swaps and does not finish, so the limit is left alone. Fedora 45's ROCr 7.2.1, extracted and
loaded on Fedora 44 through `LD_LIBRARY_PATH`, runs at unchanged speed and still crashes at the same fault
address, so the fix is not a newer runtime
([`logs/fedora44-ceilings-2026-09-16/`](logs/fedora44-ceilings-2026-09-16/),
[`logs/fedora44-hostoverhead-2026-09-16/`](logs/fedora44-hostoverhead-2026-09-16/)).

**The board throttles even at the policy this repository uses.** Twenty minutes of continuous 8B prefill at
the 1500 MHz policy holds 197.7 to 198.2 t/s in twenty of twenty-two rounds, but the edge sensor reaches
94 C, 58 of 242 clock samples sit at the lower step and the governor logs one throttle event
([`logs/fedora44-thermal-2026-09-16/`](logs/fedora44-thermal-2026-09-16/)). That throttle is the mechanism behind the
advice to take medians instead of single readings.

**llama.cpp master is not faster either.** Re-measured at the corrected clock, master with the two remaining
patches prefills 0.8 to 1.1 percent slower than the measured base and decodes within 1 percent except on the
two larger models, where it is 1.6 percent faster. The 6 percent decode gain reported from the oscillating
boot was clock noise, as was the Fedora 44 speedup itself.

**PyTorch built from source works the same.** The Fedora 43 source tree rebuilt against ROCm 7.1.1 passes the
op probe 11 of 11 and reproduces the training loop's losses and parameter drift to every printed digit.

**Fedora 45 with ROCm 7.2.2 needs one fix fewer.** Tested the same way the next day, in its own snapshot on
the same kernel. The comgr correction is upstream by then, so flash attention runs on the stock packages and
the 1.5B gate reads 8.9442 with no override; the patch script confirms it, finding no rows to change in
Fedora 45's `libamd_comgr.so.3`. rocBLAS is unchanged, still shipping gfx1010 symlinks, and the gfx1013 patch
script applied to the `rocm-7.2.2` sources with every anchor matching once. With that build the three gates
and the 12801-case op suite match Fedora 44 exactly, prefill is within 0.4 percent and decode 0.5 to 2.4
percent higher, and Vulkan on Mesa 26.2.0 prefills 1.6 to 3.4 percent slower than 26.1.8. Fedora 45 was a
development release that day, so the recipe stays on Fedora 44
([`logs/fedora45-rocm722-2026-09-16/`](logs/fedora45-rocm722-2026-09-16/)).

**An upstream Vulkan patch was worth more than anything found on the ROCm side that week**, which the nine patches after it reversed. llama.cpp
[#28507](https://github.com/ggml-org/llama.cpp/pull/28507) widens flash attention's shared-memory staging
from NVIDIA-only to AMD RDNA. Applied to master `bfdc321` and measured against the same commit unpatched,
with the clock pinned and builds alternated, qwen3-8B Vulkan prefill gains 2.8 percent at depth 0, 35.3
percent at 4096 and 64.5 percent at 8192, and the perplexity gate is bit-identical. Its author measured +12,
+34 and +50 percent on their own BC-250 with a different model, so this is a second board agreeing. The
companion patch [#27332](https://github.com/ggml-org/llama.cpp/pull/27332), a routing-density gate for
mixture-of-experts matmuls, changes nothing measurable here, but `llama-bench` never enters the regime it
targets ([`logs/vulkan-fa-staging-2026-09-17/`](logs/vulkan-fa-staging-2026-09-17/)).

**The largest ROCm gain found at this point came from a register spill, and gfx1010 shows the same spill.** Replaying the
real pp2048 graph through `test-backend-ops` on both backends localised the prefill deficit
([`logs/op-perf-hip-vs-vulkan-2026-09-17/`](logs/op-perf-hip-vs-vulkan-2026-09-17/)): with `-fa off`
it is the matmuls and only the matmuls, `MUL_MAT` at a median 1.84 times Vulkan's time against 0.75
for everything else, summing to 1.74 across the graph against 1.86 measured end to end. With `-fa on`
the flash-attention kernel is 7.05 times slower, 114.7 ms against 16.3, worse than ROCm's own
non-flash path for the same work. The cause is a register spill:
`V_DOT2_F32_F16_AVAILABLE` covers RDNA2 and later but not RDNA1, correctly, since gfx1013 has no
`v_dot2_f32_f16`; without it `ggml_cuda_mad` unpacks each `half2` into two floats, and the D=128 tile
kernel overruns the 256-VGPR budget with 569 registers spilled and 2280 bytes per lane of scratch.
gfx1010 spills identically and gfx1030 and gfx1100 not at all, so this is an RDNA1 problem rather
than a BC-250 one, and the resource figures reproduce with `hipcc` alone. An RDNA1 tile table that
doubles `nthreads` and halves `nbatch_fa` removes the spill and is worth 40 percent of prefill on the
1.5B and 42 on the 8B with the final rows, with decode unchanged. At depth it is worth more, and it
removes the flash-attention trade on ROCm: 8B pp2048 with `-fa on` goes from 194, 97 and 65 to 266,
220 and 185 at depths 0, 4096 and 8192 with the final rows, ahead of `-fa off` at every depth where it
had trailed it, and at 4096 and 8192 ahead of Vulkan's own default flash-attention path. The 1.5B's
flash-attention op itself ends at 16.9 ms against Vulkan's 16.3, from 114.7.
Rows for D=256 and D=512 followed (the MoE and the 27B gain 10 and 6 percent), and the two 14B models,
whose 40-over-8 head ratio has no power-of-two divisor and so dispatches without GQA sharing, needed a
different lever: the 64-column tile of that variant cannot be made to fit, the 32-column one is clean,
and one RDNA1-gated check in the host dispatch keeps them there, for 9 to 11 percent of pp2048.
The decode side followed the same pattern: the quantized matrix-vector kernel picks its launch geometry
from a per-architecture table that RDNA1 is not in, so gfx1013 ran the generic entry, four warps per row
with a shared-memory reduction; routing it to the RDNA2 entry, one wave per row, is worth 26 percent of
decode on the 1.5B, 14 on the 14B and a third on the MoE, with the gate bit-identical; replacing the
emulated dot product that only sums activations with `v_sad_u8` adds 2 to 3 percent, also exact; the
one-wave geometry cost the Q8_0 8B 4 percent, so the entry is type-aware, four warps for the simple types
([`logs/rdna1-mmvq-2026-09-18/`](logs/rdna1-mmvq-2026-09-18/)). The emulated `v_dot4` itself went last: a
float-activation q4_K matvec, Vulkan's formulation in HIP, is worth a further 16 percent of decode on
the 1.5B and 21 on the 14B, token-identical to the int8 path
([`logs/rdna1-fattn-spill-2026-09-17/`](logs/rdna1-fattn-spill-2026-09-17/),
[`patches/llamacpp/0004-rdna1-fattn-tile-no-spill.patch`](patches/llamacpp/0004-rdna1-fattn-tile-no-spill.patch)).
Two other explanations were tested and excluded first: a tile-size cliff, ruled out because the
deficit is flat in batch size, and the wrong tile table, worth 2.9 percent.

**The segfault at the memory limit is two bugs, and the first is fixed.** The September ceilings page matched
`segfault at 20 ... in libhsa-runtime64.so.1.18.0` to rocm-systems PR #2850 by symptom. Fedora's debuginfo
names the address: `GpuAgent::ReleaseQueueMainScratch`, the scope-guard release the PR guards. The hunk
ported to Fedora's 7.1.1 source package and rebuilt removes that crash on the exact September command,
qwen3-14B at depth 16384, and exposes a second one behind it: HIP's own teardown of the queue it failed to
create writes through a null pointer in a `Command` destructor. The runtime returns the error it should;
the layer above does not yet survive it, so the rebuilt library is not installed
([`logs/rocr-queue-scratch-2026-09-18/`](logs/rocr-queue-scratch-2026-09-18/)). The decode work's own
campaigns are in [`logs/fedora44-campaign-f32mv-2026-09-18/`](logs/fedora44-campaign-f32mv-2026-09-18/)
and its successors.

**This board is no longer the only one running ROCm on gfx1013.** A repository published on 16 September
2026, `Naster17/bc250-stack`, documents an independently built native gfx1013 ROCm stack with PyTorch, using
the same kernel patch set as here and crediting both this repository's runlist-rebuild flush and GabriWar's
original. It also carries an independently derived version of the RDNA1 macro patch, with the observation
that without it flash-attention tile dispatch mismatches between host and device and produces NaN results.
This is a second party reaching the same conclusion by a different route, which is worth more than any
argument made here.

**PyTorch, Fedora's package.** Fedora's `python3-torch` 2.9.1 is a ROCm 7.1.1 build whose architecture list omits
gfx1013. With the native rocBLAS, library operations are correct (a 64x64 matmul matches the CPU to
7.6e-06) and without it they fail with `HIPBLAS_STATUS_INTERNAL_ERROR`; either way, the first operation
that needs one of torch's own kernels dies with a general protection fault inside `libamdhip64`. A
source build for gfx1013 is needed, as on Fedora 43.

## How the measurements here are controlled

The numbers in this document come from logs and re-derive from them. What has repeatedly needed care
is everything around the number: what was held fixed, what the instrument was actually reporting, and
whether the thing measured was the thing of interest. These are the controls this work ended up with, and
they are worth more than any individual result here.

**Keep an explicit inventory of what is held constant, and treat anything on it that has never been
varied as an untested assumption, not as background.** On this board the list is the scheduler
policy, the flush parameters, the CU count, the compute type, memory mapping, graph capture, the
corpus behind the correctness gates and the benchmark tool itself. All of them have now been varied
and measured. Two boot arguments belong on the same list and are easy to miss because they live on
the kernel command line and not in any script: `ttm.pages_limit=4194304`, which is load-bearing
for the large models since TTM otherwise limits itself to half of system memory, about 7.4 GiB of this
board's 14.8, and `mitigations=off`, which does not touch the GPU but flatters every CPU-side
comparison quoted here.

**Vary one thing at a time, and counterbalance, not block.** Comparisons that differ in more
than one way are where this work has gone wrong most often. Arms presented in blocks pick up drift;
arms interleaved within a session do not. Where an effect is a few percent, pair the arms and report
how many pairs went each way alongside the means.

**Take the number an instrument gives you and measure the same thing inside the real workload.** A
replayed operation against the same shape in a live graph, a microbenchmark against `llama-bench`, a
profiler's total against the token that contains it. Several instruments here have been confidently
wrong in isolation and plainly wrong the moment they were checked against the workload they claimed
to describe.

**A mechanism visible in the source is a hypothesis about what the machine does, not a measurement of
it.** Reading the code says what should happen; only an experiment says what does.

**Sample the GPU clock during any rate measurement and report its residency beside the result.** The
oberon governor idles at 1000 MHz and short kernels do not ramp it, so a rate computed over an
unramped window is low by whatever the clock was short by. This has put wrong numbers on this page
more than once, and the check costs one background loop reading `pp_dpm_sclk`.

**Check which library actually loaded, not which one exists.** `strace -e trace=openat` names the code
object a process opens. A build that is present on disk, on `LD_LIBRARY_PATH`, or newer than the
others is not necessarily the one being measured.

**Match ggml tensors by pointer, not by name.** Names are reused along a chain, so several distinct
tensors in one graph can carry the same name, and a reading keyed on names can contradict what the
addresses say.

**One `llama-bench` test per invocation.** Passing depths as a list, `-d 0,4096,8192`, reads up to 22
percent low on either backend at some points and not others. The table `llama-bench` prints also
carries only a mean and a standard deviation; `-o jsonl` exposes `samples_ts`, the per-repetition
values, which is the only way to see that the first repetition is not the same measurement as the
rest.

**Hedging is not a control.** Labelling a claim an inference rather than a demonstration is honest and
makes it cheap to set aside later, but it does not make the claim any more or less true, and it is not
a substitute for the experiment being harder to design than it looks.

## Hardware performance counters

Nothing above this section reads a hardware counter. Kernel timings come from
[`scripts/kerntrace.cpp`](scripts/kerntrace.cpp) over roctracer, arithmetic rates from `clock64()`
inside the kernel. `rocprofv3` now works on this board, and getting it there needed two unrelated
fixes, each of which fails without an error message
([`logs/hw-counters-2026-09-25/`](logs/hw-counters-2026-09-25/),
[`scripts/build_rocprof_gfx1013.sh`](scripts/build_rocprof_gfx1013.sh)).

The first is not a gfx1013 problem. Fedora's `rocm-runtime` is built without `rocprofiler-register`:
ROCR-Runtime calls `find_package(rocprofiler-register)` and silently disables the handshake when the
package is missing, and Fedora does not list it as a build dependency. `libhsa-runtime64.so` on the
board carries no `rocprofiler_register_*` symbols, so a profiler attaches, runs, and reports
`Number of services generating output: 0`. On a stock Fedora ROCm install no ROCm profiler sees
anything on any GPU. Rebuilding the runtime into `/opt` with the package present fixes it and leaves
the packaged one alone.

The second is the familiar shape. `counter_defs.yaml` lists `gfx10`, `gfx1010`, `gfx1030`, `gfx1031`
and `gfx1032` for 89 counters, and `metrics.cpp` looks the agent name up in a map keyed by those
literals, so the bare `gfx10` entry is a key and not a wildcard and gfx1013 gets nothing. gfx1011 and
gfx1012 are missing the same way. `aqlprofile`, which programs the counters, needs no patch at all:
it dispatches on a name prefix and gfx1013 already selects its generic gfx10 builder.

Whether the gfx1010 mapping is right for gfx1013 is an assumption, so
[`scripts/counter_validate.cpp`](scripts/counter_validate.cpp) tests it against a kernel whose VALU
count is fixed by its ISA. It reads 8203 instructions per wave at every launch size, the 8192 FMAs
the compiler emitted plus the eleven that set up and reduce the accumulators. For the SQ block the
mapping holds. It does not hold everywhere: `GL2C_HIT_sum` reads exactly zero on a tiled GEMM and
`FETCH_SIZE` reports a rate thirty times the board's memory bandwidth, so the L2 counters are wrong
here and are kept in that directory as the evidence.

![counter validation](figures/counter-validation.png)

Three things came out of it. rocBLAS HGEMM really does issue the packed instruction, half of SGEMM's
VALU count, and gains nothing because its shared-memory traffic does not halve with it, which closes
the question [`logs/hgemm-isa-2026-09-24/`](logs/hgemm-isa-2026-09-24/) left open. The int8 row of
[`scripts/alu_cycles.cpp`](scripts/alu_cycles.cpp) was timing the scalar unit, because its
accumulators were uniform across the wave and the compiler moved the whole dot product to SALU; the
conclusion it supported survives, the kernel is fixed, and the fp32 and fp16 rows were never affected
because RDNA1 has no scalar float ALU. And
[`scripts/pk_gemm_square.cpp`](scripts/pk_gemm_square.cpp) shipped with a tile depth that did not
match its own comment or its own published figure, so the script returned 2.31 TFLOP/s where this
repository published 8.90; with the documented depth it returns 8.90, and the default is corrected.

![an int8 benchmark that was measuring the scalar unit](figures/counter-scalarised.png)

## Open questions

Where other eyes would help most. Several entries that used to sit here have been answered; those are
listed at the bottom so the trail stays findable.

### Why a GPU page fault sometimes takes the board down

The chain was caught whole on 20 August: a gfxhub page fault on vmid 8, a queue preemption that timed
out four seconds later, the runlist rebuild returning -62, a GPU reset, and only then a SIGBUS in
`rocr::AMD::AqlQueue::StoreRelaxed` as the reset pulled the queue mapping out from under a store in
flight ([`logs/soak-crash-2026-08-20/`](logs/soak-crash-2026-08-20/)). What stays open is the first
link, not the last: why the fault happens at all, roughly once in 254 rounds of mixed load.

Provoking one deliberately does not reproduce the escalation. Four methods were tried and none of them
escalates ([`logs/fault-probe-2026-08-22/`](logs/fault-probe-2026-08-22/)); every natural event arrived
after minutes to hours of real inference. A deliberate fault on its own is survivable, so the August
chain needed the preemption timeout that followed it, and the real first link is whatever turns a
fault into a queue that will not preempt.

On 22 September a spontaneous one was caught with the mitigation in place
([`logs/fault-caught-2026-09-22/`](logs/fault-caught-2026-09-22/)): same signature, same doorbell, and
where August logged `GPU reset begin` this logged `GPU recovery disabled.` No reset, no SIGBUS, host
still up seventeen hours later. Two things it added. The preemption failure repeated every four seconds
for twenty-four minutes, ending only when the faulting process died, so the loop is a retry against a
queue belonging to a process that cannot be torn down. And Vulkan was unaffected throughout, decoding
at full speed while the loop printed, which puts the wedge on the compute queue RADV avoids and ROCm
cannot. That is one event's worth of evidence.

An earlier note here said no GPU fault accompanied the SIGBUS. That was a measurement error worth
admitting: the check was `dmesg` run after the board had rebooted, which necessarily reports the boot
*after* the crash. It cost weeks.

### Why decode at depth varies run to run

Up to 15 percent on some models and under 1 percent on others, in one boot, clock pinned, temperature
and memory flat. The bandwidth theory is not supported: with model order rotated the coefficients of
variation are 0.7, 6.0 and 4.6 percent at 29, 46 and 80 percent of the measured 432 GB/s
bandwidth ceiling, so
variability does not rise with utilisation.

The one solid clue is that it is not in the decode loop. Repetitions inside one `llama-bench`
invocation spread by a median of 0.43 where separate invocations of the same command spread 1.20, so
whatever varies belongs to per-process setup. That made allocation and placement the suspects and
neither survived: CPU pinning does nothing, and dropping the page cache and compacting memory does
nothing either. A six-run design suggested compaction cut the spread to a third (variance ratio 7.32,
p 0.048) and a twelve-run confirmation brought it to 1.16 with the arms indistinguishable. The first
result was one arm being unusually tight in a small sample.

A first-repetition effect is real and appears in seventeen of twenty invocations, but the theory that
its size grows with depth did not survive a sweep built to test it. What the warm-up actually is stays
unidentified: clock ramp, KV cache first touch, first-launch kernel setup and first-touch page faults
all fit and nothing here separates them.

### What the SDMA fix changed

Every measurement here was taken with `HSA_ENABLE_SDMA=0`, forced instead of chosen, so enabling it
had to be re-checked. Throughput ABBA, the correctness gates, Vulkan, the allocation-churn sweep and
decode at depth are all unchanged, and a three-hour endurance run with it on is clean
([`logs/soak-sdma-2026-09-22/`](logs/soak-sdma-2026-09-22/)).

One model does pay. The qwen2.5-1.5B reads 183.61 against 196.99, confirmed by an interleaved A/B at
0.947 with ten pairs of ten in the same direction
([`logs/sdma-decode-cost-2026-09-22/`](logs/sdma-decode-cost-2026-09-22/)). If this were per-copy
overhead scaling with copy frequency the MoE would sit between the 1.5B and the 8B, and it does not:
it reads 1.004. Only the fastest model pays, and nothing here distinguishes a threshold in rate from
something particular to that model.

### Smaller ones

- Why `sched_policy=2` wedges compute on this MEC at all, when it is the documented mitigation
  elsewhere and what earlier work on this board recommended.
- A lighter replacement for the runlist rebuild. The current fix rebuilds on every map and unmap,
  heavier than necessary in principle though below run-to-run noise in practice
  ([`logs/flush-cost-2026-08-18/`](logs/flush-cost-2026-08-18/)). Two lighter candidates were built
  and both fail. Why the map side needs it at all, a stale *invalid* translation surviving into a
  fresh mapping, is unexplained.
- Whether `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` should be the default on architectures without matrix
  cores. It measured free here on the model that takes the affected path.
- Whether the precision gaps affect other pre-RDNA3 HIP GPUs with large-activation models. That needs
  more hardware than one board.
- Whether any of this works on another gfx1013 board. Everything here is one board.
- Did the mining stacks run sustained compute on this path, and if so what did their kernel and
  firmware combination do differently? [ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313)
  hints that gfx1013 worked under older combinations.

### Answered since

- **Which kernel change between 6.18 and 7.1.5 made the difference?** None. With the same patches and
  boot arguments the two are indistinguishable. The question presupposed a kernel difference that was
  really a misapplied unlock patch plus `sched_policy=2` held fixed.
- **Why does the PASID invalidation cover nothing?** On gfx10 `gmc_v10_0_flush_gpu_tlb_pasid()` walks
  VMIDs 1 to 15 asking the ATHUB which PASID each holds, and an instrumented module reports
  `valid_vmids=0 matches=0` on every call, so the loop body never runs
  ([`logs/pasid-diagnosis-2026-08-19/`](logs/pasid-diagnosis-2026-08-19/)). Found first by the
  bc250-rocm-working project; this is a second board agreeing.
- **The per-iteration alloc/free fault in PyTorch.** The same allocation-reuse defect. Loops with
  varying sizes defeat torch's block cache and do force real map and unmap traffic.
- **What stops a ROCr SDMA copy above 16384 bytes completing?** Not ROCr: the board's own SDMA
  microcode. Substituting navi12's makes every size complete. The tip came from GabriWar in response
  to these notes, which is the best argument I have for publishing a defect you cannot fix.
- **Why does the first fp16 GEMM of each batch return exactly zero?** Fedora 43's ROCm compiler-rt
  ships broken half-precision helpers; the native rocBLAS links them and they make it read `alpha` as
  zero at random ([`logs/fp16-root-cause-2026-09-15/`](logs/fp16-root-cause-2026-09-15/)). Fedora 44
  does not carry the defect. Along the way graph capture was ruled out twice: the zero count, the
  positions and the perplexity are identical to the digit with capture disabled.

Data, corrections, or a "you are holding it wrong" are all welcome, as an issue here or a note on
[ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313).
## Reproducing

- rocm-hip 6.4.2, rocblas 6.4.4, ROCm LLVM/clang 19, mesa 25.3, llama.cpp build 2da6686. Historical
  observations: kernel 6.18.9-200.fc43. Working configuration: measured on kernel 7.1.5-100.fc43
  and, with the same patch set, identically on 6.18.9-200.fc43. The kernel version is not a
  requirement; the patch set and the boot arguments are.
- The working configuration: a kernel from 6.18.9 onward, the patched module carrying the flush
  change and the 40-CU unlock ([`scripts/build_patched_amdgpu.sh`](scripts/build_patched_amdgpu.sh),
  which takes
  `SRC` and `KREL` in the environment and rebuilds the initramfs itself, since the running module
  comes from there and not from `/lib/modules`),
  `amdgpu.bc250_cc_write_mode=3`, `amdgpu.bc250_flush_pasid_kiq=0` and
  `amdgpu.bc250_flush_by_runlist=3` on the command line (all three: the flush parameter is what
  selects the corrected PASID flush at boot, and omitting it leaves the patched module running the
  stock behaviour),
  **no `amdgpu.sched_policy` argument** (hardware scheduling is the default), and
  either the navi12 SDMA microcode substituted for the board's own or `HSA_ENABLE_SDMA=0` in the
  environment of every HIP process. That last point was stale until 26 August and read as though
  the environment variable were required: it is one of two ways, and the firmware substitution is
  the one the front page now leads with, since it fixes the transfers rather than avoiding them.
  Of the three command-line parameters, only `bc250_flush_by_runlist` comes from the scripts below.
  `bc250_flush_pasid_kiq` is added by the embedded patcher in
  [`scripts/ladder_prep_rung.sh`](scripts/ladder_prep_rung.sh) and `bc250_cc_write_mode` by the
  community 40-CU unlock, neither of which the pair below produces. The runlist parameter needs the
  module built with two scripts, in this order, each taking the amdkfd directory as an argument:
  [`scripts/apply_runlist_flush.py`](scripts/apply_runlist_flush.py) then
  [`scripts/apply_svmflush_generic.py`](scripts/apply_svmflush_generic.py). That pair is
  kernel-independent and is what to use. It is verified the way a reader meets it: a 6.18.16 tree
  restored to pristine, the two scripts run against it and nothing else, then built, installed,
  booted and put through the full battery. That module comes up at 40 CU and returns the compute
  probe correct at three sizes, SGEMM clean from N=256 to 4096, perplexity 8.9442, a sustained
  N=4096 for 50 iterations clean, and no dmesg faults
  ([`logs/recipe-e2e-2026-08-17/`](logs/recipe-e2e-2026-08-17/)), which also makes 6.18.16 a third
  kernel measuring equivalent. The `_715` variants in the file table are earlier kernel-specific
  versions kept for reference and fail on 6.x; value 1 is the original unmap-only behavior and
  3 enables the map side that closes the residual, and because the parameter is writable at
  runtime the two can be A/B tested on one boot. All three scripts take the amdkfd directory as an
  argument and validate it; two defects in
  [`apply_runlist_flush.py`](scripts/apply_runlist_flush.py) were found and fixed by running it
  against a pristine tree, not the author's, so if you have an older copy, replace it. It
  ignored the directory argument and silently patched a hardcoded path, and its `kfd_chardev.c`
  anchor only matched trees where `kfd_flush_tlb` takes one argument, which is 7.x; on 6.x, where
  it takes two, that hunk failed and the map side never got applied. Both are corrected, and the
  sequence is verified end to end on a pristine 6.18.16 tree. Verify the CU count and the module's
  flush-state line before
  trusting a boot. The driver prints the derivation at init, but `dmesg` may have rotated it out by
  the time you look, so read the boot log directly:

      sudo journalctl -b -k | grep -E 'active_cu_number|bc250_flush'

  which gives `SE 2, SH per SE 2, CU per SH 10, active_cu_number 40` along with the boot arguments
  actually in force. Match on `bc250_flush` and not on `bc250`, since the latter is a common
  hostname on these boards and would match every line in the log. The independent cross-check
  that does not depend on log retention is the SIMD count, since this chip has two SIMDs per CU:

      grep -h simd_count /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u

reporting 80 for 40 CUs and 48 for 24. Read it across all nodes, because node 0 is the CPU and
reports 0. - Reproducing the historical observations: kernel 6.18.9 with `amdgpu.sched_policy=2`,
which the earlier revision of this document recommended as a safety measure. That setting is the
difference between the wedge and clean runs, and not only on newer kernels: adding it to an
otherwise working 6.18.9 turns a clean board into one where the compute probe hangs at every size
and SGEMM wedges at N=256, with 16 preemption timeouts logged. Drop it once the flush fix is in,
on any kernel. - [`reproduce.sh`](reproduce.sh) is the quickest end-to-end check, run from the
repository root. It prints the kernel, rocBLAS and Mesa versions; lists the ISAs actually embedded
in the installed rocBLAS, where gfx1013 is absent; runs the rocBLAS probe three ways (system
library, system library under the gfx1010 override, and the native build); runs an OpenCL probe on
the graphics queue; and runs the HIP compute probe at 1M and 16M threads. It also refuses to
proceed if the map-side flush bit is missing, since a reader on an older module would otherwise
hit the churn faults this document describes. On a correctly configured board every arm passes
except the system-library rocBLAS one, which is expected to fail. - Native gfx1013 rocBLAS:
[`scripts/build_rocblas_gfx1013.sh`](scripts/build_rocblas_gfx1013.sh), then
[`patches/sgemm_sweep.cpp`](patches/sgemm_sweep.cpp). - The three llama.cpp changes are in
[`patches/llamacpp/`](patches/llamacpp/) and apply with `git apply` to a checkout at or near
7ba604f. Verified both ways: each reverse-applies against the tree that produced the measurements
in this document, so the shipped patch is the code that was measured, and each forward-applies
cleanly to a fresh checkout at 7ba604f. All three conditions were also checked against llama.cpp
master in August 2026 and all three still hold there: the HIP branch still takes `prop.integrated`
while the non-HIP branch does not, the RDNA1 macro still lists only `__gfx1010__` and
`__gfx1012__`, and `kq` still gets a precision request where the `kqv` multiply below it gets
none. They are not equally necessary. The first, `prop.integrated`, is mandatory: without it every
number on this board is wrong. The third, the RDNA1 macro entry, is what makes flash attention
correct and the quantized kernels fast. The second, KQV precision, is an alternative to setting
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, which this configuration sets anyway, so it can be skipped;
it is included because it fixes the cause rather than because the recipe needs it.
Each patch's contribution was measured by removing it from the working build of the time, the
three-patch one whose 1.5B prefill is 806 t/s, not the current 1799: without the macro
entry, prefill falls from 806 to 122 t/s and generated text becomes `The???????????????`, while
perplexity stays at 8.9425 and looks perfectly healthy. - llama.cpp: build with `-DGGML_HIP=ON
-DAMDGPU_TARGETS=gfx1013`. Two restrictions that predate the allocation-reuse flush have been retested and dropped: - **Memory mapping is fine.** `--no-mmap` is not required. With mapping enabled the small model returns
8.9442 three times out of three, bit-identical to the no-mmap reference, and the 14B loads three
times out of three. Note that `--mmap` and `--no-mmap` are deprecated in current llama.cpp in
favour of `--load-mode`, so the old flags now warn (and `--mmap 1`, which reads naturally, is
rejected outright: it is a boolean). - **One benchmark per invocation is no longer needed for
stability.** The recipe warned that multi-size sweeps reallocate between tests and can trip the
load-time fault mid-run. A single invocation sweeping pp128, pp512, pp1024, pp2048, tg32 and tg64
now completes all six rows with no faults in dmesg. It is still needed for accuracy, which is a
separate matter found later: passing depths as a list reads up to 22 percent low on either
backend, so every campaign here uses one test per invocation
([README.md](README.md#prefill-at-depth)). - For anything beyond the small model, put the native
gfx1013 rocBLAS on the library path (`LD_LIBRARY_PATH=<rocblas-install>/lib`) and set
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`. One avoids `CUBLAS_STATUS_INTERNAL_ERROR` aborts
wherever inference reaches a GEMM the system library has no gfx1013 code object for; the second
fixes the qwen3-family batch corruption, costs nothing on this chip, and is required for
correctness on models with F16 weights (see the caveats). Whether a given run needs the native
library is directly checkable instead of a matter of judgement: prefix it with `ROCBLAS_LAYER=1`
and count the calls. The 1.5B Q4_K at pp512 makes zero, and that is why it is correct against the
system library, while the 8B Q8_0 calls `rocblas_sgemm` and aborts without the native one. Note
that `GGML_CUDA_FORCE_MMQ` is compile-time only in current llama.cpp, so setting it in the
environment does nothing. It is not the only one: of the fourteen environment variables this work
relies on, twelve were confirmed on the board to be read by the library that would have to read
them, and the two that are not are compile-time options of this kind
([`logs/env-audit-2026-08-26/`](logs/env-audit-2026-08-26/)). Resolving a variable against the
library instead of assuming it is read is worth the minute it takes; two of the numbers in an
earlier revision of this document rested on a variable that was never consulted. - llama.cpp version: the numbers here come from build 2da6686, which is
perplexity-verified exact on this board. Anything including commit `c7d8722` (the
`prop.integrated` restore, PR #24233) computes wrong on this board at every setting until that
flag is forced off again; see the version warning in the caveats. Whatever the build, gate it with
`llama-perplexity -f wiki.test.raw --chunks 8` against the same model under Vulkan on the same
command before trusting any other number from it. Reference points from the gated campaign
(qwen2.5-1.5B, `-fa on -c 4096 --chunks 8`): HIP 8.9442, Vulkan 8.9760, and the HIP value
reproduced bit-identically across two boots. The older `-ub 8` gate at default context reads 11.21
on both backends. - Two footguns in newer llama.cpp CLIs that both look like board failures and
are not: scripted `llama-cli` runs need `--single-turn` (`-no-cnv` no longer prevents the
interactive console, and with a closed stdin the process spins forever after generating, which
reads as a "hung load" and writes hundreds of MB of prompt characters); and the CLI defaults to
the model's full context length, so an 8 GiB+ model needs an explicit `-c` (4096 here) or it dies
in host-memory allocation ("SVM mapping failed, exceeds resident system memory limit").

Two things that cost time. A rebuilt kernel module must be compressed with `xz --check=crc32` (the
build script does this): the xz default is CRC64, which loads via userspace `modprobe` but fails the
in-kernel decompressor from the initramfs (`decompression failed with status 6`), while `xz -t`
still passes, so it looks like a bricked board. Keep a verified-good module backup. And after a
compute wedge a soft reboot often does not recover the queue; a hard power-cycle does. Always verify
`active_cu_number 40` in the boot log (via `journalctl -b -k`, since `dmesg` may have rotated),
or `simd_count 80` in the KFD topology, and a good `compute_probe` or GEMM run before trusting a
setup.

## Files

The harnesses here were written to run on one board and carry its paths. The
rocBLAS build directory, which forty-nine of them need, is overridable with
`ROCBLAS_LIB_DIR` and otherwise defaults to where it sat on that machine, so a
script run without the variable behaves exactly as it did when its log was
taken. Model paths under `/opt/models`, kernel source trees and the netconsole
addresses are not parameterised: they are what they were, and anyone re-running
these will be editing them. That is the nature of proof-of-work scripts rather
than a recipe, and the recipe itself, `reproduce.sh` and the two patch scripts,
carries no such paths.

| Path | What it is |
|------|------------|
| [`logs/README.md`](logs/README.md) | Index of all 214 log directories, one line each. The table below picks out the ones worth describing at length |
| [`logs/bench-2026-08/`](logs/bench-2026-08/) | The working-configuration benchmark campaign: per-test llama-bench logs (HIP and Vulkan), SGEMM curve, probe sweeps, depth ladder, load-reliability trials, PyTorch runs |
| [`figures/`](figures/) | The three figures above, with [`scripts/make_figures.py`](scripts/make_figures.py) to regenerate them from the logged numbers |
| [`logs/factorial/`](logs/factorial/) | The two-by-two flush-by-scheduler factorial: per-boot raw logs and verdicts, results table, and the battery/orchestration scripts ([`scripts/factorial_battery.sh`](scripts/factorial_battery.sh), [`scripts/factorial_orchestrate.sh`](scripts/factorial_orchestrate.sh)) |
| [`patches/gen_reuse.cpp`](patches/gen_reuse.cpp) | Targeted reproducer for the stale-translation-on-address-reuse mechanism, written against the description from bc250-rocm-working. Used by [`scripts/aba_mapflush_verify.sh`](scripts/aba_mapflush_verify.sh), so it is shipped alongside it |
| [`patches/seq_probe.c`](patches/seq_probe.c) | Deterministic allocation-reuse reproducer (heavy dispatch, free, realloc, dispatch); the flush-by-runlist A/B evidence in [`logs/bench-2026-08/runlist-verify/`](logs/bench-2026-08/runlist-verify/) |
| [`scripts/apply_runlist_flush.py`](scripts/apply_runlist_flush.py) | Applies the runlist-rebuild-on-unmap flush (ported from bc250-rocm-working) as the `bc250_flush_by_runlist` module parameter. Takes the amdkfd directory as an argument and validates it; works on kernels where `kfd_flush_tlb` takes either one or two arguments. Step one of two |
| [`scripts/soak_current_stack.sh`](scripts/soak_current_stack.sh) | The endurance soak for the current recipe: prefill, a perplexity gate, an allocation-churn sweep every round and a PyTorch training loop every third, which is the path the earlier soaks never covered |
| [`scripts/loose_end_controls.sh`](scripts/loose_end_controls.sh) | Three controls run together: the quantized-value model that should never reach the fp16 path, a low-bandwidth model measured the same way as the high-bandwidth one, which was the first half of the bandwidth explanation for decode variance, and the fp16 case with flash attention off |
| [`scripts/gap_probe_three.sh`](scripts/gap_probe_three.sh) | Three gaps in one run: fp16 error against amount of text evaluated, whether the SDMA boundary moves with the staging-buffer knob as well as the blit-copy one, and decode variance for a third model at intermediate bandwidth utilisation. The first arm rests on a design assumption that does not hold, and the header says so |
| [`scripts/graph_flag_control.sh`](scripts/graph_flag_control.sh) | Isolates `GGML_CUDA_DISABLE_GRAPHS=1` as a confound in the decode-variance comparison, by running the high-bandwidth model under the flag the low-rate model required |
| [`scripts/campaign_rerun.sh`](scripts/campaign_rerun.sh) | Repeats the five-model campaign on the current stack, since the most-cited figures in this document came from a build several changes ago |
| [`scripts/macro_removal_remeasure.sh`](scripts/macro_removal_remeasure.sh) | Rebuilds with the gfx1013 RDNA1 entry removed and restores it, to re-measure the healthy-perplexity-with-garbled-output arm from the tree, not cite it |
| [`scripts/sigbus_characterise.sh`](scripts/sigbus_characterise.sh) | Separates flag from depth from chance after a run died with SIGBUS in the ROCr queue path. Its depth-0 arm mis-parses `llama-bench`'s depth-0 row format, so those runs read as crashes in the raw log and are not |
| [`scripts/sdma_queue_probe.sh`](scripts/sdma_queue_probe.sh) | Samples the KFD queue descriptors during a hanging SDMA copy, with a compute-workload control that establishes whether descriptors track live state at all |
| [`scripts/fp16_graph_arms.sh`](scripts/fp16_graph_arms.sh) | Runs the zeroed-fp16-GEMM case with HIP graph capture on and off against an f32 control, to test whether the capture machinery is involved |
| [`scripts/decode_history_control.sh`](scripts/decode_history_control.sh) | Measures the same decode workload on a clean board, immediately after a 10.7 GiB model has run, and after dropping caches, to test whether memory history explains a low outlier. It does not |
| [`scripts/decode_variance.sh`](scripts/decode_variance.sh) | Repeated decode measurements at a fixed depth with the shader clock sampled during each run, written to find out why one model's decode rate moves when nothing else does |
| [`scripts/bench_fixed_stack.sh`](scripts/bench_fixed_stack.sh) | The harness behind [`logs/bench-fixed-2026-08/`](logs/bench-fixed-2026-08/): the whole five-model campaign on the patched stack, HIP and Vulkan, one benchmark per invocation with a perplexity gate before each rate is recorded |
| [`scripts/ftrace_alloc_reuse_window.sh`](scripts/ftrace_alloc_reuse_window.sh) | The ftrace capture behind the allocation-reuse mechanism: freezes the trace buffer the moment the faulting process exits, so the unmap-to-fault window can be measured rather than inferred |
| [`scripts/aba_mapflush_verify.sh`](scripts/aba_mapflush_verify.sh) | The A/B/A battery for the map-side flush, toggling the runtime-writable parameter within one boot |
| [`scripts/soak_fixed_stack.sh`](scripts/soak_fixed_stack.sh) | The endurance soak behind the stability rows: alternates a prefill hammer, a perplexity gate and an allocation-churn sweep, logging temperature and clock throughout, and stops early if a gate drifts |
| [`patches/llamacpp/`](patches/llamacpp/) | The three llama.cpp changes as applicable patches: `0001` counter-patches the `prop.integrated` regression, `0002` requests fp32 precision on the KQV matmul, `0003` adds gfx1013 to the RDNA1 macro. Each was generated from, and checked against, the tree the measurements were taken on |
| [`scripts/ladder_prep_rung.sh`](scripts/ladder_prep_rung.sh), [`scripts/ladder_rung_test.sh`](scripts/ladder_rung_test.sh) | Build the patch set against an arbitrary Fedora kernel from koji and test that rung (unlock, sustained GEMM, perplexity, churn); the per-kernel results behind the Observation 3 correction are in [`logs/ladder-2026-08-13/`](logs/ladder-2026-08-13/) |
| [`scripts/apply_svmflush_generic.py`](scripts/apply_svmflush_generic.py) | Step two of two, applied on top of the above: the parameter becomes a runtime-writable bitmask (1 unmap, 2 map) and the rebuild is added to the KFD SVM map and unmap paths, which is what actually closes the fault (evidence in [`logs/svm-flush-2026-08/`](logs/svm-flush-2026-08/): the ftrace correlation and the A/B/A battery). Locates its call sites by enclosing function and not by literal context, so it is not tied to a kernel version |
| [`scripts/apply_mapflush_715.py`](scripts/apply_mapflush_715.py), [`scripts/apply_svmflush_715.py`](scripts/apply_svmflush_715.py) | The earlier 7.1.5-specific versions of step two, superseded by the generic script above and kept only for reference. They match literal context from that tree and fail on 6.x, where `kfd_flush_tlb` takes two arguments |
| [`logs/mmq-2026-08-14/`](logs/mmq-2026-08-14/) | Per-tensor statistics from the fp16-compute and f32-compute runs that localized the zeroed value projection (the directory name preserves the original, mistaken, "MMQ" label), one line per tensor (`STATS <name> <op> n= sum= sumsq= maxabs=`); diff two of them positionally to see it |
| [`patches/membw.cpp`](patches/membw.cpp) | Streaming-read bandwidth measurement, 432 GB/s on this board, used above to show which decode rates are memory-bound and which are not |
| [`patches/pytorch/0001-fedora-rocm-build.patch`](patches/pytorch/0001-fedora-rocm-build.patch) | The three source changes needed to build PyTorch for gfx1013 against distribution-packaged ROCm, not the AMD installer layout: the lib64 CMake module path, the ROCm version computed from a HIP build number, and a version header Fedora does not ship |
| [`patches/pytorch/torch_train.py`](patches/pytorch/torch_train.py), [`torch_train_diverge.py`](patches/pytorch/torch_train_diverge.py) | GPU training loop against a CPU reference, and the follow-up that separates numerical drift from a defect by looking at single-step gradients, divergence growth, and run-to-run determinism |
| [`patches/torch_opprobe.py`](patches/torch_opprobe.py) | Eleven PyTorch operations run in isolation so one failure does not mask the rest, which is what separates the library paths from the wheel's own kernels |
| [`patches/torch_ctypes_test.py`](patches/torch_ctypes_test.py), [`patches/bc250_ext.hip`](patches/bc250_ext.hip) | A gfx1013 kernel run on the stock wheel's own allocator memory, establishing that missing code objects are the only blocker before committing hours to a build |
| [`patches/torch_fp16_zero_cross.py`](patches/torch_fp16_zero_cross.py) | Cross-check of the zeroed fp16 GEMM in a second implementation on the same rocBLAS |
| [`patches/hgemm_zero_probe2.cpp`](patches/hgemm_zero_probe2.cpp) | Standalone attempt at the zeroed fp16 GEMM: same shape, own stream, fresh converted buffer per call. Runs clean over 100 calls, which is the point, the defect needs the full model's context. Build with `hipcc -x hip --offload-arch=gfx1013` or the conversion kernel silently does not run |
| [`patches/sdma_angles.c`](patches/sdma_angles.c) | Runs ten SDMA variants, each in its own watchdogged process so one hang does not end the run: pinned against pageable, both directions, sync against async, and the device-side operations that turn out to use blit kernels instead of SDMA |
| [`patches/sdma_probe.c`](patches/sdma_probe.c) | Walks `hipMemcpy` from 4 KiB to 2 GiB, pageable and pinned, both directions, with a per-copy watchdog, because the SDMA failure is a silent hang, not an error; locates the 4 KiB to 64 KiB boundary described above (`hipcc -x hip`) |
| [`patches/dgemm_iter.cpp`](patches/dgemm_iter.cpp) | FP64 DGEMM probe via native gfx1013 rocBLAS (spot-checked against CPU) |
| [`patches/mandelbrot.cpp`](patches/mandelbrot.cpp) | Custom HIP kernel example (FP64 Mandelbrot to PGM) |
| [`patches/torch_matmul_bench.py`](patches/torch_matmul_bench.py) | PyTorch preallocated-buffer matmul benchmark (the allocation-discipline demonstration) |
| [`patches/amdgpu-flush-pasid-mmio.patch`](patches/amdgpu-flush-pasid-mmio.patch) | The one-line kernel change for the correctness observation |
| [`logs/umr/`](logs/umr/) | July captures of a natural stall, with kernel stacks showing the waiting thread parked in `kfd_wait_on_events`. Kept as archival: they belong to the event-latency thread from that period, and the load-hang conclusion that thread fed into was later retracted as a harness artifact (see the multi-boot note). The stacks themselves are real captures |
| Older probes kept for the record | [`patches/evtlat.c`](patches/evtlat.c) and [`patches/loadmimic.c`](patches/loadmimic.c) (event-latency and load-pattern reproducers from the July investigation), [`patches/kfd_skip_eviction_gfx1013.py`](patches/kfd_skip_eviction_gfx1013.py) and [`patches/amdgpu-fence-fallback-2ms.patch`](patches/amdgpu-fence-fallback-2ms.patch) (kernel experiments that did not pan out), [`scripts/bench_prefill.sh`](scripts/bench_prefill.sh) and [`scripts/sweep_cfg.sh`](scripts/sweep_cfg.sh) (early benchmark harnesses, superseded by the campaign scripts above) |
| [`patches/compute_probe.c`](patches/compute_probe.c) | Bare HIP compute reproducer (native gfx1013, CPU-checked) |
| [`patches/ocl_compute_probe.c`](patches/ocl_compute_probe.c) | OpenCL port of the probe (graphics-queue comparison via RustiCL) |
| [`patches/ocl_vecadd.c`](patches/ocl_vecadd.c) | Minimal OpenCL vector add, a smoke test that the graphics-queue compute path works at all before running the larger probe against it |
| [`patches/sgemm_sweep.cpp`](patches/sgemm_sweep.cpp) | Native gfx1013 rocBLAS SGEMM sweep with CPU check and timing |
| [`patches/sgemm_iter.cpp`](patches/sgemm_iter.cpp) | Leak-free single-process SGEMM probe (one allocation, per-iteration check) used for the eviction trace |
| [`patches/rocblas_probe.c`](patches/rocblas_probe.c) | Standalone rocBLAS SGEMM availability/correctness test |
| [`scripts/build_patched_amdgpu.sh`](scripts/build_patched_amdgpu.sh) | Build the patched amdgpu module (module-only) |
| [`scripts/build_rocblas_gfx1013.sh`](scripts/build_rocblas_gfx1013.sh) | Build a native gfx1013 rocBLAS on Fedora system ROCm |
| [`scripts/native_fa.sh`](scripts/native_fa.sh) | A July hypothesis about which combination would work, kept as the record of it rather than as a recipe. Two of its three ingredients have since been settled differently: `GGML_CUDA_FORCE_MMQ` is compile-time only so setting it in the environment does nothing, and the flash-attention problem was a missing architecture-macro entry. The working recipe is in [Reproducing](#reproducing) |
| [`logs/ftrace/wedge_eviction_stack.txt`](logs/ftrace/wedge_eviction_stack.txt) | Function-tracer stacks showing the wedge is a queue eviction (triggered by the process's `munmap`) whose MEC preemption times out, plus the recipe to reproduce it |
| [`logs/wedge_knob_sweep.txt`](logs/wedge_knob_sweep.txt) | Per-knob results from the 6.18-era sweep: scheduler, CWSR, interrupt, mcbp, preemption timeout, hugepages, firmware version, XNACK, pacing, newer-kernel source check. Its conclusion, "no knob removed it", does not hold up: the scheduler knob does remove it, and this file records `sched_policy=0` as wedging like policy 2. It does not record which flush the module carried, which is how it cannot settle that on its own; see the correction at the end of Observation 2 |
| [`logs/stock/`](logs/stock/), [`logs/patched/`](logs/patched/) | The Observation 1 A/B: the bare compute probe on the stock module against the module carrying the corrected flush, across clock settings and fresh boots. This is where the silent wrong results were first counted |
| [`logs/soak-large-2026-08-14/`](logs/soak-large-2026-08-14/) | The 8.1 hour large-model soak: 78 rounds rotating the 8B, the 14B and the 35B MoE, with `soak.log` holding the per-round gates and `thermals.tsv` the temperature and clock samples behind the 94C peak |
| [`logs/kernel-7.1.5/`](logs/kernel-7.1.5/) | Newer-kernel test: `compute_probe` (fresh-boot samples + first sweep) and native rocBLAS on Fedora kernel 7.1.5 with the 40-CU unlock, showing the correctness defect and the wedge both persist there (measured under `sched_policy=2`; see the working-configuration section for the reinterpretation). Sampler: [`scripts/probe_kernel_sweep.sh`](scripts/probe_kernel_sweep.sh) |
| [`logs/deep-dive-2026-07-28/`](logs/deep-dive-2026-07-28/) | Follow-up round: the amdgpu VM-fault decode (TCP/UTCL2 read permission fault), a PyTorch native-gfx1013 matmul sweep (correct single, wedges sustained), and the gfx1010-symlink note. Summary in that folder's README |
| [`patches/bw_probe.cpp`](patches/bw_probe.cpp) | Bare HIP streaming-read kernel (no arithmetic, no rocBLAS) that reproduces the size-dependent wedge/fault |
| [`patches/sdma_one.c`](patches/sdma_one.c) | Single-size SDMA copy loop, taking the size in bytes and looping so the queue stays busy while its descriptor is sampled. Written because the older probe takes a repetition count and not a size, which invalidated a first attempt at the one-byte comparison |
| [`patches/fp16_solutions.cpp`](patches/fp16_solutions.cpp) | Enumerates the Tensile solutions rocBLAS offers for the failing fp16 shape and validates each against a CPU reference |
| [`logs/inference/decode_aperture_violation.txt`](logs/inference/decode_aperture_violation.txt) | Earlier `AMD_LOG_LEVEL=3` decode aperture-violation trace (on a different llama.cpp build; later runs place the fault on the runtime host-to-device copy) |
| [`logs/inference/decode_copybuffer_aperture_violation.txt`](logs/inference/decode_copybuffer_aperture_violation.txt) | Later `AMD_LOG_LEVEL=3` trace (llama.cpp b9265): the aperture violation aborts on `__amd_rocclr_copyBuffer` with no compute kernel dispatched first (second sample in `...violation2.txt`) |
| [`logs/inference/decode_campaign_stats.txt`](logs/inference/decode_campaign_stats.txt) | Single-boot decode campaign: per-attempt verdict (clean / aperture fault / load timeout), showing the intermittent load as the dominant blocker |
| [`scripts/decode_stats.sh`](scripts/decode_stats.sh) | Runs the decode campaign above, classifying each attempt by its last GPU dispatch |
| [`scripts/clean_build_control.sh`](scripts/clean_build_control.sh) | Builds a clean tree at the campaign commit carrying only the three shipped patches, and measures it against the instrumented working tree in the same boot, to test whether the instrumentation moved any quoted number |
| [`scripts/counterbalanced_repeat.sh`](scripts/counterbalanced_repeat.sh) | ABBA and rotated-order repeats of two comparisons that had been run as blocks. It reversed the sign of one of them, so blocked designs are not used here for small effects |
| [`scripts/corpus_instrument_check.sh`](scripts/corpus_instrument_check.sh) | Varies the two constants no measurement here had ever varied: the single corpus behind every correctness gate, and the single benchmark tool behind almost every throughput figure |
| [`scripts/kqv_removal_remeasure.sh`](scripts/kqv_removal_remeasure.sh) | Removes the KQV precision line, rebuilds, measures the three arms in the configuration the originals came from, and restores. Also checks that `GGML_CUDA_NO_VMM`, a constant the original harness carried, is inert |
| [`scripts/integrated_flag_remeasure.sh`](scripts/integrated_flag_remeasure.sh) | Reverts the `prop.integrated` counter-patch, rebuilds, measures, and restores, in the configuration the original figures came from |
| [`scripts/sdma_onebyte_intervention.sh`](scripts/sdma_onebyte_intervention.sh) | The one-byte SDMA comparison, alternated: 16384 bytes against 16385 through the same binary, sampling the queue descriptor while the queue is kept busy |
| [`scripts/decode_sdma_mqd.py`](scripts/decode_sdma_mqd.py) | Decodes the SDMA queue descriptor from a KFD dump against `struct v10_sdma_mqd` in the kernel headers, so the fields are named, not guessed. Written after a coarser reading of the same dumps produced a misleading comparison |
| [`scripts/fp16_dispatch_trace.sh`](scripts/fp16_dispatch_trace.sh) | Traces what the fp16 and f32 paths actually dispatch to rocBLAS, which shows all 36 value projections are parameter-identical |
| [`scripts/fp16_instrumentation_ab.sh`](scripts/fp16_instrumentation_ab.sh) | Alternated arms testing whether tracing suppresses the defect, and a raw-bits read of the alpha and beta llama.cpp passes |
| [`scripts/fp16_pool_arms.sh`](scripts/fp16_pool_arms.sh) | Alternated arms testing the fp16 defect against the pool-bypass and no-reuse hooks already in the tree |
| [`scripts/override_trap_probe.sh`](scripts/override_trap_probe.sh) | Runs the bare compute probe with and without `HSA_OVERRIDE_GFX_VERSION`, back to back on one boot, which is what turned the silent-zeroing warning from a recollection into a capture |
| [`scripts/sdma_firmware_identity.sh`](scripts/sdma_firmware_identity.sh) | Reads the SDMA firmware headers and compares the board's blob against navi12's byte by byte, needing the backup since the substitution overwrites the original |
| [`scripts/sdma_firmware_ab.sh`](scripts/sdma_firmware_ab.sh) | Whether enabling the now-working SDMA is worth it: throughput and load time with it on against off, alternated, plus repeated gates |
| [`scripts/sdma_size_sweep.sh`](scripts/sdma_size_sweep.sh) | Transfer-size sweep with SDMA on against off, alternated per size, which locates the band where each path wins |
| [`scripts/kernel_718_battery.sh`](scripts/kernel_718_battery.sh) | The comparison battery run on kernel 7.1.8 after porting all four patches to a fresh tree |
| [`scripts/vmid_flush_candidates.sh`](scripts/vmid_flush_candidates.sh) | Tests whether a VMID-level flush can replace the runlist rebuild, using a model load instead of the reproducer that cannot tell the cells apart |
| [`scripts/sdma_depth_ab.sh`](scripts/sdma_depth_ab.sh) | Alternated decode-at-depth comparison with SDMA on against off |
| [`scripts/sdma_constant_and_fp16_scope.sh`](scripts/sdma_constant_and_fp16_scope.sh) | Re-tests the SDMA constant against churn and ceilings, and widens the fp16 model survey |
| [`scripts/fp16_flashattn_arms.sh`](scripts/fp16_flashattn_arms.sh) | The fp16 defect across flash attention on and off, two rounds of four cells |
| [`scripts/sigbus_hunt.sh`](scripts/sigbus_hunt.sh) | Thirty repeats of the deep-decode measurement, counting crashes and capturing backtraces |
| [`scripts/fp16_arch_arms.sh`](scripts/fp16_arch_arms.sh) | The fp16 defect under gfx1013 against gfx1010, using a llama.cpp built for both so the override has kernels to run |
| [`scripts/fp16_library_arms.sh`](scripts/fp16_library_arms.sh) | The blocked first attempts at varying the rocBLAS build |
| [`scripts/fault_count.sh`](scripts/fault_count.sh) | A helper that counts GPU faults from both the current and the previous boot, for harnesses written after the dmesg blind spot was found |
| [`scripts/journal_retro_sweep.sh`](scripts/journal_retro_sweep.sh) | Sweeps every retained boot in the persistent journal for GPU faults and resets, identifying boots by kernel and scheduler policy, not by wall-clock time, which jumps across reboots here |
| [`scripts/soak_sdma_alternating.sh`](scripts/soak_sdma_alternating.sh) | The endurance soak alternating SDMA on and off round by round, which ran 253 rounds bit-identical before the board reset itself |
| [`scripts/nommap_ceiling.sh`](scripts/nommap_ceiling.sh) | The depth ladder in both load modes, and memory sampled through one run of each at the failing depth, which is how the no-mmap context ceiling was traced to reclaimable memory |
| [`scripts/decode_variance_process_state.sh`](scripts/decode_variance_process_state.sh) | Decode at depth under three process-state arms, default, CPU-pinned and cache-dropped, rotated |
| [`scripts/decode_variance_confirm.sh`](scripts/decode_variance_confirm.sh) | The twelve-round confirmation of that comparison, which is where the apparent effect evaporated |
| [`scripts/f32_workaround_cost.sh`](scripts/f32_workaround_cost.sh) | What the f32 compute-type workaround costs, measured on a stated configuration after the original figures were found to have no shipped log |
| [`scripts/kqv_ladder_remeasure.sh`](scripts/kqv_ladder_remeasure.sh) | The KQV context ladder reproduced, which showed the defect reaching context 1024 where the upstream note had called it clean |
| [`scripts/kqv_periodicity.sh`](scripts/kqv_periodicity.sh) | Twelve consecutive identical runs at one context, which refuted the apparent periodicity in the ladder and showed the wrong answers are discrete |
| [`scripts/kqv_dispatch_trace.sh`](scripts/kqv_dispatch_trace.sh) | Every rocBLAS call logged across eight runs, showing identical call sequences returning four different answers |
| [`scripts/kqv_first_divergence.sh`](scripts/kqv_first_divergence.sh) | An attempt to locate the first divergent tensor with the eval callback, which produced byte-identical dumps for a reason that turned out to be workload rather than instrumentation |
| [`scripts/kqv_divergence_control.sh`](scripts/kqv_divergence_control.sh) | The same with a control arm on the perplexity configuration, which is what eventually exposed the workload explanation |
| [`scripts/kqv_graph_capture_arms.sh`](scripts/kqv_graph_capture_arms.sh) | The defect with HIP graph capture disabled |
| [`scripts/kqv_sync_probe.sh`](scripts/kqv_sync_probe.sh) | A patch that synchronises after every node and copies nothing, with a counter proving it ran |
| [`scripts/kqv_fusion_probe.sh`](scripts/kqv_fusion_probe.sh) | A switch that refuses every operator fusion, with a counter proving it ran |
| [`scripts/kqv_fusion_reference.sh`](scripts/kqv_fusion_reference.sh) | What the correct perplexity is when fusion is refused, measured on the patched build so the fusion arm could be scored against the right reference |
| [`scripts/kqv_granularity_probe.sh`](scripts/kqv_granularity_probe.sh) | The scheduler patched to submit one node per graph launch with no callback installed |
| [`scripts/kqv_hostcopy_probe.sh`](scripts/kqv_hostcopy_probe.sh) | Every result read back to the host and discarded, the last of the five eliminations |
| [`scripts/gpu_reset_test.sh`](scripts/gpu_reset_test.sh) | Asks the driver to reset the device from an idle GPU, which is how the reset was found to be fatal on its own |
| [`scripts/fault_hunt.sh`](scripts/fault_hunt.sh) | A reboot-surviving loop that runs the fault-prone workload and a correctness gate every fifth round, keeps a rolling map of the live process, reads both the current and previous boot after every iteration, and separates faults raised on purpose by the probes from natural ones, which otherwise look identical |
| [`scripts/fault_under_load.sh`](scripts/fault_under_load.sh) | Fires the fault probe while a benchmark runs, so the fault lands with queues busy |
| [`scripts/campaign_current_config.sh`](scripts/campaign_current_config.sh) | The five-model throughput table and the four-model correctness gates re-run end to end on the configuration the README now recommends, both backends, same flags as the original |
| [`scripts/defects_still_open.sh`](scripts/defects_still_open.sh) | Re-checks that the documented open defects still reproduce, and the fixed ones stay fixed, after the microcode, kernel and driver-parameter changes |
| [`scripts/bc250_power`](scripts/bc250_power) | Switches the board on or off through a smart plug, waiting for ssh on the way up and shutting down cleanly on the way down, so a measurement can bracket its own power window. The switching half is specific to this bench and is driven by a Home Assistant instance whose address, token and switch entity all come from outside this repository |
| [`scripts/frontpage_verify.sh`](scripts/frontpage_verify.sh) | The front-page claims that other re-runs had not reached, bandwidth, DGEMM, context ceilings, the 27B prefill and concurrent ROCm with Vulkan, batched into one powered window |
| [`scripts/fp16_arch_and_queue_probes.sh`](scripts/fp16_arch_and_queue_probes.sh) | The fp16 defect across presented architectures, and decode rate against the queue descriptor the process is given |
| [`scripts/install_probe_module.sh`](scripts/install_probe_module.sh) | Installs a rebuilt amdgpu module the documented way, stripping, compressing with crc32 and running dracut, keeping the previous module |
| [`scripts/kfd_reset_probe.sh`](scripts/kfd_reset_probe.sh) | Adds a module parameter that calls the KFD reset path directly, so the mitigation can be tested without waiting for a fault |
| [`scripts/netconsole_capture.sh`](scripts/netconsole_capture.sh) | Arms netconsole so the kernel log survives a machine that stops writing to disk, which is how the reset was found to succeed before the host stalls |
| [`scripts/aslr_confirm.sh`](scripts/aslr_confirm.sh) | The confirmation run for the address-randomisation effect, eight rounds per arm, with the prediction recorded before the data |
| [`scripts/three_gaps.sh`](scripts/three_gaps.sh) | Three gaps found while re-checking the investigation: the fp16 mechanism checked directly and not through perplexity, whether the native rocBLAS can serve a foreign override, and decode variance against address space randomisation |
| [`scripts/gaps2.sh`](scripts/gaps2.sh) | Whether the zeroed GEMMs survive with graph capture disabled, and whether decode variance depends on primed depth |
| [`scripts/ubatch_zero_test.sh`](scripts/ubatch_zero_test.sh) | Tests the batch-boundary model of the zeroed fp16 GEMM by predicting the zero count from the micro-batch size before measuring it |
| [`scripts/resume_bisect_probe.sh`](scripts/resume_bisect_probe.sh) | Names each IP block as it resumes, so a block that is entered and never returns identifies itself; this is what located the reset stall inside GFX |
| [`scripts/gfx_resume_bisect.sh`](scripts/gfx_resume_bisect.sh) | Bisects inside the GFX resume, around constants_init, rlc_resume and cp_resume; this is what placed the reset stall in the Command Processor |
| [`scripts/cp_resume_bisect.sh`](scripts/cp_resume_bisect.sh) | Bisects inside cp_resume, around the KIQ, KCQ and graphics-ring resumes; this is what placed the reset stall in the KIQ |
| [`scripts/kiq_init_bisect.sh`](scripts/kiq_init_bisect.sh) | Bisects the seven reset steps inside kiq_init_queue, and splits the register read from the write; its own prints move the failure point, which is the finding |
| [`scripts/kiq_settle_delay.sh`](scripts/kiq_settle_delay.sh) | A runtime-tunable delay before KIQ init on the reset path, which falsified the settle-time reading at 50 and 500 milliseconds |
| [`scripts/kiq_gap_probe.sh`](scripts/kiq_gap_probe.sh) | Varies what sits between the KIQ scheduler register read and its write; its arms printed nothing, which is how the instrument was found to sit downstream of the stall |
| [`scripts/kiq_read_probe.sh`](scripts/kiq_read_probe.sh) | Puts the same knob on both sides of that read, so every configuration can be repeated against one byte-identical module |
| [`scripts/kiq_reg_probe.sh`](scripts/kiq_reg_probe.sh) | Reads a global, a scratch and an RLC register just before the one the reset path hangs on, so the first that hangs names how much of the block is unreachable |
| [`scripts/kiq_repeat_summary.sh`](scripts/kiq_repeat_summary.sh) | Renders the repetition campaign one row per trial, so a configuration that splits across its repetitions stays visible |
| [`scripts/reset_trial_repeat.sh`](scripts/reset_trial_repeat.sh) | Triggers a reset N times and keeps every capture, since a stall on this board needs a rate, not an anecdote |
| [`scripts/reset_path_compare.sh`](scripts/reset_path_compare.sh) | The sequence for asking whether the debugfs reset and the reset the fatal events take fail at the same instruction, which the bisect assumed |
| [`scripts/fault_usability_hunt.sh`](scripts/fault_usability_hunt.sh) | Waits for a natural fault under `gpu_recovery=0` and interrogates the machine before anything else touches the GPU; baselines the journal per boot instead of per service start, and treats a round whose workload died as evidence, both of which it failed to do the day it caught its fault |
| [`scripts/fault_hunt_status.sh`](scripts/fault_hunt_status.sh) | One screen of hunt state, distinguishing a hunt that is merely running from one that has caught something |
| [`patches/faultprobe.hip`](patches/faultprobe.hip) | A kernel that writes far outside its allocation, which reproduces the fault signature of the events that kill the board |
| [`patches/faultbusy.hip`](patches/faultbusy.hip) | The same fault raised from inside a process holding thirty-two kernels in flight across four streams, so the queues needing preemption belong to the faulting context |
| [`logs/ladder-churn-2026-08-16/`](logs/ladder-churn-2026-08-16/) | Cross-kernel churn evidence, one directory per boot, each with the kernel, the runlist value, whether the map hook was in the loaded module, and the SIMD count. Includes the 6.18.9 pair that attributes the 6.x stall to the same defect, not to a short timeout |
| [`logs/kernel-equivalence-2026-08-17/`](logs/kernel-equivalence-2026-08-17/) | The same validation battery on 6.18.9 and 7.1.5, showing them indistinguishable, plus the CU-count by scheduler-policy factorial cells |
| [`logs/qwen38-2026-08-17/`](logs/qwen38-2026-08-17/) | Qwen3.8-27B on both backends: benchmarks, perplexity gates, and the context ladder to the 16384 ceiling |
| [`logs/context-ceilings-2026-08-17/`](logs/context-ceilings-2026-08-17/) | Four models pushed to failure at depth, with the two distinct memory failure modes separated and a discarded reading kept and explained |
| [`logs/sdma-interrupt-2026-08-17/`](logs/sdma-interrupt-2026-08-17/) | Whether the SDMA completion interrupt arrives, using the trap instrumentation from bc250-rocm-working: 31 interrupts at boot, none for either side of the 16 KiB threshold, and the interrupt-handler rings 1 and 2 found zeroed rather than left alive |
| [`logs/recipe-retest-2026-08-17/`](logs/recipe-retest-2026-08-17/) | Retest of two recipe restrictions written before the allocation-reuse flush: memory mapping and one-benchmark-per-invocation, both now unnecessary |
| [`logs/`](logs/) | Captured run logs: correctness, rocBLAS sweeps, RustiCL comparison, inference, benchmarks, older-kernel attempt |
| [`logs/campaign-rerun-2026-08-18/`](logs/campaign-rerun-2026-08-18/) | The five-model campaign repeated on the current stack. Correctness unchanged and throughput unchanged within about one percent on four of five models; the 8B decode figure moved and is chased down in the section on it |
| [`logs/historical-sources/`](logs/historical-sources/) | Backing logs recovered from the board for figures quoted in this document whose original run directories were never shipped, one file per figure |
| [`logs/flush-cost-2026-08-18/`](logs/flush-cost-2026-08-18/) | The runlist flush toggled across 3, 1, 0 and back within one boot: what the map-side bit costs (nothing measurable) and what happens without any of it (an aperture violation) |
| [`logs/depth8192-comparison-2026-08-18/`](logs/depth8192-comparison-2026-08-18/) | Three models compared at a depth where all of them keep graph capture on, which the depth-16128 comparison could not do. Presented in blocks and not interleaved, so it is superseded for the variance question by the counterbalanced run and kept because the flush-cost sweep shares its harness |
| [`logs/clean-build-2026-08-18/`](logs/clean-build-2026-08-18/) | A clean tree at the campaign commit with only the three shipped patches, measured against the instrumented working tree in one boot. Refutes the idea that instrumentation moved the numbers, and shows the reported error bars understate the real spread about threefold |
| [`logs/kqv-remeasure-2026-08-18/`](logs/kqv-remeasure-2026-08-18/) | The KQV precision arm rebuilt and measured again, after its figures were found cited with no surviving log. Two reproduce to four decimals; the corrupted value does not reproduce as a value at all |
| [`logs/integrated-remeasure-2026-08-18/`](logs/integrated-remeasure-2026-08-18/) | The `prop.integrated` arm rebuilt and measured again after its figure was found cited in a patch header with no surviving log. The claim holds and the A/B/A returns bit-identically |
| [`logs/fp16-dispatch-2026-08-19/`](logs/fp16-dispatch-2026-08-19/) | Three hypotheses about the zeroed fp16 GEMM closed: kernel selection, instrumentation sensitivity, and argument corruption. Also the reason two traces appeared to show a corrupt alpha and did not |
| [`logs/reproduce-verify-2026-08-19/`](logs/reproduce-verify-2026-08-19/) | `reproduce.sh` run from a clean copy of the repository on the board, passing every stage. The script had only ever been run from the directory it was developed in |
| [`logs/patch-currency-2026-08-19/`](logs/patch-currency-2026-08-19/) | Whether the three llama.cpp patches are still needed and still apply, checked against upstream master 174 commits after the base this work measures at |
| [`logs/fp16-pool-2026-08-19/`](logs/fp16-pool-2026-08-19/) | Whether the fp16 defect is about the pool temporary it writes into. It is not: bypassing the pool changes nothing |
| [`logs/membw-2026-08-19/`](logs/membw-2026-08-19/) | The 432 GB/s memory-bandwidth ceiling, measured from the shipped source. It underpins every share-of-bandwidth figure here and had no backing log until an check of whole-number claims found it |
| [`logs/torch-probe-2026-08-19/`](logs/torch-probe-2026-08-19/) | The eleven-operation PyTorch probe on both builds, run because the figures it backs had no shipped log. It corrects the stock wheel from 1 of 11 to 3, and shows why `LD_LIBRARY_PATH` cannot substitute the wheel's bundled rocBLAS |
| [`logs/torch-train-2026-08-19/`](logs/torch-train-2026-08-19/) | The 50-step training loop against its CPU reference, run twice. Confirms the per-step loss agreement and records the accumulated parameter difference the write-up had been leaving out |
| [`logs/sdma-firmware-2026-08-19/`](logs/sdma-firmware-2026-08-19/) | The navi12 microcode substitution that fixed SDMA: the full size sweep, the correctness gates with SDMA enabled, and the ABBA throughput comparison showing it buys nothing measurable here |
| [`logs/sdma-sizes-2026-08-19/`](logs/sdma-sizes-2026-08-19/) | Where working SDMA helps and where it hurts, by transfer size: indistinguishable below the threshold, 13 percent faster just above it, four times slower at 16 MiB |
| [`logs/pasid-diagnosis-2026-08-19/`](logs/pasid-diagnosis-2026-08-19/) | Instrumented proof that the gfx10 PASID flush never matches a VMID on this ASIC and is a silent no-op, plus a direct VMID fallback that was built, tested and rejected |
| [`logs/kernel-718-2026-08-19/`](logs/kernel-718-2026-08-19/) | The full patch set ported to kernel 7.1.8 and measured against 7.1.5: both gates bit-identical, throughput within noise. Extends the kernel-independence finding to a fifth release |
| [`logs/vmid-flush-2026-08-20/`](logs/vmid-flush-2026-08-20/) | Two VMID-level replacements for the runlist rebuild, both built and both rejected, against a workload the control shows does discriminate |
| [`logs/sdma-depth-2026-08-20/`](logs/sdma-depth-2026-08-20/) | Decode at depth with SDMA on against off, alternated. Closes the SDMA re-test, and shows a single-sample version of the same comparison producing a 25 percent gap that is not there |
| [`logs/fp16-flashattn-2026-08-20/`](logs/fp16-flashattn-2026-08-20/) | The fp16 defect across flash attention and compute type as a full 2x2, adding the f32 control cell the 18 August flash-attention arm did not have |
| [`logs/deep-decode-faults-2026-08-20/`](logs/deep-decode-faults-2026-08-20/) | Thirty deep-decode runs hunting the SIGBUS, which did not reproduce, and a GPU memory fault that dmesg did not log |
| [`logs/fp16-arch-2026-08-20/`](logs/fp16-arch-2026-08-20/) | The fp16 defect disappearing when the board is driven as gfx1010, which moves the suspicion from the fp16 path to the gfx1013 kernels |
| [`logs/fp16-library-2026-08-20/`](logs/fp16-library-2026-08-20/) | Two earlier attempts to vary the library, both blocked, recorded because the blocks explain why the dual-architecture build was needed |
| [`logs/fp16-kernel-isolation-2026-08-20/`](logs/fp16-kernel-isolation-2026-08-20/) | Three blocked attempts to separate device identity from GEMM code objects, and why the two are coupled by the hardware, not by the experiment |
| [`logs/rocblas-recovery-2026-08-20/`](logs/rocblas-recovery-2026-08-20/) | The native rocBLAS destroyed by a build following a symlink, and reconstructed from two partial copies; verified against the established gate values |
| [`logs/torch-pristine-2026-08-20/`](logs/torch-pristine-2026-08-20/) | A genuinely stock PyTorch ROCm wheel on this board, which aborts instead of partially working, correcting two earlier figures that both came from modified wheels |
| [`logs/rocblas-rebuild-attempt-2026-08-20/`](logs/rocblas-rebuild-attempt-2026-08-20/) | An incomplete rebuild of the native rocBLAS, four blockers cleared and the fifth recorded, plus the finding that the build is not currently reproducible from what the board holds |
| [`logs/fp16-solutions-2026-08-20/`](logs/fp16-solutions-2026-08-20/) | Every Tensile solution for the failing shape exercised directly: none produces zeros and all six agree, so no single kernel is the culprit and the defect still needs the running model |

## References

- [ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313): BC-250 system freeze after compute
  workloads. anrp and ahorek found `flush_pasid_uses_kiq = false`. Still open, and AMD was
  engaged in the thread, as of an August 2026 check; both are statements about that check rather
  than about the thread today.
- [Mesa MR !33116](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/33116) and
  [Mesa 25.1 release notes](https://docs.mesa3d.org/relnotes/25.1.0.html): RADV disables the
  gfx1013 compute-only queue (commit `7271b8ee`). The MR is by Ivan Avdeev (`provod` on GitLab,
  `w23` on GitHub), a community contributor, not an AMD employee and not "RADV's author".
- [Mesa issue #11982](https://gitlab.freedesktop.org/mesa/mesa/-/issues/11982): "AMD
  CYAN_SKILLFISH support", the support discussion. Closed, as of an August 2026 check.
- [ROCm/rocm-libraries PR #8838](https://github.com/ROCm/rocm-libraries/pull/8838) by boondocklabs:
  adding gfx1013 support to rocBLAS and Tensile. Unmerged when this was written, and that is why the
  native build in this repository is necessary; the pull request itself is the place to check whether
  that is still true.
- [kernel bug #216645](https://bugzilla.kernel.org/show_bug.cgi?id=216645): a different system (a
  Dell laptop with a Navi/RDNA1 RX 5600M) hanging with "Fence fallback timer expired" and amdgpu
  interrupts ceasing. Not a BC-250 report, but the same fence-fallback / lost-interrupt symptom this
  board prints every boot, so it is useful background.
- [duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock): the 40-CU unlock
  and the module-build pipeline reused here.
- [GabriWar/bc250-rocm-working](https://github.com/GabriWar/bc250-rocm-working): identified the
  stale-translation-on-reuse mechanism by driver instrumentation and demonstrated the
  runlist-rebuild flush that this repo ports and validates; also documented the session-drift
  measurement hazard that this repo's interleaved A/B protocol follows.
- [DryhoppedIPA/bc250-gfx1013-fix](https://github.com/DryhoppedIPA/bc250-gfx1013-fix): identified
  the graphics-side compute-queue corruption as a threadgroup-dimension dispatch-mode issue with
  a one-line RADV workaround, and repairs the queue lifecycle; their patched RADV was built and
  verified here (it exposes the dedicated compute queues and runs llama.cpp correctly, a few
  percent faster).
- [github.com/akandr/bc250](https://github.com/akandr/bc250): the related BC-250 Vulkan setup.
- [Preprint on Zenodo](https://doi.org/10.5281/zenodo.21364833): the write-up of these notes as a
  single paper (doi:10.5281/zenodo.21364833). The currently published version predates the working
  configuration and still carries the entanglement and firmware-limit conclusions that the
  sections above correct; a revised version is in preparation.

## Author and license

Copyright (c) 2026 Artur Andrzejczak, written with assistance from Claude.

| | |
|---|---|
| code | [AGPL-3.0-or-later](LICENSE) |
| documentation | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [`scripts/apply_runlist_flush.py`](scripts/apply_runlist_flush.py), [`scripts/apply_svmflush_generic.py`](scripts/apply_svmflush_generic.py) | GPL-2.0-only: they embed kernel C derived from GabriWar's work |
| [`patches/`](patches/) | each patch under its upstream project's licence, GPL-2.0-only for the kernel and MIT for llama.cpp. [`patches/PROVENANCE.md`](patches/PROVENANCE.md) records what is original here |
