# ROCm / HIP on the AMD BC-250 (gfx1013, Cyan Skillfish)

The BC-250 is a cheap ex-mining blade carrying an RDNA1-class APU (gfx1013, Cyan Skillfish) with
around 14 GiB of usable shared memory, 24 compute units by default and 40 with the community
unlock. Vulkan has worked on it for a while. ROCm largely did not, and this documents getting it
to.

Most of the ROCm compute stack works once a corrected TLB flush, hardware scheduling, the 40-CU
unlock and a flush-on-map-and-unmap workaround are in place. This page is the recipe and the
measurements. The investigation that produced them, including the conclusions it had to withdraw,
is in [INVESTIGATION.md](INVESTIGATION.md).

One caveat belongs up here rather than in the defect table. Under sustained load the board can still
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

## Making it work

**1. Patch and build the driver module.** Two scripts, in this order, each taking the amdkfd
directory of a kernel tree:

    python3 scripts/apply_runlist_flush.py     <tree>/drivers/gpu/drm/amd/amdkfd
    python3 scripts/apply_svmflush_generic.py  <tree>/drivers/gpu/drm/amd/amdkfd

The first is a hand-port of GabriWar's runlist-rebuild flush, made runtime-switchable; the second
extends it to the SVM map side, which is this repository's part. Those two produce
`bc250_flush_by_runlist` and nothing else, which is worth stating because the
check below expects three. The other two come from elsewhere:
`bc250_flush_pasid_kiq` is added to `gmc_v10_0.c` by the embedded patcher in
[`scripts/ladder_prep_rung.sh`](scripts/ladder_prep_rung.sh) (the `FLUSHPARAM` step, which turns
`flush_pasid_uses_kiq` into a module parameter defaulting to the stock behaviour), and
[`patches/amdgpu-flush-pasid-mmio.patch`](patches/amdgpu-flush-pasid-mmio.patch) is the same change
hardcoded rather than parameterised, which works but leaves nothing to set at boot.
`bc250_cc_write_mode` comes from the community 40-CU unlock referenced under
[References](#references) and is not produced by anything here. So the module this repository
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
a parameter this board was configured with rather than as a demonstrated requirement.

Consider adding `amdgpu.gpu_recovery=0`, understanding what it does and does not buy: measured
against a real fault, it prevents the reset and keeps the host alive, but the GPU stays unusable
until reboot and still looks healthy to enumeration
([`logs/fault-usability-2026-08-24/`](logs/fault-usability-2026-08-24/)). A GPU reset does not
appear to be survivable on this
board, so the rare fault described under Known defects takes the whole machine down rather than
just the process. The driver source suggests why: neither MODE1 nor MODE2 has an implementation for
this chip, both report success anyway, and the register state captured after a "reset" is the
state from before it. Upstream also lists this chip as having recovery disabled by default, but
that list is unreachable on devices without RAS support, which is why the parameter has to be set
by hand. A PCI function-level reset, tried as an alternative, keeps the host alive but loses the GPU
until power-off ([`logs/reset-smu-gc-2026-09-14/`](logs/reset-smu-gc-2026-09-14/)). Two small driver
changes, in [`patches/amdgpu/`](patches/amdgpu/), were tested as runtime-switchable equivalents:
making the recovery default reachable refused every
KFD reset request at the driver default and kept the host up, four trials of four, and making the
unimplemented resets fail rather than report success kept the host up and the GPU computing after a
deliberate reset, four of four, though the reboot that followed needed a power cycle each time. A
gfx9-style per-queue reset ported to gfx10 ran and did not recover the queue
([`logs/reset-honest-2026-09-15/`](logs/reset-honest-2026-09-15/)). Rebinding the driver hangs the
host the same way, even on a healthy GPU, and s2idle suspend does not return on this board at all
([`logs/rebind-recovery-2026-09-15/`](logs/rebind-recovery-2026-09-15/),
[`logs/suspend-recovery-2026-09-15/`](logs/suspend-recovery-2026-09-15/)). That parameter stops the driver requesting a reset on the path those faults take.
It is left out of the line above because it means a wedged GPU stays wedged until reboot instead of
being reset, which is a trade rather than a pure gain. What it leaves behind is not a guess: a
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
`LD_LIBRARY_PATH`. Expect this step to need work: the script is a worked example rather than an
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

The script only touches helpers that match the broken variant and needs a CPU with F16C. Run it on
the llama.cpp `libggml-hip.so` as well, which carries the same pair.

Confirm the native library is the one actually loaded rather than assuming `LD_LIBRARY_PATH` won:
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

## llama.cpp inference

![ROCm against Vulkan across six models](figures/fig-rocm-vs-vulkan.png)

Tokens per second, same build on both backends, every row gated on perplexity under a matching
configuration. The build is verified: every log behind this table reports `build: 7ba604f (1)`. This used
to say "same build and boot", which the timestamps do not support. The Vulkan figures were taken on
12 August around 20:15 and four of the five HIP rows on 13 August around 09:25, thirteen hours
apart, so whether the board stayed up between them is not recorded either way. The last column is
HIP decode as a share of Vulkan decode on the same row, so
100 percent means the two backends tie. It is not a bandwidth figure: this document also carries a
share-of-the-402-GiB/s-ceiling column elsewhere, and the two are different quantities.

| model | HIP pp512 | VK pp512 | HIP tg64 | VK tg64 | decode share |
|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 805.6 | 1842.2 | 113.5 | 211.0 | 54 percent |
| qwen3-8B Q8_0 | 241.0 | 401.1 | 39.2 | 39.1 | 100 percent |
| deepseek-r1-14B Q4_K_M | 95.4 | 199.0 | 20.3 | 34.5 | 59 percent |
| qwen3-14B Q4_K_M | 97.4 | 202.8 | 21.5 | 34.2 | 63 percent |
| qwen3.6-35B-A3B MoE IQ2_M | 287.6 | 455.4 | 34.3 | 86.5 | 40 percent |

The figure carries a sixth model the table does not, qwen3.8-27B UD-IQ3_XXS at 69.23 against 97.94
prefill and 7.84 against 17.18 decode, measured on 17 August rather than in the 12 August campaign
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
39.2 in the table, so the row above is a good sample rather than a tight figure. An earlier
revision of this page called that row parity with Vulkan; on the pooled numbers ROCm is nearer 95
percent of Vulkan there, which is still much closer than on any other model tested. Read every
rate here to a few percent rather than to the second decimal.

**Correctness.** Wikitext perplexity, same model, command and boot, context 2048 over eight
chunks, flash attention on. The eight runs behind this table were taken within a fifteen-minute
window, so "same boot" holds here and is checkable from the timestamps.

One thing this table is not: it is not the gate that preceded the rates above. It is a stronger
re-gate, eight chunks on both backends, taken after every rate in the throughput table. The gates contemporaneous with the rates are the
two-chunk HIP runs of 12 August in
[`logs/bench-fixed-2026-08/`](logs/bench-fixed-2026-08/), and for the four models re-measured on
13 August the nearest matching-configuration gate is about thirteen hours earlier rather than in
the same session. Every model is gated; the ordering was tidier in the telling than in the run.
Perplexity depends on how much text is evaluated, so figures here are
only comparable at the same chunk count; elsewhere in this repository the 8B is often gated over
two chunks instead, which gives 9.0975 rather than 7.3503:

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
why. The first: the ROCr runtime reports memory access faults the kernel log does not carry, and
twenty-eight repeats of the deepest 8B measurement produced one such fault visible only in the process
output ([`logs/deep-decode-faults-2026-08-20/`](logs/deep-decode-faults-2026-08-20/)). The second
is worse, because it was a mistake rather than a limitation. That run was described here as having
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

## GPGPU

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
to 432 from a fresh run rather than a repeat to one decimal.

About 61 percent of the 7.68 TFLOP/s FP32 peak, from an untuned Tensile build. Re-measured on the
current configuration, the three large sizes reproduce within a couple of percent; the small ones
need the first call excluded, since one cold GEMM at N=512 costs 6.8 ms against about 0.16 ms warm
and swamps a twenty-iteration average
([`logs/defects-recheck-2026-08-22/`](logs/defects-recheck-2026-08-22/)). FP64 DGEMM reaches
about 95 percent of its rate peak. Measured streaming-read memory bandwidth is 432 GB/s (402
GiB/s), reproducing to 0.12 percent across three runs
([`logs/membw-2026-08-19/`](logs/membw-2026-08-19/)).

**PyTorch** works when built from source for gfx1013 (`PYTORCH_ROCM_ARCH=gfx1013`): 11 of 11
operations in the probe including fp16 matmul, and a 50-step training loop whose losses track a
CPU reference to within 1.799e-05 at every step, returning an identical final loss of 0.00048 on
all fourteen runs of an eight-hour soak. Accumulated parameter difference after those fifty steps
is 9.312e-03, past the 1e-3 threshold the script itself checks, so its built-in verdict reads as
disagreement; that is drift between two backends rather than a defect, and the per-step loss
agreement is why ([`logs/torch-train-2026-08-19/`](logs/torch-train-2026-08-19/)).

The stock wheel is not usable on this board at all. Installed fresh, `torch 2.9.1+rocm6.4` aborts
at the first library-dispatched operation, shipping no Tensile library for gfx1013 or any gfx101x.
Copying the native kernels into it stops the abort and reaches only 1 of 11, and it cannot be done
through `LD_LIBRARY_PATH` in any case, since `torch/lib` carries `RPATH $ORIGIN` and the bundled
library wins. Build from source. Build notes and the distribution-ROCm fixes are in
[`patches/pytorch/`](patches/pytorch/); two earlier figures for the stock wheel, and why both were
wrong, are in [`logs/torch-pristine-2026-08-20/`](logs/torch-pristine-2026-08-20/).

## Known defects

| defect | fix or workaround | status |
|---|---|---|
| PASID TLB flush covers nothing under hardware scheduling: silent wrong results, KIQ freeze | `amdgpu.bc250_flush_pasid_kiq=0` | fixed here, not upstream |
| Software-scheduler eviction path wedges sustained compute | do not set `amdgpu.sched_policy=2` | understood; 2x2 factorial at both CU counts |
| Allocation reuse on the KFD SVM paths faults after free and realloc | `amdgpu.bc250_flush_by_runlist=3` | fixed here; costs less than run-to-run noise. Lighter replacements tested and rejected: MMIO and SDMA invalidation of the assigned VMID (the request latches, the ACK never sets), rewriting its page-table base, and a rebuild filtered to one PASID ([`logs/tlb-alt-2026-09-15/`](logs/tlb-alt-2026-09-15/)) |
| rocBLAS ships no gfx1013 code objects | native build (PR #8838 approach) | fixed by building; the PR was closed unmerged by the stale bot on 2026-09-09 |
| PyTorch ships no gfx1013 code objects | build with `PYTORCH_ROCM_ARCH=gfx1013` | fixed by building |
| llama.cpp `prop.integrated` regression produces plausible-looking wrong output | [`patches/llamacpp/0001-hip-integrated-false.patch`](patches/llamacpp/0001-hip-integrated-false.patch) | bisected to c7d8722; the bisect's own output was not kept, so what is captured is the effect at that code line ([A/B/A](logs/integrated-remeasure-2026-08-18/)) rather than the search |
| llama.cpp KQV fp16 accumulation corrupts batched attention, worse with context but present at 1024 | [`patches/llamacpp/0002-kqv-f32-precision.patch`](patches/llamacpp/0002-kqv-f32-precision.patch) | fixed here, not upstream |
| gfx1013 missing from llama.cpp's RDNA1 macro: garbled generation, and quantized matmul much slower | [`patches/llamacpp/0003-gfx1013-rdna1-macro.patch`](patches/llamacpp/0003-gfx1013-rdna1-macro.patch) | fixed here, not upstream. The garbled output is captured on both arms ([`logs/macro-remeasure-2026-08-18/`](logs/macro-remeasure-2026-08-18/)); the speed figure is not, see INVESTIGATION.md |
| HIP graph instantiation fails past a primed depth of 12000 on the 14B | `GGML_CUDA_DISABLE_GRAPHS=1` | workaround is not throughput-neutral, see below. The depth, the failing call and the model are stated more precisely in this repository than its captures support; what is captured is one failing primed-depth run on deepseek-r1-14B and no run of that configuration with the flag set (see INVESTIGATION.md) |
| fp16 cuBLAS path returns an all-zero layer-0 value projection ([root cause](logs/fp16-root-cause-2026-09-15/)) | `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, or repair the libraries with [`scripts/fix_half_helpers.py`](scripts/fix_half_helpers.py) | **root cause found, fixed locally, not upstream.** Not a gfx1013 kernel defect: Fedora 43's ROCm compiler-rt builtins archive (`rocm-clang-runtime-devel-19-14.rocm6.4.2`) carries `__extendhfsf2`/`__truncsfhf2` built for an integer-register convention, while ROCm clang 19 passes half values in `%xmm0`. The native rocBLAS links them in, so converting alpha returns whatever was left in a register; when that is zero, rocBLAS hands Tensile a K=0 problem and the GEMM returns zeros. Replacing the two helpers with F16C instructions in a copy of the library gives 9.1117 on qwen3-8B four runs of four (f32 9.0975) and 7.7645 on qwen3-14B against 16.7385 unrepaired. The llama.cpp HIP backend built with the same toolchain carries the same pair (f16 clamp and fill paths). With both repaired libraries on the path the default fp16 path needs no environment variable (9.1117 and 7.7645 again). A test program shows `-mf16c` avoids the helpers; a rocBLAS rebuilt with it, or with a correct archive, was not tried. History of the hunt, including the gfx1010 result and the trace that hid it, in [INVESTIGATION.md](INVESTIGATION.md) and [`logs/fp16-scalar-2026-09-15/`](logs/fp16-scalar-2026-09-15/) |
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

## Limits

- One board, one stack. Nothing here says another BC-250 behaves the same way.
- Decode at depth varies run to run by as much as 15 percent on some models and under 1 percent on
  others, in one boot, with the clock pinned (residency at 1500 MHz is 83 to 85 percent in every
  run) and temperature and memory flat. The cause is not established. A memory-bandwidth
  explanation was the working theory and is not supported: measured with model order rotated, the
  coefficients of variation are 0.7, 6.0 and 4.6 percent at 29, 46 and 80 percent of the bandwidth
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
  until then SDMA could not complete a transfer at all. It is therefore a constant behind almost
  everything here rather than a setting that was chosen, and now that the microcode substitution
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
  rebuilding first. The build tree and the probes are still present.
- The board boots with `mitigations=off`, which is worth naming since it flatters every CPU-side
  comparison here. It does not
  affect the GPU figures, but every CPU-side number quoted for comparison was taken with CPU
  speculative-execution mitigations disabled, which flatters the host: the PyTorch CPU reference of
  about 20 seconds against the GPU's 0.26 is the case where this matters most.
- What a fault under `amdgpu.gpu_recovery=0` leaves behind has now been measured, and the answer is
  worse than the expectation this page once carried. A natural fault arrived after
  about 190 rounds: the usual chain, page fault to preemption failure to runlist rebuild `-62`, and
  no reset at all, exactly as the parameter promises. The host stayed up. But the GPU did not come
  back. Every GPU process for the next hour died, seven of them, the last segfaulting inside
  `libamdhip64`, while the driver logged 1193 preemption failures. Worse, the wedge is not visible
  to any ordinary check: the device stays on the bus, the runtime still enumerates it and still
  reports 14 GiB free. Only actually running something fails
  ([`logs/fault-usability-2026-08-24/`](logs/fault-usability-2026-08-24/)). The parameter is still
  worth setting, since a live host is recoverable by a reboot when you choose and a hung one is not,
  but it buys a clean shutdown rather than continued service. An ordinary reboot restores the board
  fully, so the wedge is a state rather than damage. That is one fault, observed once; whether every
  fault leaves the machine in the same state is untested.

## Reproducing

[`reproduce.sh`](reproduce.sh) builds and runs the probes: rocBLAS code objects and SGEMM against
the system library, the override and the native build; compute correctness through the graphics
queue and the compute queue; and a CU-count check. It gates on the module and scheduler
configuration and explains what is wrong rather than producing misleading output. Run from a clean
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

Ten of the scripts are audits rather than experiments, and each exists because something got past
the previous version of it. Four of them report a non-zero number on a healthy tree, so a reader
running them should know what to expect: `audit_fault_checks.sh` names 25 harnesses that count
faults with a pattern predating a kernel message change, left as they are because rewriting a
counter in a harness whose run is already logged would change what that log means;
`audit_single_capture.py` lists 11 pages resting on one capture, which is a prompt to say so rather
than a defect; `audit_citations.py` reports 4, one of them real and disclosed on its own page;
and `audit_counts.py` reports 41 counted claims of which 10 have no literal match, most being
derived by counting rather than quoted. The other six should report zero.
[`scripts/audit_figures.py`](scripts/audit_figures.py) checks that every figure quoted here still
has a log behind it, matching on numeric boundaries and allowing for rounding, and excluding both
the log READMEs and the per-tensor instrumentation dumps, since a figure found only in prose
validates itself and a dump of 79000 arbitrary floats will match almost anything.
[`scripts/audit_env_vars.sh`](scripts/audit_env_vars.sh) checks that every environment variable
relied on is actually read by the library that would have to read it; the run is in
[`logs/env-audit-2026-08-26/`](logs/env-audit-2026-08-26/), twelve of fourteen read and the two
compile-time ones not.
[`scripts/audit_links.py`](scripts/audit_links.py) checks the cross-references, including ones
broken by line wrapping and paths quoted in inline code rather than as links.
[`scripts/audit_logs.sh`](scripts/audit_logs.sh) checks that every log directory says what produced
it. [`scripts/audit_orphans.py`](scripts/audit_orphans.py) checks that no log directory or script is
unreachable from any document, [`scripts/audit_formatting.py`](scripts/audit_formatting.py) catches
the damage automated edits leave in prose, including em dashes and words broken across a line wrap,
[`scripts/audit_counts.py`](scripts/audit_counts.py) tests counted claims, the kind that say how many
runs of how many did something, against the logs, and [`scripts/audit_single_capture.py`](scripts/audit_single_capture.py) finds
pages that rest on a single capture without saying so.
[`scripts/audit_citations.py`](scripts/audit_citations.py) pairs each quoted string with the log
directory cited beside it and asks whether that directory contains it, which is the one check that
would have caught two sentences here that were true and cited the wrong experiment.
And [`scripts/audit_fault_checks.sh`](scripts/audit_fault_checks.sh) checks that no harness
counts GPU faults using `dmesg` alone, which cannot see a fault from a run that ended by resetting
the board.

## How this was arrived at

[INVESTIGATION.md](INVESTIGATION.md) is the full account: how each defect was found, what the
measurements were, and the conclusions this work published and later had to withdraw. The
withdrawals share one cause worth stating here, since it is the most transferable part of the work.
Every one of them came from comparing conditions that differed in more than one way, usually
because a workaround adopted early had quietly become part of the apparatus. Everything that
survived came from an intervention on a single variable: a module parameter toggled live within one
boot, a two-by-two factorial, one source line reverted and restored, a git bisect, a byte-level
bracket.

A second class of error showed up later and is worth separating from the first, because no amount
of care about experimental design catches it. Several claims here rested on an instrument that
could not have detected what it was being used to rule out. Fault counts came from `dmesg` after
the board had rebooted, when it necessarily reports the boot that followed the crash. A figure
audit passed a number because it appeared inside an unrelated longer number, and passed others
because they appeared in prose written by the author, which made the check circular. Each time the
instrument returned nothing and the nothing was reported as evidence. The cheap defence is to ask,
before believing a negative result, what a positive one would have looked like and whether the
instrument could have produced it.

## References

- [ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313): BC-250 freeze after compute
  workloads; anrp and ahorek found `flush_pasid_uses_kiq = false`. Open as of an August 2026 check.
- [ROCm/rocm-libraries PR #8838](https://github.com/ROCm/rocm-libraries/pull/8838): gfx1013 support
  in rocBLAS. Closed unmerged by the stale bot on 2026-09-09, which is why the native build here is
  necessary.
- [Mesa MR !33116](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/33116) by Ivan Avdeev
  (w23), a community contributor: disables the gfx1013 compute queue in RADV.
- [akandr/bc250](https://github.com/akandr/bc250): the board itself and its Vulkan setup.
- [duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock),
  [GabriWar/bc250-rocm-working](https://github.com/GabriWar/bc250-rocm-working) (whose
  runlist-rebuild flush is the fix step 1 above ports, whose SDMA trap instrumentation this used,
  and whose navi12 microcode tip fixed SDMA outright),
  [DryhoppedIPA/bc250-gfx1013-fix](https://github.com/DryhoppedIPA/bc250-gfx1013-fix): community
  work this depends on.
- [Preprint on Zenodo](https://doi.org/10.5281/zenodo.21364833). Read it against this page rather
  than on its own. It predates the working configuration described here, and several of its
  conclusions have since been withdrawn rather than merely superseded, including that the SDMA
  failure was genuine to the board and that the host-side SIGBUS was a rare defect of its own. A
  revision is pending.

## Author and license

Author: Artur Andrzejczak. Prepared with assistance from Claude.

Code: [AGPL-3.0](LICENSE). Docs: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
