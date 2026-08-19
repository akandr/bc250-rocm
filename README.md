# ROCm / HIP on the AMD BC-250 (gfx1013, Cyan Skillfish)

The BC-250 is a cheap ex-mining blade carrying an RDNA1-class APU (gfx1013, Cyan Skillfish) with
around 14 GiB of usable shared memory, 24 compute units by default and 40 with the community
unlock. Vulkan has worked on it for a while. ROCm largely did not, and this documents getting it
to.

Most of the ROCm compute stack works once a corrected TLB flush, hardware scheduling, the 40-CU
unlock and a flush-on-map-and-unmap workaround are in place. This page is the recipe and the
measurements. The investigation that produced them, including the conclusions it had to withdraw,
is in [INVESTIGATION.md](INVESTIGATION.md).

Everything here is one board, one software stack. Measurements are reproducible and the logs are
included; explanations are working theories. Corrections welcome.

Environment: Fedora 43, ROCm 6.4.2, LLVM/clang 19, Mesa 25.3 RADV for the Vulkan comparison, the
oberon governor at 1500 MHz, llama.cpp master 7ba604f. The kernel version is not an ingredient:
6.18.9, 6.18.16, 6.19.14 and 7.1.5 measure identically with the same patch set. Below 6.18 the
board does not come up at all.

## Making it work

**1. Patch and build the driver module.** Two scripts, in this order, each taking the amdkfd
directory of a kernel tree:

    python3 scripts/apply_runlist_flush.py     <tree>/drivers/gpu/drm/amd/amdkfd
    python3 scripts/apply_svmflush_generic.py  <tree>/drivers/gpu/drm/amd/amdkfd

Then build and install module-only with [`scripts/build_patched_amdgpu.sh`](scripts/build_patched_amdgpu.sh),
which rebuilds the initramfs. That last part is not optional: the running module comes from the
initramfs, and forgetting it is the most common way to spend a day measuring a module you are not
running.

After rebooting, confirm the patched module is the one loaded. The three parameters below exist
only in it, so if they are absent the stock module is running whatever is in `/lib/modules`:

    ls /sys/module/amdgpu/parameters/ | grep bc250
    # expect: bc250_cc_write_mode  bc250_flush_by_runlist  bc250_flush_pasid_kiq

Checking the file's checksum proves nothing here, since the module that matters is the one baked
into the initramfs.

**2. Boot with these arguments, and without `sched_policy`:**

    amdgpu.bc250_cc_write_mode=3 amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=3

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

**3. Set two environment variables for HIP processes:**

    export HSA_ENABLE_SDMA=0
    export GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32

The first avoids the SDMA path, which never completes a transfer above 16384 bytes. The second is
required for correctness on models carrying F16 weights.

**4. Build a native gfx1013 rocBLAS** with
[`scripts/build_rocblas_gfx1013.sh`](scripts/build_rocblas_gfx1013.sh) and put it on
`LD_LIBRARY_PATH`. The system rocBLAS has no gfx1013 code objects, only symlinks to the gfx1010
ones, and GEMMs abort against it. Small quantized models avoid rocBLAS entirely, so this step is
easy to think you got away with skipping.

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

**6. Verify.** Run [`reproduce.sh`](reproduce.sh) from the repository root; it reports the CU count,
which should read 40. Then gate real work on **both** perplexity and generated text, because they
catch different faults: with the RDNA1 macro entry missing, perplexity reads a healthy-looking
8.9425 against a correct 8.9442 while generation returns `The???????????????????????`.

## llama.cpp inference

![ROCm against Vulkan across five models](figures/fig-rocm-vs-vulkan.png)

Tokens per second, same build and boot on both backends, every row gated on perplexity before its
rate was recorded.

| model | HIP pp512 | VK pp512 | HIP tg64 | VK tg64 | decode share |
|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 805.6 | 1842.2 | 113.5 | 211.0 | 54 percent |
| qwen3-8B Q8_0 | 241.0 | 401.1 | 39.2 | 39.1 | 100 percent |
| deepseek-r1-14B Q4_K_M | 95.4 | 199.0 | 20.3 | 34.5 | 59 percent |
| qwen3-14B Q4_K_M | 97.4 | 202.8 | 21.5 | 34.2 | 63 percent |
| qwen3.6-35B-A3B MoE IQ2_M | 287.6 | 455.4 | 34.3 | 86.5 | 40 percent |

Decode is competitive, and closest on the 8B, where both backends are working against the same
memory-bandwidth limit at roughly three quarters to four fifths of the measured ceiling. Prefill trails Vulkan by 1.4x
to 2.3x, narrowing as models grow. That gap is between llama.cpp's quantized matmul kernels and
Vulkan's, not a library deficiency: tracing shows zero rocBLAS calls on the ROCm side at pp512.

Each rate above is a single `llama-bench` invocation, and that matters more than it looks.
`llama-bench` prints an error bar computed across repetitions *within* one invocation, which for
the 8B understates the spread across invocations by about threefold. Pooling every independent
measurement of 8B decode on record gives mean 37.34 with sd 1.20 over eleven runs, against the
39.2 in the table, so the row above is a good sample rather than a tight figure. An earlier
revision of this page called that row parity with Vulkan; on the pooled numbers ROCm is nearer 95
percent of Vulkan there, which is still much closer than on any other model tested. Read every
rate here to a few percent rather than to the second decimal.

**Correctness.** Wikitext perplexity, same model, command and boot, context 2048, flash attention
on:

| model | ROCm/HIP | Vulkan |
|---|---|---|
| qwen3-8B Q8_0 | 7.3503 +/- 0.232 | 7.3792 +/- 0.233 |
| qwen3-14B Q4_K_M | 6.3970 +/- 0.193 | 6.4548 +/- 0.195 |
| deepseek-r1-14B Q4_K_M | 6.0013 +/- 0.173 | 6.0416 +/- 0.175 |
| qwen3.6-35B-A3B MoE IQ2_M | 5.1887 +/- 0.134 | 5.2041 +/- 0.134 |

Every pair agrees far inside one standard error. Three endurance soaks (8h, 8.1h and 8h04m) each
returned a bit-identical gate value on every round, with no faults.

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
campaign. Prefill reaches further than decode on the largest model (the 27B processes a
16384-token prompt at 41.4 t/s but cannot generate at that depth), because generation needs the
whole cache resident alongside the weights.

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

About 61 percent of the 7.68 TFLOP/s FP32 peak, from an untuned Tensile build. FP64 DGEMM reaches
about 95 percent of its rate peak. Measured streaming-read memory bandwidth is 432 GB/s (402
GiB/s), reproducing to 0.12 percent across three runs
([`logs/membw-2026-08-19/`](logs/membw-2026-08-19/)).

**PyTorch** works fully when built from source for gfx1013 (`PYTORCH_ROCM_ARCH=gfx1013`): 11 of 11
probe operations including fp16 matmul, and a 50-step training loop whose losses stay within
1.799e-05 of a CPU reference at every step, returning an identical final loss (0.00048) on all
fourteen runs of an eight-hour soak. Over those fifty steps the accumulated parameter difference
reaches 9.312e-03, which is beyond the 1e-3 threshold the script itself checks, so its built-in
verdict reads as disagreement; that is accumulated floating-point difference between two backends
rather than a defect, and the per-step loss agreement is the reason for reading it that way
([`logs/torch-train-2026-08-19/`](logs/torch-train-2026-08-19/)). The
stock wheel manages 3 of 11, because it ships no gfx1013 code objects: the host-to-device copy,
fp32 `matmul` and fp32 `addmm` survive, and everything reaching one of torch's own kernels fails
with `invalid device function`. Replacing its bundled rocBLAS with the native build has been
reported to lift that to 4 of 11 and no further; note that this cannot be done with
`LD_LIBRARY_PATH`, since `torch/lib` is built with `RPATH $ORIGIN` and the bundled library wins
([`logs/torch-probe-2026-08-19/`](logs/torch-probe-2026-08-19/)). Build notes and the
distribution-ROCm build fixes are in [`patches/pytorch/`](patches/pytorch/).

## Known defects

| defect | fix or workaround | status |
|---|---|---|
| PASID TLB flush covers nothing under hardware scheduling: silent wrong results, KIQ freeze | `amdgpu.bc250_flush_pasid_kiq=0` | fixed here, not upstream |
| Software-scheduler eviction path wedges sustained compute | do not set `amdgpu.sched_policy=2` | understood; 2x2 factorial at both CU counts |
| Allocation reuse on the KFD SVM paths faults after free and realloc | `amdgpu.bc250_flush_by_runlist=3` | fixed here; costs less than run-to-run noise |
| rocBLAS ships no gfx1013 code objects | native build (PR #8838 approach) | fixed by building; PR still open |
| PyTorch ships no gfx1013 code objects | build with `PYTORCH_ROCM_ARCH=gfx1013` | fixed by building |
| llama.cpp `prop.integrated` regression produces plausible-looking wrong output | `patches/llamacpp/0001` | bisected to c7d8722; report prepared |
| llama.cpp KQV fp16 accumulation corrupts at long context | `patches/llamacpp/0002` | report prepared |
| gfx1013 missing from llama.cpp's RDNA1 macro: garbled generation, quantized kernels 9.2x slow | `patches/llamacpp/0003` | report prepared |
| HIP graph instantiation fails past a primed depth of 12000 on the 14B | `GGML_CUDA_DISABLE_GRAPHS=1` | workaround is not throughput-neutral, see below |
| fp16 cuBLAS path returns an all-zero layer-0 value projection | `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | **open.** Hits the default path, not an opt-in mode. Damages every corpus tested by a similar factor. Not kernel selection, not instrumentation, not corrupt arguments, not the pool temporary: all 36 calls are parameter-identical and only the first fails |
| SDMA never completes a copy above 16384 bytes | `HSA_ENABLE_SDMA=0` | **open.** Narrowed to the submission end: the queue is fully configured (ring, doorbell, writeback address) and its write pointer is zero, so nothing was ever placed in the ring |
| SIGBUS storing to the ROCr AQL queue during deep decode | none | **open observation.** Seen once in about twelve runs, not reproduced in a deliberate repeat |

Two notes on the workarounds. `GGML_CUDA_DISABLE_GRAPHS=1` is not neutral for throughput: on the
8B at depth 16128, measured in ABBA order so drift cancels, disabling capture is about 13 percent
faster, consistently across three rounds. An earlier measurement of this repository's own reported
the opposite sign, from a design that ran the two arms in blocks; that figure is withdrawn.
`HSA_ENABLE_SDMA=0` costs nothing measurable, and forcing the same blit path by a second mechanism
(`GPU_FORCE_BLIT_COPY_SIZE`) reproduces its effect exactly.

## Limits

- One board, one stack. Nothing here says another BC-250 behaves the same way.
- Decode at depth varies run to run by as much as 15 percent on some models and under 1 percent on
  others, in one boot, with the clock pinned (residency at 1500 MHz is 83 to 85 percent in every
  run) and temperature and memory flat. The cause is not established. A memory-bandwidth
  explanation was the working theory and is not supported: measured with model order rotated, the
  coefficients of variation are 0.7, 6.0 and 4.6 percent at 29, 42 and 80 percent of the bandwidth
  ceiling, so variability does not rise with utilisation.
- Effects of a few percent are hard to establish on this board at all. Two careful designs of the
  same graph-capture comparison returned differences of opposite sign, and only the counterbalanced
  one is trustworthy. Treat any small difference here, including ones stated above, as needing a
  counterbalanced repeat before it means anything.
- The wedge and freeze behaviour that dominated earlier work is routed around by this
  configuration rather than repaired, and its root cause is not known.
- Numbers quoted from a single benchmark invocation carry more uncertainty than the printed error
  bar suggests, as noted above.

## Reproducing

[`reproduce.sh`](reproduce.sh) builds and runs the probes: rocBLAS code objects and SGEMM against
the system library, the override and the native build; compute correctness through the graphics
queue and the compute queue; and a CU-count check. It gates on the module and scheduler
configuration and explains what is wrong rather than producing misleading output. Run from a clean
copy of this repository on the board it passes every stage, and the output is kept in
[`logs/reproduce-verify-2026-08-19/`](logs/reproduce-verify-2026-08-19/) so a reader can see what
passing looks like before running it.

The measurement harnesses are in [`scripts/`](scripts/), one per experiment, each with a header
saying what question it was written to answer. Raw output is in [`logs/`](logs/), one directory per
run with a README naming the harness that produced it. Two of the scripts are audits rather than
experiments: [`scripts/audit_figures.py`](scripts/audit_figures.py) checks that every figure quoted
in this repository still has a log behind it, and
[`scripts/audit_env_vars.sh`](scripts/audit_env_vars.sh) checks that every environment variable
relied on is actually read by the library that would have to read it, and
[`scripts/audit_links.py`](scripts/audit_links.py) checks the cross-references, including ones
broken by line wrapping, and [`scripts/audit_logs.sh`](scripts/audit_logs.sh) checks that every log
directory says what produced it.

## How this was arrived at

[INVESTIGATION.md](INVESTIGATION.md) is the full account: how each defect was found, what the
measurements were, and the conclusions this work published and later had to withdraw. The
withdrawals share one cause worth stating here, since it is the most transferable part of the work.
Every one of them came from comparing conditions that differed in more than one way, usually
because a workaround adopted early had quietly become part of the apparatus. Everything that
survived came from an intervention on a single variable: a module parameter toggled live within one
boot, a two-by-two factorial, one source line reverted and restored, a git bisect, a byte-level
bracket.

## References

- [ROCm/ROCm#6313](https://github.com/ROCm/ROCm/issues/6313): BC-250 freeze after compute
  workloads; anrp and ahorek found `flush_pasid_uses_kiq = false`. Open as of an August 2026 check.
- [ROCm/rocm-libraries PR #8838](https://github.com/ROCm/rocm-libraries/pull/8838): gfx1013 support
  in rocBLAS. Open as of an August 2026 check, which is why the native build here is necessary.
- [Mesa MR !33116](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/33116) by Ivan Avdeev
  (w23), a community contributor: disables the gfx1013 compute queue in RADV.
- [akandr/bc250](https://github.com/akandr/bc250): the board itself and its Vulkan setup.
- [duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock),
  [GabriWar/bc250-rocm-working](https://github.com/GabriWar/bc250-rocm-working),
  [DryhoppedIPA/bc250-gfx1013-fix](https://github.com/DryhoppedIPA/bc250-gfx1013-fix): community
  work this depends on.
- [Preprint on Zenodo](https://doi.org/10.5281/zenodo.21364833). The published version predates the
  working configuration described here. Update TBD.

## Author and license

Author: Artur Andrzejczak. Prepared with assistance from Claude.

Code: [AGPL-3.0](LICENSE). Docs: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)