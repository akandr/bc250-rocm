# The zeroed fp16 GEMM: scalars, launch errors, and a trace that makes it disappear, 2026-09-15

Harness [`scripts/fp16_scalar_probe.sh`](../../scripts/fp16_scalar_probe.sh) plus one-off runs recorded
below; instrumentation [`scripts/apply_fp16_scalar_probe.py`](../../scripts/apply_fp16_scalar_probe.py)
and three smaller additions made on the board the same day, all captured in
`ggml-cuda.instrumentation.diff`. Kernel 7.1.8, llama.cpp 7ba604f with the existing BC250
instrumentation, native gfx1013 rocBLAS, qwen3-8B Q8_0, `-fa on -c 2048`, `HSA_ENABLE_SDMA=0`,
experiment amdgpu module with production settings. The defect and everything eliminated before
today are in the fp16 section of [INVESTIGATION.md](../../INVESTIGATION.md).

## 1. Scalars

The idea: a kernel receiving alpha = 0 and beta = 0 would produce exactly the recorded failure,
and Tensile kernels take their scalars as launch arguments.

- **Device pointer mode** (`BC250_PTRMODE_DEVICE=1`): the same GEMM through `rocblas_gemm_ex` with
  alpha and beta in device memory. Status 0, defect unchanged: 20.4944, 21.5478, 18.9806 against
  17.8338 and 19.9354 without it and the f32 reference 9.0975 (`log`, `f16_ptrdev1.log`). Moving the
  scalars to a different transport does not matter.
- **Stamped output** (`BC250_TEMP_SENTINEL=1 BC250_TEMP=1`), sums of each GEMM output
  (`f16_stamp_beta0.sums.txt`): position 37 sums to exactly 0 over a buffer stamped beforehand, so
  the kernel ran and wrote.
- **Stamped output with beta 1** (`BC250_BETA1=1`, `f16_stamp_beta1.sums.txt`): position 37 sums to
  555008, which is exactly the stamp (0x3c3c as half is 1.05859375, times 524288). So the output was
  beta times C with nothing added. This is weaker evidence than it looks: with a stamp of about 1.06
  the absolute sum measures the signed sum of the product, which can cancel, and neighbouring calls
  differ from the stamp by only -1892 and +831. It is consistent with a zero product and would not
  establish one on its own; the beta-0 exact zero is the strong observation.

## 2. Launch errors

`BC250_LASTERR=1` synchronises and reads the sticky HIP error after every fp16 GEMM
(`f16_lasterr.lines.txt`): all 288 report `hipSuccess` while outputs 37, 73, 109, 181, 217, 253 and
254 are zero. No error is raised anywhere.

## 3. The runtime trace suppresses it

Chunk 1 only, one run per row unless stated. Rows marked * are transcribed from the session output
because their logs were deleted for size (the level-3 and level-4 logs are 22 and 41 MB); the
others have files here or in `timing_and_mask_results.txt`.

| arm | perplexity | zeroed outputs |
|---|---|---|
| f32 reference (`timing_f32ref.log`) | 7.2672 | none (no fp16 path) |
| control, three runs | 14.1344 | 37 73 109 |
| control, one run | 11.7443 | 73 109 |
| `AMD_LOG_LEVEL=4`, two runs | 7.2768 | none |
| `AMD_LOG_LEVEL=3` | 7.2768 | none |
| `AMD_LOG_LEVEL=2` * | 14.1344 | 37 73 109 |
| `AMD_LOG_LEVEL=1` | 14.1344 | 37 73 109 |
| `AMD_SERIALIZE_KERNEL=3` (`timing_serialize3.log`) | 14.1344 | 37 73 109 |
| `AMD_DIRECT_DISPATCH=0`, two runs * (`timing_dd0.log`) | 16.2484 | 37 73 109 110 |
| `AMD_DIRECT_DISPATCH=1` * | 14.1344 | 37 73 109 |
| `HSA_ENABLE_INTERRUPT=0` * | 14.1344 | 37 73 109 |
| 1 ms sleep before each fp16 GEMM (`timing_delay1ms.log`) | 14.1344 | 37 73 109 |
| 10 ms sleep * | 14.1344 | 37 73 109 |
| 100 us sleep * | 18.3329 | 37 38 73 109 |
| `hipPointerGetAttributes` on src0, src1, dst first, two runs (`timing_ptrattr1.log`) | 14.1344 | 37 73 109 |
| `bc250_flush_by_runlist=1` instead of 3, two runs * | 11.7443, 14.1344 | 73 109; 37 73 109 |
| level 3, `AMD_LOG_MASK=1` (API calls only), three runs | 7.2768 | none |
| level 3, `AMD_LOG_MASK=128` (kernel arguments only), two runs | 14.1344 | 37 73 109 |
| level 3, masks 48 (queue+signal), 64 (locks), 768 (copy), 18432 (init+code) * | 19.88, 14.13, 14.13, 14.13 | defect present in all |

7.2768 is the correct fp16 answer: the f32 reference is 7.2672, and the 0.13 percent gap is the
same ordinary half-precision difference as 9.1117 against 9.0975 at two chunks.

An earlier mask test using hexadecimal values (`AMD_LOG_MASK=0x80` and so on) is not in the table:
every one produced the same 335 lines as mask 0, so the runtime evidently did not parse them, and
those runs tested nothing.

What it shows. The defect is removed by ROCclr's API-call trace and by nothing else tried. It is not
the volume or cost of logging: the kernel-argument trace prints more lines, 108578 against 94482,
and leaves the defect intact, and both take the same 28 seconds of wall time. It is not slowness
before the GEMM (sleeps of 100 us to 10 ms), not waiting for the stream, not serialising launches,
not the dispatch mode, not interrupts, not a pointer lookup on the GEMM's buffers, and not the
runlist flush. Something the HIP API trace wrapper does on every API call changes the result. The
wrapper in the current `rocm-systems` source formats each call's arguments and its duration, and
nothing in that formatting has an evident side effect, but that source is newer than the 6.4.2
library on the board, so this is not settled by reading.

Which API call's tracing matters was not established by this section; section 4 below takes that
further with an `LD_PRELOAD` shim.

## Why it matters

(Superseded by section 5: the cause is a toolchain helper linked into rocBLAS, not the runtime.)

It is a reproducible handle on a defect that until now had none. The practical rule does not change:
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` avoids the path at no measurable cost.

## 4. Narrowing the trace effect (same day, later)

**Which API family.** [`../../patches/hip_api_shim.cpp`](../../patches/hip_api_shim.cpp) is an
`LD_PRELOAD` interposer that forwards the HIP API calls made by llama.cpp and rocBLAS and adds a side
effect to chosen families. Forwarding alone leaves the defect in place (11.7443, zeros at 73 and 109).
Its call counts for one chunk: `hipMemcpyAsync` 8007, `hipEventSynchronize` 7980, `hipEventRecord`
7976, `hipGetDevice` 5910, `hipGetLastError` 5497, `hipLaunchKernel` 4275, `hipStreamSynchronize` 220,
`hipMalloc`/`hipFree` 153, and `hipExtModuleLaunchKernel` 144, one per fp16 GEMM, which is how rocBLAS
launches its Tensile kernels. Printing a line to stderr on every call of a family changes nothing for
any family, nor for all of them together, 49189 lines (`shim_family_sweep.txt`). So it is not the
trace's output, or anything else that happens at the API boundary as seen from outside the runtime,
that removes the defect; it is something the runtime does internally when that log category is on.

**Host threads.** Pinning the whole process to one CPU, two or four keeps the defect
(`cpu_pinning.txt`: 16.2484, 14.1344, 14.1344, 11.7443). A race between host threads is therefore an
unlikely explanation.

**Kernel choice.** Forcing each of the six Tensile solutions rocBLAS offers for this shape (621 to 626,
none producing zeros in isolation per [`../fp16-solutions-2026-08-20/`](../fp16-solutions-2026-08-20/)) gives
zeros at 37, 73 and 109 every time, and with solution 621 forced the API trace still removes them
(`forced_solutions.txt`). All six forced runs give the same 14.1344; that is plausible for variants
differing in launch geometry, but it is also what an ignored index would produce, so this shows that
no forced solution escapes the defect rather than that each kernel was exercised.

**Where this leaves it** (as written before section 5, which supersedes it). Every externally controllable variable tested has been eliminated: scalars,
their transport, launch errors, kernel choice, delays, stream synchronisation, launch serialisation,
dispatch mode, interrupts, pointer attribute queries, host threads, and the runlist flush. What
removes the defect is internal to the HIP 6.4.2 runtime and switched on by its API log category. The
runtime source available (`rocm-systems` develop) is newer than the library on the board and shows no
side effect in that category's logging, so the next step is a build of the 6.4.2 runtime with the
API-category log calls individually disabled. The reproduction below uses only public tools and two
environment variables.

    # defect: a different wrong perplexity most runs, zeros visible with the BC250_TEMP probe
    GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 llama-perplexity -m qwen3-8b-q8_0.gguf -ngl 99 -fa on -c 2048 --chunks 1 -f wiki.test.raw
    # same command, correct result
    AMD_LOG_LEVEL=3 AMD_LOG_MASK=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 llama-perplexity ...

## 5. Bisecting the trace inside the runtime, and the root cause

**Update, same day: the cause is found and it is not in the runtime.** See
[`../fp16-root-cause-2026-09-15/`](../fp16-root-cause-2026-09-15/). The half-precision conversion
helpers linked into the native rocBLAS read the wrong register, so rocBLAS sometimes reads alpha as
zero and hands Tensile a K=0 problem. The readings above that point into the HIP runtime ("internal
to the HIP 6.4.2 runtime") are withdrawn; the trace changes leftover register
contents, not the computation. What follows is the path that led there.

The HIP 6.4.2 runtime was rebuilt from the Fedora `rocclr-6.4.2` source with
[`scripts/apply_hip_trace_filter.py`](../../scripts/apply_hip_trace_filter.py), which gates the API
trace per function. The rebuilt library reproduces both the defect and its suppression.

- **Which traced call** (`trace_filter_round1.txt`, `trace_filter_round2.txt`): tracing only the entry
  print removes the defect, tracing only the exit print does not. By name, tracing
  `hipExtModuleLaunchKernel` alone (479 lines) removes it, as do `hipLaunchKernel` and `hipMemcpyAsync`;
  `hipMemcpy`, events, streams, device and error queries do not.
- **What about that call** (`trace_filter_round3.txt`, `trace_filter_round4.txt`): at the entry of
  `hipExtModuleLaunchKernel`, a write to stderr, a clock read, a sleep, filling or zeroing 16 KiB of
  stack, and a matched no-op that returns immediately all remove the defect; the same clock read at
  `hipMemcpy` does not. A no-op cannot change a computation, so this pointed at state left behind by
  earlier code rather than at anything the runtime does. (`actions` in round 3 counts
  the line only the `write` arm prints, hence 144 there and 0 elsewhere.)
- **Heap state** (`heap_knobs.txt`): `GLIBC_TUNABLES=glibc.malloc.tcache_count=0` removes the defect
  three times at one chunk (7.2768) and at two chunks gives 9.1117, the correct fp16 value;
  `tcache_count=1` zeroes all 144 fp16 GEMMs (perplexity 6627258.6969). Perturbing freed memory
  (`MALLOC_PERTURB_`) and `mxfast=0` change nothing. The tcache knob made the defect switchable in a
  program with no llama.cpp in it, which is what exposed the chain on the root-cause page.
