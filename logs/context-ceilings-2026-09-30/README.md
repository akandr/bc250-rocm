# Context ceilings without llama-bench's copy of the cache, 2026-09-30 to 2026-10-10

## What was open

The context ceilings on the front page until October
([`logs/fedora44-ceilings-2026-09-16/`](../fedora44-ceilings-2026-09-16/)) were llama-bench's, not
ROCm's. After the depth fill, llama-bench copies the sequence state into a host buffer
(`llama_state_seq_get_data`, added upstream in PR #16944), also with `-r 1`, where the copy is never
used. The copy is as large as the KV cache. On ROCm clr pins it, pinned host memory counts against KFD's
memory limit, and with HIP graphs on, the first generated token is then left without the memory to
instantiate its graph. With the copy, qwen3-14B with an f16 cache generates at 13312 and fails at 14336.
At 14336, KFD's count climbs from 11299 MiB to the limit in the 20 seconds after the fill while free RAM
falls by 2.2 GiB, the size of the copy, and the first token's graph then fails. Without the copy the same
build generates at 16384 and at 24576 ([`snapshot-check/`](snapshot-check/)).

So every ceiling was measured again with the copy switched off: five models, three KV cache types, ROCm
in three variants and Vulkan.

## Method

Four arms:

- `hip`: ROCm as shipped, the thirteen-patch build, HIP graphs on. "ROCm" below means this arm.
- `hipng`: the same with `GGML_CUDA_DISABLE_GRAPHS=1`.
- `hippc`: the same with `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0`, which keeps the graphs and turns off clr's
  AQL packet capture.
- `vk`: Vulkan (RADV), the build the front page compares against.

One invocation per depth, `llama-bench -m MODEL -ngl 99 -fa on -ctk C -ctv C -p 0 -n 64 -d DEPTH -r 1`,
mmap on, built with [`scripts/llama-bench-no-depth-cache.patch`](../../scripts/llama-bench-no-depth-cache.patch)
and run with `LLAMA_BENCH_NO_DEPTH_CACHE=1` ([`scripts/build_llama_bench_nosnap.sh`](../../scripts/build_llama_bench_nosnap.sh)).
A depth generates if llama-bench prints its tg64 line, and fails otherwise.
[`scripts/ceilings_bisect.sh`](../../scripts/ceilings_bisect.sh) doubles the depth until one fails, then
bisects in steps of 1024 tokens until the bracket is an eighth of the one it started from. It does not
re-run what memory already implies: a depth that generated in the two earlier passes, which had the copy,
generates without it, a smaller cache cannot fail where a larger one fitted, and a larger one cannot fit
where a smaller one failed. 29 cells take their deepest pass from those earlier passes
([`earlier-passes/`](earlier-passes/)); no failure carries over.

After every failure a depth-0 run checks that the GPU still works, and a new GPU fault line in the kernel
log stops the campaign; neither happened. Every run samples KFD's accounting, GTT use and free memory
every five seconds (`runs-mem.tar.xz`).

The 1.5B with quantised caches ran last and in two stages, because its ceilings lie near and past a
million tokens and one run there takes up to a day: its eight cells were first bisected to half their
bracket, then to an eighth like the rest ([`scripts/ceilings_nosnap_chain.sh`](../../scripts/ceilings_nosnap_chain.sh),
[`scripts/ceilings_refine_chain.sh`](../../scripts/ceilings_refine_chain.sh)).

## Result

Deepest depth that generated / shallowest that failed ([`ceilings.tsv`](ceilings.tsv)):

| model | cache | ROCm | no graphs | no packet capture | Vulkan | Vulkan / ROCm |
|---|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | f16 | 393216 / 425984 | 425984 / 458752 | 425984 / 458752 | 458752 / 491520 | 1.17 |
| | q8_0 | 688128 / 737280 | 692224 / 745472 | 692224 / 745472 | 802816 / 860160 | 1.17 |
| | q4_0 | 1105920 / 1179648 | 1198080 / 1277952 | 1198080 / 1277952 | 1376256 / 1548288 | 1.24 |
| qwen3-8B Q8_0 | f16 | 32768 / 36864 | 32768 / 36864 | 32768 / 36864 | 45056 / 50176 | 1.38 |
| | q8_0 | 61440 / 65536 | 61440 / 68608 | 65536 / 73728 | 81920 / 92160 | 1.33 |
| | q4_0 | 114688 / 118784 | 118784 / 133120 | 114688 / 129024 | 163840 / 184320 | 1.43 |
| qwen3-14B Q4_K_M | f16 | 26624 / 29696 | 27648 / 30720 | 26624 / 29696 | 38912 / 43008 | 1.46 |
| | q8_0 | 51200 / 53248 | 51200 / 57344 | 51200 / 57344 | 65536 / 73728 | 1.28 |
| | q4_0 | 89088 / 95232 | 95232 / 106496 | 95232 / 106496 | 131072 / 147456 | 1.47 |
| qwen3.8-27B UD-IQ3_XXS | f16 | 25600 / 27648 | 24576 / 27648 | 24576 / 27648 | 49152 / 55296 | 1.92 |
| | q8_0 | 47104 / 49152 | 49152 / 55296 | 49152 / 55296 | 98304 / 110592 | 2.09 |
| | q4_0 | 88064 / 91136 | 82944 / 92160 | 82944 / 92160 | 184320 / 206848 | 2.09 |
| qwen3.6-35B-A3B IQ2_M | f16 | 90112 / 101376 | 98304 / 110592 | 98304 / 110592 | 163840 / 184320 | 1.82 |
| | q8_0 | 157696 / 168960 | 159744 / 168960 | 159744 / 168960 | 286720 / 322560 | 1.82 |
| | q4_0 | 236544 / 256000 | 239616 / 259072 | 239616 / 259072 | 537600 / 573440 | 2.27 |

**Vulkan goes 1.17 to 2.27 times as deep as ROCm**, least on the 1.5B and most on the 27B and the MoE.
**A quantised cache takes ROCm 1.75 to 1.92 times as deep with `q8_0` and 2.63 to 3.50 times with
`q4_0`**, against f16. The three ROCm arms end within a bisection step or two of each other, and none is
ahead on every model. The 1.5B with a `q4_0` cache generates past a million tokens on both backends.

**Without the copy ROCm goes about twice as deep as before**, 1.54 to 2.27 times
([`earlier-passes/bisect-results.txt`](earlier-passes/bisect-results.txt), `hip` lines):

| model | f16 | q8_0 | q4_0 |
|---|---|---|---|
| qwen3-8B Q8_0 | 16384 → 32768 | 32768 → 61440 | 61440 → 114688 |
| qwen3-14B Q4_K_M | 13312 → 26624 | 25600 → 51200 | 49152 → 89088 |
| qwen3.8-27B UD-IQ3_XXS | 11264 → 25600 | 22528 → 47104 | 40960 → 88064 |
| qwen3.6-35B-A3B IQ2_M | 45056 → 90112 | 81920 → 157696 | 153600 → 236544 |

On the other three arms the copy mattered much less. Their shallowest failures moved up by 8 percent at
most, except on the MoE with quantised caches and the 27B with `q4_0`, where the arms without graphs or
without packet capture went 12 to 50 percent deeper. Vulkan holds the copy in RAM, outside KFD's
accounting.

## Why ROCm stops where it does

KFD counts everything ROCm maps for the GPU against one limit: 13423 MiB on this board, 63/64 of RAM
less 1.5 GiB. The GTT buffers, which are the weights, the KV cache and the compute buffers, count
against it, and so does pinned host memory. llama.cpp gives the CPU part of its graph, the inputs
included, the GPU backend's host buffer type, which is pinned (`llama-context.cpp`, "for faster transfer
of the intermediate state"), and with flash attention the largest input is the KQ mask, 512 rows of the
context length in f16: 1 KiB per token of context. Once the context exists, KFD's count is TTM's count of
the GTT buffers, plus that input buffer, plus a constant per model and arm: 90 to 91 MiB for the 1.5B at
every depth from 8192 to 1118208, 78 to 108 MiB for the others ([`kfd-accounting.txt`](kfd-accounting.txt)).

Past the limit a ROCm run ends in one of four ways ([`ceilings.tsv`](ceilings.tsv), last column):

- **The context is not created** (24 of 45 cells): the KV cache and compute buffers do not fit. It
  fails as soon as the model has loaded, 1 to 286 s in.
- **The context fits with a few MiB to spare, and the first operation of the fill that needs a temporary
  buffer fails** (12 cells): rocBLAS's GEMM (`ggml_cuda_mul_mat_cublas`) or ggml's memory pool
  (`ggml_cuda_pool_leg::alloc`), `ROCm error` 56 to 270 s in, right after the load. KFD's count is then
  3 to 28 MiB short of the limit.
- **The context fits, but the first generated token finds KFD's budget spent** (7 cells, all `hip`). In
  five of them the pinned input buffer did not fit. ggml-cuda then falls back to ordinary host memory
  with a message only at debug level, `ggml_cuda_host_malloc: failed to allocate 132.34 MiB of pinned
  memory: out of memory` for the 8B with `q4_0` at 118784 ([`followup/`](followup/)); clr pins the mask
  on the fly for every upload, and KFD's count climbs 1 KiB per filled token until it reaches the limit.
  In the other two, the 8B and the 27B with `q8_0`, everything fitted with 2 to 3 MiB to spare. Either
  way, with HIP graphs and clr's packet capture both on, the first generated token cannot instantiate its
  graph, 11 minutes to 12 hours in. With clr's error log on, the 8B with `q8_0` at 65536 shows the step
  that fails: clr cannot create the stream its packet capture needs (`Cannot create new Stream object`,
  `Packet capture failed`), and `hipGraphInstantiate` returns `out of memory`.
- **Two cells segfault** instead, the MoE with `q4_0` on `hipng` and `hippc`, 72 and 126 s in. Under gdb
  the `hipng` one crashes in the first token the process computes, llama-bench's one-token warmup before
  the fill, with KFD's count at 13414 of 13423 MiB in the last sample before it
  ([`followup/`](followup/)). clr loads each code object when the first of its kernels is launched, and
  here that load fails (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`). clr 7.1.1's `ihipLaunchKernel`
  (`hipamd/src/hip_platform.cpp`) then takes the host-side function pointer for a kernel handle, and the
  launch faults reading through it, in `amd::Kernel::getDeviceKernel`. The allocation failure that ends
  the other cells with a `ROCm error` ends this one in a segfault.

Without graphs, or without packet capture, a run whose input buffer did not fit generates. The 1.5B with
an f16 cache at 425984 starts every arm with KFD's count 91 MiB above TTM's, where the input buffer
would need 507: `hip` aborts at its first token after a 98-minute fill, `hipng` and `hippc` generate at
15.6 t/s. With a `q4_0` cache at 1198080 the two arms start 96 MiB above TTM, where the input buffer
would need 1261, reach the limit about four hours into the fill and generate at 2.13 t/s after twelve
and a half hours; `hip` had failed one step below, at 1179648. That is what those two arms buy, and it
is little: one bisection step at most, 0.6 to 8 percent on the 1.5B.

## Vulkan

Vulkan does not go through KFD. All 53 Vulkan failures were the kernel's OOM killer, 2 to 50 s into the
run, while it allocated. Its deeper passes push the rest of the system out to swap: up to 13.7 GiB
written to swap during one run (`summary.tsv`, `swapped_out_pages`, 4 KiB pages).

Three of its passes, the MoE with `q4_0` at 501760 and 537600 and the 1.5B with `q4_0` at 1376256,
completed through gfx ring timeouts: at least 2040 on the MoE's two, and 10092 on the 1.5B's, from 18
hours into its 25-hour fill to its end, one every 2.3 s by then ([`ring-timeouts.txt`](ring-timeouts.txt)).
With `amdgpu.gpu_recovery=0`, which this board's setup requires, the ring kept advancing; with recovery
on, the driver would have tried a GPU reset at the first of them, which this chip does not survive. No
other run logged one.

## What a million tokens costs

The fill grows with the square of the depth: for the 1.5B, 304 to 326 s per (100k tokens)² on ROCm and
354 to 376 on Vulkan, from 262144 tokens up. ROCm's deepest passes, 1198080 tokens on the two arms
without graphs or packet capture, took 12.4 and 12.5 hours. Vulkan's, 1376256, took 24.8, a quarter to a
third more than its shallower passes predict; it is the run that logged the ring timeouts.

Decode at the ceiling, with the clock at 1500 MHz through every second of the decode
([`decode-windows.txt`](decode-windows.txt)):

| 1.5B, cache | ROCm | ROCm, no graphs or no packet capture | Vulkan |
|---|---|---|---|
| q8_0 | 2.90 t/s at 688128 | 2.88 at 692224 | 9.16 at 688128, 7.82 at 802816 |
| q4_0 | 2.32 at 1105920 | 2.13 at 1198080 | 3.44 at 1376256 |

At 688128 tokens with a `q8_0` cache Vulkan decodes 3.2 times as fast as ROCm. On the front page the two
are level from 4096 to 30720 tokens with an f16 cache. Where ROCm falls behind between there and here was
not investigated.

## These are memory limits, not usable context

The models were trained on 32768 tokens (the 1.5B), 40960 (the 8B and the 14B) and 262144 (the 27B and
the MoE) ([`trained-context.txt`](trained-context.txt)). Every ceiling of the 1.5B lies past its trained
length, by 12 to 42 times; so do the 8B's and the 14B's with a quantised cache, and the MoE's on Vulkan
with a quantised cache. A model that generates there shows that the memory holds the context, and nothing
about whether its output is any use. llama-bench fills the context with random tokens.

## Conditions

- Fedora 44, kernel 7.2.5 with the patched module, `amdgpu.gpu_recovery=0`, the thirteen-patch ROCm build
  and the front page's Vulkan build, 1500 MHz clock policy, every other GPU and memory user stopped.
- The oberon governor's GPU temperature limit was 100 °C from 30 September 19:33, instead of the packaged
  95 °C. Long fills still reach it, and the governor then holds 1000 MHz for 5 s, so the fill times are
  those of a board at its thermal limit. The ceilings are memory limits and do not depend on the clock.
- The clock was sampled every minute from 30 September 19:22 and every second from 2 October 23:25
  (`clock-1min.log.xz`, `clock-1s.log.xz`, [`scripts/clock_sampler.sh`](../../scripts/clock_sampler.sh)).
  Every decode window with one-second samples, which covers every pass from 3 October on, was at 1500
  MHz throughout but one: `hippc` with `q4_0` at 1118208, whose first three seconds caught one of the
  governor's steps at the end of the fill; its rate, 2.30 t/s, is `hipng`'s at the same depth. The
  per-minute samples cannot separate a decode from the fill around it, so no earlier decode rate is
  quoted here; `ceilings.tsv` lists them as measured.
- mmap on, llama-bench's default. The throughput campaigns pass `-mmp 0`, which on ROCm costs some
  context ([`logs/nommap-ceiling-2026-09-22/`](../nommap-ceiling-2026-09-22/)).

## Files

- `results.log`: every cell, run and result line of the campaign, in order.
- `summary.tsv`: every run, with its result, decode rate, seconds and pages written to swap.
- `ceilings.tsv`, `kfd-accounting.txt`: the tables above, from
  [`scripts/ceilings_analyse.py`](../../scripts/ceilings_analyse.py).
- `decode-windows.txt`: the clock through each pass's decode, from
  [`scripts/decode_window_clock.py`](../../scripts/decode_window_clock.py).
- `runs/`: llama-bench's output for every run; `runs-mem.tar.xz`: the five-second memory samples.
- `seed-ok.tsv`, `earlier-passes/`: the passes of the two earlier campaigns with the copy that seeded this
  one, their summaries, the earlier bisection's results, and the output of each earlier pass that is a
  cell's deepest pass here.
- `followup/`: three runs repeated on 10 October with more output
  ([`scripts/ceilings_followup.sh`](../../scripts/ceilings_followup.sh),
  [`scripts/ceilings_pinned_fallback.sh`](../../scripts/ceilings_pinned_fallback.sh)): the 8B with
  `q8_0` at 65536 with llama-bench's log and clr's error log on, the MoE with `q4_0` at 259072 on
  `hipng` under gdb, and the 8B with `q4_0` at 118784 with the log on, stopped once its fill had
  started; each with its memory samples.
- `snapshot-check/`: the 14B without the copy at two depths where it failed with it
  ([`scripts/nosnap_check.sh`](../../scripts/nosnap_check.sh)), and `with-copy_*`, the earlier pass's
  run at 14336 with its memory samples.
- `ring-timeouts.txt`, `trained-context.txt` ([`scripts/gguf_context.py`](../../scripts/gguf_context.py)).
- `clock-1min.log.xz`, `clock-1s.log.xz`: the SMU's gfxclk and GPU temperature, the sampler's
  `clock1hz.log` and `clock_raw.log`.

## How to reproduce

Build the patched llama-bench next to the ROCm and Vulkan builds with
[`scripts/build_llama_bench_nosnap.sh`](../../scripts/build_llama_bench_nosnap.sh), then run
[`scripts/ceilings_bisect.sh`](../../scripts/ceilings_bisect.sh) with `LLAMA_BENCH_NO_DEPTH_CACHE=1`,
`HIP` and `VK` pointing at it, and `MODELS`, `KVS` and `ARMS` for the cells to measure; it resumes from
its own `summary.tsv`. Without `SEED` it starts every cell from 8192. Allow a day for each 1.5B run with
a quantised cache.
