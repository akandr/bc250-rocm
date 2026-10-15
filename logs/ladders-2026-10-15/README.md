# The 8B decode ladder and the 1.5B prefill ladder, 2026-10-15

## What was open

The depth tables on the front page had decode against depth for the 1.5B only and prefill against depth
for the 8B only, so the two metrics could not be compared on one model. This run measures the two missing
ladders with the same arms and method as the existing ones.

## Method

[`scripts/ladders_8b_15b.sh`](../../scripts/ladders_8b_15b.sh), one `llama-bench` invocation per point,
arms interleaved within each depth, three passes, 25 seconds of cooling between points, the shader clock
(`pp_dpm_sclk`) sampled every two seconds during each run and the edge temperature logged after it. ollama,
`hw-watcher.timer`, cron and the board's other timers were stopped for the window; oberon governor at
1500 MHz, kernel 7.2.5. No fault.

- **A, decode:** qwen3-8B Q8_0, `-p 0 -n 64 -r 3 -d DEPTH` at 0, 4096, 8192, 16384 and 24576. Arms: ROCm
  `build-hip-pkf16` as shipped (`GGML_CUDA_GRAPH_OPT=0`), the same patches with `GGML_CUDA_GRAPH_OPT=1` and
  its fix (`llama-graphopt/build-hip-pkf16`, `GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1`), and Vulkan `build-vk-f44`.
  All three run the `llama-bench` of [`scripts/llama-bench-no-depth-cache.patch`](../../scripts/llama-bench-no-depth-cache.patch)
  with `LLAMA_BENCH_NO_DEPTH_CACHE=1`, each linked to its build's libraries: with stock `llama-bench`, which
  pins a copy of the KV cache after the fill, the 8B fails at 16384 on ROCm with `ggml-cuda.cu:111: ROCm
  error`, out of memory (`../ladders-2026-10-15-try1` was that attempt and is not kept). Without the copy
  every repetition refills the context, which is why these runs take twice as long as the 1.5B's.
- **B, prefill:** qwen2.5-1.5B Q4_K_M, `-p 2048 -n 0 -r 3 -d DEPTH` at 0 to 30720, stock `llama-bench` like
  the 8B prefill ladder it pairs with ([`logs/prefill-deep-2026-10-12/`](../prefill-deep-2026-10-12/)).
  Arms: ROCm, Vulkan, and Vulkan with [#28507](https://github.com/ggml-org/llama.cpp/pull/28507)
  (`llama-vk28507/build-vk-patched`, master bfdc321, no `-mmp`).
- **Phase 2** ([`scripts/ladders_phase2.sh`](../../scripts/ladders_phase2.sh)): the 8B's 16384 and 24576
  points again with `-r 1` and three minutes of idle before each arm, three passes, because the first phase
  throttled there.

## Result

Medians of three passes, tokens per second.

**A. Decode, qwen3-8B:**

| depth | 0 | 4096 | 8192 | 16384 | 24576 |
|---|---|---|---|---|---|
| ROCm, as shipped | 38.43 | 36.13 | 34.03 | 29.62 | 27.53 |
| ROCm, option with its fix | 39.85 | 37.08 | 34.72 | 29.27 | 27.43 |
| Vulkan | 39.06 | 35.73 | 33.22 | 29.25 | 26.19 |
| as shipped / Vulkan | 0.984 | 1.011 | 1.024 | 1.013 | 1.051 |
| with the fix / Vulkan | 1.020 | 1.038 | 1.045 | 1.001 | 1.047 |

**ROCm decodes the 8B ahead of Vulkan from 4096 tokens on**, where on the 1.5B it is level: as shipped by
1 to 5 percent, with the option and its fix by 4 to 5 percent, and 2 percent behind at an empty context
as shipped. At 16384 and 24576 the ROCm arms throttled: three refills of a 16k or 24k context are three
prefills back to back, and the clock samples sit at 1000 MHz for 30 to 40 percent of the run on both ROCm
arms (31 to 40 of 114 at 16384, 59 to 79 of about 205 at 24576) against the loading seconds only on Vulkan
(7 to 9 of 169 and 202). The spreads say the same, up to 2.3 on the ROCm arms there and under 0.1 on Vulkan.
The option's row falling to parity at 16384 is this and not the option.

**Phase 2**, the same two depths with `-r 1` after three idle minutes per arm, three passes, every run
at 1500 MHz once loaded (six to eleven samples at 1000 MHz, the load): ROCm 30.71, 30.75, 30.71 at 16384 and
28.00, 27.99, 28.00 at 24576; with the fix 31.44, 31.44, 31.41 and 28.59, 28.35, 28.54; Vulkan 29.32, 29.29,
29.43 and 26.14, 26.11, 26.19. Medians 30.71 / 31.44 / 29.32 and 28.00 / 28.54 / 26.14, so **1.05 and 1.07
times Vulkan as shipped, 1.07 and 1.09 with the fix**: the lead grows with depth, and the dip to parity in the
three-pass run was the clock.

**B. Prefill, qwen2.5-1.5B:**

| existing context | ROCm | Vulkan | Vulkan with #28507 | ROCm / Vulkan | ROCm / #28507 |
|---|---|---|---|---|---|
| 0 | 1676.5 | 1718.7 | 1721.7 | 0.975 | 0.974 |
| 4096 | 1205.2 | 1139.5 | 1258.5 | 1.058 | 0.958 |
| 8192 | 941.4 | 888.6 | 978.3 | 1.059 | 0.962 |
| 16384 | 654.4 | 585.6 | 680.7 | 1.117 | 0.961 |
| 24576 | 501.5 | 452.1 | 521.4 | 1.109 | 0.962 |
| 30720 | 426.7 | 386.5 | 443.8 | 1.104 | 0.962 |

**On the 1.5B ROCm leads stock Vulkan by 6 to 12 percent behind a filled context**, against 1.4 to 2.3
times on the 8B, and trails Vulkan with #28507 by 4 percent at every depth, where on the 8B it is level.
No clock drop at any depth: a 30720-token fill takes the 1.5B under a minute. The 8B's lead is therefore
mostly the model's attention share, and the flash-attention patch that produces it matters less the
smaller the model.

## What this does not say

One model per metric again, two models in all. The 8B decode row at 16384 and 24576 is a throttled
measurement of ROCm against an unthrottled Vulkan, so its ratios there are floors (see phase 2). Nothing
here is at an empty context with the usual pp512/tg64 arguments, which the headline table has.

## Files

- `main/log`: every point of A and B with its clock histogram (`clk=` lists MHz:samples) and the edge
  temperature after it; `main/raw_*`, `err_*`: each `llama-bench` invocation's output; `main/clk_*`: the
  clock samples, one per two seconds.
- `phase2/`: the same for phase 2. `phase2/cpu_train.txt`: the CPU reference for the front page's PyTorch
  figure ([`scripts/train_cpu_ref.py`](../../scripts/train_cpu_ref.py)), the same training step on the board's six
  cores (12 threads) in the same PyTorch build, which has neither MKL nor oneDNN: 0.13, 0.31 and 0.46 ktok/s at
  batch 1, 4 and 16, against 33.8 on the GPU at batch 16.
