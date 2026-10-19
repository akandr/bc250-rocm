# The 8B one rung deeper, and decode against depth on the other four models, 2026-10-17 and 18

## What was open

Three gaps in the depth tables. The 8B prefill ladder stopped at 24576 cached tokens although its f16 cache
allows one more pp2048 rung on ROCm (28672 cached, 30720 in context, under the 32768 limit), and so did its
decode ladder. And decode against depth existed for the 1.5B and the 8B only, so nothing said how the two 14B
models, the MoE and the 27B behave once the context is full.

## Method

Three scripts, all built on the ladder harness of
[`logs/ladders-2026-10-15/`](../ladders-2026-10-15/): one `llama-bench` invocation per point, `-r 1`, arms
interleaved within each point, three passes, the shader clock (`pp_dpm_sclk`) sampled every two seconds
during each run and the edge temperature logged after it. ollama, `hw-watcher.timer`, cron,
`dnf-makecache.timer`, `sysstat-collect.timer` and `plocate-updatedb.timer` were stopped for the window;
oberon governor at 1500 MHz, kernel 7.2.5, Fedora 44, `HSA_ENABLE_SDMA=0`. No fault in any run.

- **A** ([`scripts/ladders_rest.sh`](../../scripts/ladders_rest.sh), directory `a/`): qwen3-8B Q8_0, `-p 2048
  -n 0 -d 28672 -r 1` after three idle minutes, three arms as in
  [`logs/prefill-deep-2026-10-12/`](../prefill-deep-2026-10-12/): ROCm `build-hip-pkf16` (thirteen
  patches), Vulkan `build-vk-f44`, and Vulkan with [#28507](https://github.com/ggml-org/llama.cpp/pull/28507)
  (`llama-vk28507/build-vk-patched`, master bfdc321, no `-mmp`). Stock `llama-bench`, like the rest of that
  ladder; it fitted, so the no-depth-cache fallback the script carries was not used.
- **B** ([`scripts/ladders_rest_b.sh`](../../scripts/ladders_rest_b.sh), directory `b/`): decode, `-p 0 -n 64
  -d DEPTH -r 1` after 90 idle seconds, on deepseek-r1-14B Q4_K_M, qwen3-14B Q4_K_M, qwen3.6-35B-A3B MoE
  IQ2_M and qwen3.8-27B UD-IQ3_XXS, at 4096, 8192 and 16384, ROCm as shipped (`GGML_CUDA_GRAPH_OPT=0`)
  against Vulkan, both with the `llama-bench` of
  [`scripts/llama-bench-no-depth-cache.patch`](../../scripts/llama-bench-no-depth-cache.patch) linked to
  each build's libraries, because the stock one's pinned copy of the KV cache would not fit the 27B's
  25600-token ROCm limit. Then 24576 on the models whose ROCm limit allows it. The script stops at the
  first failed run (see the incident below).
- **C1** ([`scripts/ladders_rest_c.sh`](../../scripts/ladders_rest_c.sh), directory `c1/`): the 24576 points
  of qwen3-14B and the MoE for three more passes with 120 idle seconds, and deepseek-14B at 24576 on
  Vulkan alone, three passes, where ROCm does not fit it.
- **C2** (same script, directory `c2/`): the 27B at 8192 and 16384 once more with `-r 1` after five idle
  minutes, three passes. In B its long fills (80 and 160 seconds of prefill at about 100 tokens/s) heated
  the board past the governor's threshold within the 90-second rests: 24 to 31 of about 125 clock samples at
  1000 MHz at 16384 on both arms, against the 9 to 15 that model loading alone leaves, and the governor
  logged its overheat line during the first 27B run at 8192. The B figures for the 27B at 16384 are kept but
  are not the measurement.

- **D** ([`scripts/ladders_rest_d.sh`](../../scripts/ladders_rest_d.sh) and
  [`scripts/ladders_rest_d2.sh`](../../scripts/ladders_rest_d2.sh), directories `d/` and `d2/`, 2026-10-18): the 8B's
  decode behind 28672, 30720 and 32704 cached tokens, the three arms of
  [`logs/ladders-2026-10-15/`](../ladders-2026-10-15/) (ROCm as shipped, ROCm with `GGML_CUDA_GRAPH_OPT=1` and
  its fix, Vulkan), the no-depth-cache `llama-bench`, `-r 1` after three idle minutes, three passes. The first
  ROCm run at 30720 filled the context and aborted at an allocation (`ggml-cuda.cu:111`, SIGABRT, no kernel
  fault, nothing leaked), so the script stopped as designed and `d2/` finished the job: passes 2 and 3 at 28672
  on all arms, then 30720 on Vulkan alone. 32704 was not tried.

**The incident, and a rule.** The first B attempt (its runs are in `a/` as `B_p1_ds14_*`) tried deepseek-14B
at 24576 on ROCm. It does not fit: `ggml-cuda.cu:111: ROCm error` at load, and the process aborted with a
core dump. Right after it `MemAvailable` read 1.2 GiB instead of the 14.1 GiB before, with no process
holding anything and KFD's own accounting at zero (`/sys/kernel/debug/kfd/mem_limit`), which would have
starved every later run. The board was rebooted and B restarted with the risky depths last and a stop on
the first failure. The 27B's failure at 24576 ("failed to create context", a clean error return) and the
8B's abort behind 30720 in part D left the memory alone. [`logs/leak-test-2026-10-18/`](../leak-test-2026-10-18/)
then tried every ending of a `llama-bench` run, normal exit, SIGKILL and a context that cannot be created,
on both backends, and none of them leaks; what does, reproducibly, is ollama's load and unload of a model,
and ollama had run overnight before both mornings on which the board started short of memory. The drop
after the deepseek-14B abort stays a single observation. The rule for a campaign is the same either way:
log `MemAvailable` before every run, put the runs that might not fit last, stop at the first failure, and
reboot before measuring on a board that has run ollama since boot.

## Result

### A. qwen3-8B prefill, pp2048 behind 28672 cached tokens

| pass | ROCm | Vulkan | Vulkan with #28507 |
|---|---|---|---|
| 1 | 120.76 | 51.14 | 122.02 |
| 2 | 123.46 | 51.28 | 124.34 |
| 3 | 122.68 | 51.03 | 122.97 |
| **median** | **122.7** | **51.1** | **123.0** |

**2.40 times stock Vulkan, 1.00 of Vulkan with #28507**, continuing the ladder (1.39, 1.72, 2.04, 2.29 at
4096 to 24576). As at 24576, the fill throttles: 21 to 27 of about 105 clock samples at 1000 MHz on ROCm
against the loading seconds only on the Vulkan arms (15 to 16 of about 100 and 185), and the governor logged
an overheat during the second ROCm run. The ROCm figure is a floor.

### D. qwen3-8B decode, tg64 behind 28672 cached tokens

| pass | ROCm, as shipped | ROCm, option with its fix | Vulkan |
|---|---|---|---|
| 1 | 26.93 | 27.34 | 24.88 |
| 2 | 26.79 | 27.40 | 25.06 |
| 3 | 26.83 | 26.73 | 25.02 |
| **median** | **26.83** | **27.34** | **25.02** |

**1.07 times Vulkan as shipped and 1.09 with the option and its fix**, the same two ratios as at 24576 in that
ladder's phase 2. The as-shipped arm ran at 1500 MHz once loaded (9 to 15 samples at 1000 MHz, the load) and
Vulkan too (9 of 156); the option's arm throttled during its fill in all three passes (18 to 22 of about 85,
and the governor logged its overheat line in the first), so its figure is a floor. Behind 30720 ROCm does not
hold the 8B at all with the model in pinned host memory: the fill completes and the next allocation fails,
`ggml-cuda.cu:111`, abort. Vulkan decodes there at 24.47, 24.39, 24.54 (median 24.5).

### B and C. Decode, tg64 behind cached tokens, ROCm as shipped / Vulkan

Medians, tokens per second; n is the number of passes behind each median (four where the aborted first
attempt had measured the point too, and its values agree with the rerun to within 0.2).

| model | depth | ROCm | Vulkan | ROCm / Vulkan | n |
|---|---|---|---|---|---|
| deepseek-r1-14B Q4_K_M | 0 (headline table) | 32.6 | 35.1 | 0.93 | |
| | 4096 | 26.5 | 31.0 | 0.85 | 4 |
| | 8192 | 22.4 | 28.1 | 0.80 | 4 |
| | 16384 | 15.0 | 23.7 | 0.63 | 4 |
| | 24576 | does not fit | 20.4 | | 3 |
| qwen3-14B Q4_K_M | 0 | 32.7 | 34.7 | 0.94 | |
| | 4096 | 27.3 | 31.4 | 0.87 | 3 |
| | 8192 | 23.7 | 28.7 | 0.82 | 3 |
| | 16384 | 16.5 | 25.0 | 0.66 | 3 |
| | 24576 | 12.2 | 21.8 | 0.56 | 4 |
| qwen3.6-35B-A3B MoE IQ2_M | 0 | 71.2 | 86.9 | 0.82 | |
| | 4096 | 69.0 | 83.9 | 0.82 | 3 |
| | 8192 | 67.0 | 81.5 | 0.82 | 3 |
| | 16384 | 63.5 | 78.1 | 0.81 | 3 |
| | 24576 | 60.2 | 73.9 | 0.81 | 4 |
| qwen3.8-27B UD-IQ3_XXS | 0 | 15.2 | 17.6 | 0.86 | |
| | 4096 | 14.9 | 17.3 | 0.87 | 3 |
| | 8192 | 14.7 | 16.4 | 0.90 | 6 |
| | 16384 | 10.8 | 16.5 | 0.66 | 6 |
| | 24576 | does not fit | | | |

The 27B rows pool B and C2, six runs each. Its ROCm side reads 14.66 to 14.70 at 8192 and 10.81 to 10.93 at
16384 in all of them, rest or no rest. Its Vulkan side at 8192 splits into two levels, 15.17, 15.21, 15.83 and
16.92, 16.94, 16.96, with the same 10 to 11 loading samples at 1000 MHz in both groups, so it is not the
clock; the median, 16.4, is between them and the ratio is 0.87 to 0.97 depending on which level Vulkan
lands on. At 16384 both arms throttle from the fill itself, 16 to 21 samples at 1000 MHz on ROCm and 16 to
31 on Vulkan after a five-minute rest as after a 90-second one, and the figures are stable all the same
(Vulkan 16.44 to 16.48 across six runs).

**Two shapes.** The MoE keeps the same distance at every depth, 0.81 to 0.82: its decode is bound by the
quantised matrix-vector work on the active experts, which does not grow with the context, and the ROCm
side of that is the matvec gap the front page describes. The dense models fall away as the cache grows:
both 14Bs go from 0.93 at an empty context to 0.80 at 8192 and about two thirds at 16384, qwen3-14B to
0.56 at 24576, and the 27B drops the same way between 8192 and 16384, 0.90 to 0.66. The drop between 8192 and 16384 is the large one, a third of the ROCm speed against a sixth
of Vulkan's, and it reproduces to the tenth across four passes (deepseek-14B at 16384: 15.03, 14.99, 14.97,
15.02). The 8B, the same head size and GQA ratio 4 instead of 5, does not have it (1.05 at 16384, cold).
Where the time goes on the 14Bs at depth is not measured here; it is the next thing to profile with
rocprofv3, attention over a long cache being the only part of the token that grows.

Every value in this directory is a median of runs that reproduce to a few tenths; the per-run clock
histograms are in the `clk_*` files and the raw `llama-bench` tables in `raw_*`.
