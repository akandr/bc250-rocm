# The 8B prefill ladder to 24576 tokens, 2026-10-12

## What was open

The prefill-against-depth table on the front page stopped at 8192 cached tokens, where the September
runs stopped ([`logs/depth-thirteen-2026-09-22/`](../depth-thirteen-2026-09-22/),
[`logs/vulkan-fa-staging-2026-09-17/`](../vulkan-fa-staging-2026-09-17/)), while the decode ladder ran to
30720. The 8B holds 32k of f16 KV cache on ROCm, so 16384 and 24576 were measurable and had not been.

## Method

[`scripts/prefill_deep.sh`](../../scripts/prefill_deep.sh): qwen3-8B Q8_0, `llama-bench -p 2048 -n 0 -fa on
-ngl 99 -r 3 -d DEPTH`, one invocation per point, three arms interleaved within each depth, three passes,
25 seconds of cooling between points, the shader clock (`pp_dpm_sclk`) sampled every two seconds during
each run and the edge temperature logged after it. The arms: ROCm `build-hip-pkf16` (the thirteen patches,
`-mmp 0`, `HSA_ENABLE_SDMA=0`), Vulkan `build-vk-f44` (both at 7ba604f) and the `vulkan-fa-staging` build of
master bfdc321 carrying [#28507](https://github.com/ggml-org/llama.cpp/pull/28507), whose `llama-bench` no
longer takes `-mmp` and loads the model the default way. 8192 was measured again as the control point.
ollama, `hw-watcher.timer`, cron and the board's other timers were stopped for the window; the oberon
governor was the packaged one at 1500 MHz, kernel 7.2.5. No fault in either phase.

[`scripts/prefill_deep2.sh`](../../scripts/prefill_deep2.sh), the second phase, adds the 0 and 4096 points
with the same arms, so that the whole ladder is one day's measurement, and repeats 24576 with `-r 1` and
three minutes of idle before each arm, to see whether the throttling the first phase logged there is
inherent to a 24k fill or an accumulation across points.

## Result

Medians of the three passes, tokens per second (`phase1/log`, `phase2/log`):

| existing context | ROCm, thirteen patches | Vulkan | Vulkan with #28507 | ROCm / Vulkan | ROCm / #28507 |
|---|---|---|---|---|---|
| 0 | 390.9 | 367.4 | 377.2 | 1.06 | 1.04 |
| 4096 | 302.2 | 217.9 | 294.1 | 1.39 | 1.03 |
| 8192 | 246.5 | 143.2 | 242.4 | 1.72 | 1.02 |
| 16384 | 171.8 | 84.1 | 176.5 | 2.04 | 0.97 |
| 24576 | 134.6 | 58.9 | 135.3 | 2.29 | 0.99 |

**ROCm's lead over stock Vulkan grows with context, 1.06 to 2.29 times, and against Vulkan carrying #28507
it is level at every depth, 0.97 to 1.04.**

**The September figure at 8192 was low.** The thirteen-patch build read 231.6 ± 21.6, 209.2 ± 4.5 and
214.6 ± 13.4 there on 22 September (`log-followup`, section D), and the front page carried the median,
214.6. The nine invocations of 12 October read 246.42 to 246.60, spreads under 0.1, at edge temperatures
of 65 to 76 °C and a clock that never left 1500 MHz once the model was loaded. The Vulkan arms reproduce
September within a few percent, 143.2 against 143.8 and 242.4 against 236.1, so the correction is to the
ROCm point alone: 1.72 times stock Vulkan at 8192, not 1.49, and level with #28507 rather than behind it.

**At 24576 every arm throttles.** Every run has six to nine clock samples at 1000 MHz, the seconds before
the model is loaded, and at 8192 and 16384 that is all of them. At 24576 the ROCm runs have 51 to 59 of
133 to 138 samples at 1000 MHz, the Vulkan runs 35 to 37 of 200 to 203, and the #28507 runs 26 to 34 of
111 to 119. The governor logged `GPU overheated, throttling` at 19:48, during the first 24576 point, and silences later
warnings, so the `GOVERNOR THROTTLED` lines the harness printed after that one are the same event seen
again, not new ones. The second phase repeats 24576 with `-r 1`, each arm started after
three minutes of idle, from 56 to 57 °C: ROCm 132.8, 137.2 and 137.7, Vulkan 58.5, 58.3 and 59.0, the
#28507 build 133.6, 134.2 and 134.8, with 18 to 23 of 87 to 89 samples at 1000 MHz on ROCm, 12 to 13 of
142 on Vulkan and 12 to 13 of 82 on the #28507 build, which is 10 to 15 samples beyond the load on ROCm
and a handful on the Vulkan arms. Medians 137.2, 58.5 and 134.2, ratios 2.35 and 1.02, within two percent
of the first phase: less throttling from a cold start, the same result, so a 24k fill throttles by itself
on this cooling rather than by accumulation, and the 24576 row is a floor for the 1500 MHz figure. Its
0 and 4096 points are the ones in the table above.

## What this does not say

One model and one prompt length. 30720 was not measured: 30720 + 2048 exceeds the 8B's 32k ceiling on ROCm
with f16 KV cache ([`logs/context-ceilings-2026-09-30/`](../context-ceilings-2026-09-30/)). The 24576
row is a ratio of two throttled arms, ROCm the more throttled, so it bounds the 1500 MHz figure from below.

## Files

- `phase1/log`: the first phase, every point with its clock histogram (`clk=` lists MHz:samples) and the
  edge temperature after it; `phase1/raw_*`, `err_*`: each `llama-bench` invocation's output;
  `phase1/clk_*`: the clock samples, one per two seconds.
- `phase2/`: the same for the second phase (0, 4096, and 24576 with `-r 1`).
