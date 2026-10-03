# The decode ladder with the multi-stream option, 2026-09-24

> **Correction, 2 October 2026.** The rates here stand, but the option, as shipped, computes wrong
> tokens on the models where it launches streams; the 1.5B's replies happened to match the default in
> ordinary runs. [`logs/graphopt-correctness-2026-10-02/`](../graphopt-correctness-2026-10-02/) has the
> evidence and a fix that keeps the speed.

The front page's decode-at-depth table was measured before `GGML_CUDA_GRAPH_OPT=1` existed in this
repository's vocabulary ([`logs/graph-opt-2026-09-24/`](../graph-opt-2026-09-24/)), and decode at depth
is where ROCm had already drawn level, so the published row understated the build. This is the same
ladder with a third arm.

[`scripts/depth_graphopt.sh`](../../scripts/depth_graphopt.sh) is section H of
[`scripts/depth_thirteen.sh`](../../scripts/depth_thirteen.sh) unchanged except for the arm: one
`llama-bench` invocation per depth and never a `-d` list, which depresses readings by up to 22 percent;
arms interleaved within each depth; `-mmp 0 -ngl 99 -fa on`, `HSA_ENABLE_SDMA=0`; a 25-second cool-down
between points, because deep decode run back to back reaches 93 C in six minutes and the governor then
drops the clock by 28 percent in a way that reads as a clean measurement of a slower machine. Two
passes, qwen2.5-1.5B, on `build-hip-pkf16` against `build-vk-f44`.

**Neither pass throttled and neither faulted**, and the as-shipped arm reproduces the published row:
195.0, 176.8, 163.1, 141.1, 123.0, 112.4 against the front page's 197.2, 176.5, 163.0, 141.3, 123.2,
112.6.

## The ladder

| depth | ROCm as shipped | ROCm, `GRAPH_OPT` | Vulkan | as shipped / Vulkan | with option / Vulkan | option worth |
|---|---|---|---|---|---|---|
| 0 | 195.0 | **209.8** | 212.3 | 0.92 | 0.99 | **1.076** |
| 4096 | 176.8 | **187.1** | 179.2 | 0.99 | **1.04** | 1.058 |
| 8192 | 163.1 | **172.1** | 160.6 | 1.02 | **1.07** | 1.055 |
| 16384 | 141.1 | **147.5** | 138.9 | 1.02 | **1.06** | 1.045 |
| 24576 | 123.0 | **127.6** | 120.6 | 1.02 | **1.06** | 1.037 |
| 30720 | 112.4 | **116.2** | 110.9 | 1.01 | **1.05** | 1.033 |

**With the option on, ROCm decodes faster than Vulkan at every depth from 4096 to 30720**, by 4 to 7
percent, and is within one percent at an empty context. Without it the same build is level, 0.99 to
1.02, from 4096 onwards and 8 percent behind at depth 0.

The option is worth most at depth 0, 1.076, and least at 30720, 1.033, which is the shape to expect: as
the context fills, more of the token goes into attention over a longer cache and less into the layer
work whose independent branches the optimiser can overlap.

Twelve paired comparisons, two passes at six depths: **the option wins 12 of 12**, and the option arm
beats Vulkan in **10 of 12**, the two exceptions being both passes at depth 0.

## What this does not say

Two passes, not the three the original ladder used, so each figure is the median of two `llama-bench`
means, not three. The effect is 3 to 8 percent and every one of the twelve pairs agrees in sign,
which is why it is reported at this size; a one or two percent effect would not be.

One model. The ladder is the 1.5B because that is the model the front page's depth table tracks. The
option launches no streams at all on the MoE and the 27B
([`logs/graph-opt-2026-09-24/`](../graph-opt-2026-09-24/)), so their depth behaviour will not have
moved, and that was not re-measured here.

The prefill-at-depth table was not redone. Prefill does not move with this option at any batch size
measured ([`logs/campaign-graphopt-2026-09-24/`](../campaign-graphopt-2026-09-24/)), so it should not
move with depth either, but that is an inference instead of a measurement.

## Files

`log` is both passes with the temperature at every point.
