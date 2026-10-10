# The decode ladder with the multi-stream option's fix, 2026-10-10

## What was open

[`logs/depth-graphopt-2026-09-24/`](../depth-graphopt-2026-09-24/) measured `GGML_CUDA_GRAPH_OPT=1`
at depth as shipped, and as shipped the option computes wrong tokens
([`logs/graphopt-correctness-2026-10-02/`](../graphopt-correctness-2026-10-02/)). Its fix,
`alloc-deps.diff` behind `GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1`, had been timed at an empty context only.

## Method

[`scripts/depth_graphopt_fix.sh`](../../scripts/depth_graphopt_fix.sh) is
[`scripts/depth_graphopt.sh`](../../scripts/depth_graphopt.sh) with two more arms and a third pass:
one `llama-bench -p 0 -n 64 -d DEPTH -r 3` invocation per point, arms interleaved within each depth,
a 25-second cool-down between points, `-mmp 0 -ngl 99 -fa on`, `HSA_ENABLE_SDMA=0`, qwen2.5-1.5B Q4_K_M.
The five arms are the campaign build (`build-hip-pkf16`) as shipped and with the option; the build of
[`scripts/graphopt_mechanism.sh`](../../scripts/graphopt_mechanism.sh), which carries the fix, with the
option alone and with the option and the fix; and Vulkan (`build-vk-f44`). Every other GPU user was
stopped and the oberon governor was the packaged one. No pass throttled or faulted. Each figure below is
the median of the three passes' `llama-bench` means.

## Result

| depth | ROCm as shipped | ROCm, option with the fix | Vulkan | as shipped / Vulkan | with the fix / Vulkan |
|---|---|---|---|---|---|
| 0 | 196.6 | 210.9 | 212.0 | 0.93 | 1.00 |
| 4096 | 176.3 | 187.1 | 179.1 | 0.98 | **1.04** |
| 8192 | 162.9 | 171.9 | 161.6 | 1.01 | **1.06** |
| 16384 | 141.1 | 147.5 | 138.1 | 1.02 | **1.07** |
| 24576 | 122.9 | 127.8 | 121.0 | 1.02 | **1.06** |
| 30720 | 112.3 | 116.3 | 111.3 | 1.01 | **1.04** |

**The fix costs nothing at any depth.** In the same build, the option with the fix reads 0.9999 to
1.0031 of the option without it, and is ahead in 11 of 18 paired comparisons, which is what chance
gives.

**With the fixed option ROCm decodes faster than Vulkan at every depth from 4096 to 30720**, by 4.4 to
6.8 percent, in 15 of 15 paired comparisons. At an empty context the two are level, 0.995. The option is
worth 1.073 at depth 0 and 1.035 at 30720. Without it ROCm is level with Vulkan from 4096 on, 0.98 to
1.02, and 7 percent behind at depth 0.

The control: the as-shipped, unfixed-option and Vulkan rows reproduce those of 24 September to within
0.9 percent at every depth, the largest difference being depth 0, 196.6 against 195.0.

## What this does not say

One model, the one the front page's depth table tracks. The option launches no streams on the MoE and
the 27B ([`logs/moe-stream-aliasing-2026-09-25/`](../moe-stream-aliasing-2026-09-25/)), so there is
nothing for the fix to change there. Prefill was not measured; the option does not move it at an empty
context ([`logs/campaign-graphopt-2026-09-24/`](../campaign-graphopt-2026-09-24/)).

## Files

- `log`: every point, in order, with the edge temperature after it.
