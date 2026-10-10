# The thirteen published patches rebuild the measured build, 2026-10-10

## What was open

Every figure on the front page comes from `build-hip-pkf16`, built on 21 September from the board's
working tree, not from the files in [`patches/llamacpp/`](../../patches/llamacpp/). Whether those
thirteen patches rebuild it had not been checked.

## Result

**One patch did not apply as published. With it regenerated, the rebuild computes what the measured
build computes, at the same speed.**

- Patch 0008 carried one line of the working tree's debug output in the context of its second hunk,
  `if (getenv("BC250_PATH")) fprintf(...)`, which upstream does not have, so `git am` stopped there. The
  regenerated 0008 is the change the old one makes when applied with fuzz, against upstream's context.
  With it the thirteen apply to 7ba604f with `git am` as they stand (`am.log`, `commits.txt`).
- Configured as step 7 of the front page says, the rebuild has the same GGML options as
  `build-hip-pkf16` (`ggml-options-diff.txt` is empty).
- Its source differs from the working tree in two files, only by additions that do nothing unless an
  environment variable is set: statistics in `common/debug.cpp` behind `GGML_DEBUG_STATS`, and checks,
  traces and alternative GEMM paths in `ggml-cuda.cu` behind `GGML_CUDA_NO_POOL`,
  `GGML_CUDA_POOL_NOREUSE` and `BC250_*` (`tree-diff.patch`: 261 lines added, none removed).
- 128 greedy tokens from every benchmark model are byte-identical between the two builds, and between
  two runs of the measured build on either side of the rebuild's:

| model | `build-hip-pkf16` | rebuild | `build-hip-pkf16` again |
|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | `7033302a` | `7033302a` | `7033302a` |
| qwen3-8B Q8_0 | `90b056d9` | `90b056d9` | `90b056d9` |
| deepseek-r1-14B Q4_K_M | `96357da6` | `96357da6` | `96357da6` |
| qwen3-14B Q4_K_M | `679a50d9` | `679a50d9` | `679a50d9` |
| qwen3.6-35B-A3B MoE IQ2_M | `b5ca281a` | `b5ca281a` | `b5ca281a` |
| qwen3.8-27B UD-IQ3_XXS | `817f1ab2` | `817f1ab2` | `817f1ab2` |

- Prefill and decode agree to within half a percent, tokens per second, median of nine samples per
  build:

| model | pp512 measured | pp512 rebuild | ratio | tg64 measured | tg64 rebuild | ratio |
|---|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1798.22 | 1798.93 | 1.000 | 197.83 | 197.56 | 0.999 |
| qwen3-8B Q8_0 | 406.39 | 407.54 | 1.003 | 38.48 | 38.48 | 1.000 |
| deepseek-r1-14B Q4_K_M | 192.19 | 192.24 | 1.000 | 32.37 | 32.33 | 0.999 |
| qwen3-14B Q4_K_M | 193.33 | 193.45 | 1.001 | 32.37 | 32.38 | 1.000 |
| qwen3.6-35B-A3B MoE IQ2_M | 587.20 | 587.20 | 1.000 | 70.01 | 70.34 | 1.005 |
| qwen3.8-27B UD-IQ3_XXS | 102.04 | 102.01 | 1.000 | 15.03 | 15.02 | 0.999 |

## Method

[`scripts/rebuild_check.sh`](../../scripts/rebuild_check.sh) clones 7ba604f from the board's own
llama.cpp tree, applies the patches with `git am`, compares the two source trees, and configures and
builds `llama-cli` and `llama-bench` as step 7 says. The replies are `llama-cli -ngl 99 -fa on -st -c 4096
--temp 0 -s 1 -n 128` with the prompt of [`logs/graphopt-correctness-2026-10-02/`](../graphopt-correctness-2026-10-02/).
The speed is `llama-bench -mmp 0 -ngl 99 -fa on -r 3`, prefill (`-p 512 -n 0`) and decode (`-p 0 -n 64`)
in separate invocations, three rounds with the order of the builds swapped every round. Every other GPU
user was stopped, and the oberon governor was the packaged one. In the clock samples the GPU ran at
1500 MHz through the tests except for the first second of some invocations, at 1000 MHz while the
governor ramped up; the arms alternate, so both builds share it.

## Files

- `log`: the run.
- `am.log`, `commits.txt`: `git am`'s output and the thirteen commits it made.
- `ggml-options-diff.txt`: the GGML options of the two CMake caches compared; empty.
- `tree-diff.patch`, `tree-diff-summary.txt`: the board's working tree against the rebuilt tree.
- `replies/`: every reply.
- `bench.jsonl`: every `llama-bench` result with its samples, `arm` naming the build.
