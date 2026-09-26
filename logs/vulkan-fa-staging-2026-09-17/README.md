# Two upstream Vulkan patches tested on this board, 2026-09-17

Both are open, unmerged llama.cpp pull requests whose authors report measurements taken on a BC-250, which
makes them worth checking here instead of reading:

- [#28507](https://github.com/ggml-org/llama.cpp/pull/28507), "vulkan: enable FA shmem staging on AMD RDNA
  (scalar path)". Flash attention stages K and V through shared memory; the gate was NVIDIA-only when it was
  introduced, and this widens it to AMD RDNA. Seven lines.
- [#27332](https://github.com/ggml-org/llama.cpp/pull/27332), "vulkan: use density gate for MUL_MAT_VEC_ID
  path". Chooses the vector, not the tiled kernel for mixture-of-experts matmuls by routed density
  when the device has no coopmat2, which gfx1013 does not.

Both applied cleanly to master `bfdc321`, the commit already measured in
[`../llamacpp-master-recheck-2026-09-14/`](../llamacpp-master-recheck-2026-09-14/), and both were built into
one binary and compared against the same commit unpatched. Fedora 44, kernel 7.1.8, clock pinned at 1500 MHz,
one test per `llama-bench` invocation, builds alternated, three rounds of `-r 2`; `log` lists every sample.
Medians of six:

| test | master | with both patches | change |
|---|---|---|---|
| qwen3-8B Q8_0, pp2048 at depth 0 | 366.82 | 377.20 | +2.8 % |
| qwen3-8B Q8_0, pp2048 at depth 4096 | 217.01 | 293.62 | **+35.3 %** |
| qwen3-8B Q8_0, pp2048 at depth 8192 | 143.51 | 236.13 | **+64.5 %** |
| qwen3.6-35B MoE, tg64 | 86.06 | 86.01 | -0.1 % |
| qwen3.6-35B MoE, pp512 | 472.57 | 475.81 | +0.7 % |

The qwen3-8B perplexity gate is unchanged: 7.3948 +/- 0.23389 on both builds (`gates.txt`).

## The ROCm backend for the same case

Measured immediately afterwards on the same boot, same model and flags (`hip_pp2048_d*.jsonl`): ROCm
prefills at 197.5, 98.3 and 65.5 t/s at depths 0, 4096 and 8192, against patched Vulkan's 377.2, 293.6 and
236.1. The gap widens from 1.9 times to 3.6 times with depth, so this patch turns a large Vulkan advantage
into a larger one rather than changing which backend to use for long contexts.

## Reading

The flash-attention change is a large win at depth on this board, and it grows with depth exactly as its
author reported on their own BC-250: they measured +12 percent at depth 0, +34 at 4096 and +50 at 8192 on a
different model, against +2.8, +35 and +64 here. Prefill at depth is where the ROCm and Vulkan backends are
furthest apart, so this widens that gap instead of closing it.

The mixture-of-experts gate shows nothing here, which is not evidence against it: it targets batched decode
with 9 to 64 routed tokens, and `llama-bench` decodes one sequence at a time. Testing it properly needs a
batched harness, which was not run.

Neither patch is merged upstream. #28507 had no review at all at the time of this test.
