# What prefill actually runs, and a null result about switching it, 2026-09-19

The seven-patch build prefills at half Vulkan's rate on the 1.5B (925 against 1850) and the kernel trace
of [`logs/kerntrace-2026-09-19/`](../kerntrace-2026-09-19/) said the MMQ GEMM was most of it. This is the
confirmation, and a null result worth writing down so nobody repeats it
([`scripts/round4_prefill.sh`](../../scripts/round4_prefill.sh)).

## MMQ is 92 percent of prefill

`llama-bench -p 512 -n 0 -r 1` on the 1.5B under the tracer (`trace-1.5b-pp-X.txt`; every kernel is
listed twice, once demangled and once not, so the real total is half the reported one):

| kernel | dispatches | device time | share of prefill |
|---|---|---|---|
| `mul_mat_q<q4_K, 64>` | 166 | 429 ms | 76 % |
| `mul_mat_q<q6_K, 64>` | 27 | 92 ms | 16 % |
| `flash_attn_tile<128,128,32,2>` | 28 | 15.5 ms | 3 % |
| everything else | ~560 | ~27 ms | 5 % |

Prefill on this board is one kernel, and that kernel is llama.cpp's own quantised GEMM.

## The null result: `GGML_CUDA_FORCE_CUBLAS` and `GGML_CUDA_FORCE_MMQ` are not environment variables

Set in the environment, both change nothing: pp128, pp512 and pp2048 are identical to the default on the
1.5B, the 8B and the MoE across two passes, and the traces are the same kernels with the same dispatch
counts (`log`). They are compile-time options, `#ifdef GGML_CUDA_FORCE_CUBLAS` in `mmq.cu`'s
`ggml_cuda_should_use_mmq`, so an environment variable of that name does nothing at all. Anything that
compares the two prefill paths has to compare two builds; round five does.

The one thing the run does show is how the 8B's prefill scatters at small batches: pp128 reads 197 with a
spread of 35 on the default arm of both passes and 224 to 226 with a spread of 5 on the other arms,
which is the first-invocation warmup of that model, not the setting.
