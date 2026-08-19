# Qwen3.8-27B on both backends, 2026-08-17

Added to check that the working configuration holds on a model released after
this work, and to find where a 16 GiB board runs out. The model is
`Qwen3.8-27B-UD-IQ3_XXS`, 11.09 GiB of weights for 27.3 billion parameters, the
largest dense model tried here.

Everything ran strictly one process at a time. That matters at this size: the
first attempts failed twice, once inside `ggml_cuda_mul_mat_cublas` and once at
model load, purely because a second process still held memory. Those failures
look like the defects documented elsewhere in this repo and are not.

## Files

| file | what it is |
|---|---|
| `inv50.log` | the run log, all results in order |
| `q38_rocm_bench.log`, `q38_vulkan_bench.log` | pp128, pp512 and tg128 on each backend |
| `q38_rocm_ppl.log`, `q38_vulkan_ppl.log` | perplexity gates, 2 chunks at ctx 2048 |
| `q38_ctx_*.log` | prompt-processing ladder at 4096, 8192, 16384, 32768 |
| `q15_rocm.log`, `q15_vulkan.log` | the 1.5B measured in the same session, as an anchor against the published numbers |

## Results

| backend | pp128 | pp512 | tg128 | perplexity |
|---|---|---|---|---|
| ROCm | 62.6 | 69.2 | 7.84 | 6.2487 |
| Vulkan | 93.1 | 97.9 | 17.18 | 6.2651 |

The two backends agree on perplexity to 0.26 percent, which is the check worth
having: a backend producing fast nonsense looks identical in the rate columns.

Prompt processing scales to 16384 (59.2, 49.1, 41.4 t/s at 4k, 8k, 16k) and
fails at 32768 with `failed to create context`, an allocation refused before any
kernel runs. Generation does not reach as far: priming the cache to 16128 and
then decoding runs out of memory, so the decode ceiling is 8192. That
distinction is measured in `../context-ceilings-2026-08-17/`.

The 1.5B anchor came out at 807.9 t/s prefill on ROCm and 1844.3 on Vulkan,
within a percent of the published campaign numbers, which is what makes the
27B numbers comparable to the rest of the document.

Collected by `llama-bench` and `llama-perplexity` invocations on the qwen3.8 27B model, the same call shapes used elsewhere here.
