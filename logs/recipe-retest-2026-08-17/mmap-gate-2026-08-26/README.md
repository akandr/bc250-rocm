# The mmap-on gate, measured, 2026-08-26

The parent page lifts the recipe's `--no-mmap` requirement and gives as evidence
that the 1.5B returns 8.9442 on three of three runs with mapping enabled. Those
runs were never kept, and no captured gate anywhere in this repository ran with
mapping on, so the claim had no artifact and nothing else covered it.

Measured rather than caveated, nine days later, on the configuration the README
now recommends:

| load mode | run | perplexity |
|---|---|---|
| `-lm mmap` | 1 | 8.9442 +/- 0.17287 |
| `-lm mmap` | 2 | 8.9442 +/- 0.17287 |
| `-lm mmap` | 3 | 8.9442 +/- 0.17287 |
| `--no-mmap` | control | 8.9442 +/- 0.17287 |

Three runs with mapping enabled, bit-identical to each other and to a no-mmap
control taken in the same session, error bar included. The original claim was
right and now has evidence behind it.

The control matters more than the repeats here. The reference value 8.9442 is
quoted throughout this repository from runs taken on other days and other
kernels, so comparing against it alone would have left the door open to the two
arms differing for some other reason. Running both in one sitting closes that.

Same configuration as every other gate in this work except the load mode:
qwen2.5-1.5B Q4_K_M, context 4096, eight wikitext chunks, flash attention on,
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, native rocBLAS. Kernel 7.1.8, navi12 SDMA
microcode, `amdgpu.gpu_recovery=0`.
