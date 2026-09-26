# Two community RADV patches measured, 2026-09-17

Both come from BC-250 owners and both are outside Mesa. They were built into one driver and compared
against the same Mesa 26.1.8 source unpatched, with the llama.cpp Vulkan binary held constant. Neither
driver was installed: both were loaded through `VK_ICD_FILENAMES` from a private prefix.

- **Compute-queue exposure** (`tri3gubki-ops/bc250-async-compute-bazzite`, patch 0001). Upstream RADV
  refuses to expose the dedicated compute queues on gfx1013, `/* GFX1013 is known to have broken compute
  queue */` in `ac_gpu_info.c`. The patch removes that and adds `CHIP_GFX1013` to
  `has_async_compute_threadgroup_bug`, which is the Iceland and Tonga workaround.
- **Signed-dot reassociation** (`dmorazasanchez/bc250-fsr4`, `bc250-fsr4-i24.patch`). gfx1013 is excluded
  from RADV's accelerated dot product, so the 4x8 signed dot expands to scalar multiplies; the patch
  rewrites that expansion using `imul24_relaxed` and reassociates the adds. Its author reports the native
  instruction returning wrong answers on this chip, so the fallback is being optimised, not replaced.

Both applied to Mesa 26.1.8 without modification. Harness: [`scripts/radv_patch_ab.sh`](../../scripts/radv_patch_ab.sh). Fedora 44, kernel 7.2.5 with the bc250 module, clock
pinned at 1500 MHz.

## The compute-queue patch does what it claims

`vulkaninfo` against the two drivers, same board, same boot:

| driver | queue families reported |
|---|---|
| stock 26.1.8 | one graphics family (1 queue), plus a sparse-binding family |
| patched | the same, **plus a compute family with 4 queues** |

So the queues are real and enumerable; upstream is hiding them instead of the hardware lacking them.

## Effect on llama.cpp: about one percent

Medians of nine samples, three alternated rounds of `-r 3`:

| test | stock | patched | change |
|---|---|---|---|
| qwen2.5-1.5B pp512 | 1850.18 | 1869.09 | +1.0 % |
| qwen2.5-1.5B tg64 | 212.35 | 215.87 | +1.7 % |
| qwen3-8B pp512 | 394.70 | 394.89 | 0.0 % |
| qwen3-8B tg64 | 39.07 | 39.05 | -0.1 % |

The qwen3-8B perplexity gate is 9.1388 on both drivers, bit-identical.

## Reading

Small and only on the small model. That fits what each patch does: llama.cpp submits its work on one queue,
so exposing three more changes nothing by itself, and the dot-product rewrite only pays where the quantized
vector path dominates, which on the 8B it does not at these shapes. The 1.5B gain of 1.0 and 1.7 percent is
outside the run-to-run spread here, both drivers being repeatable to about 0.1 percent, but it is small
enough that it should be repeated on another board before being called a result.

Neither patch is a reason to leave the distribution driver on its own. The compute-queue one is worth
knowing about for a different reason: it shows the queues enumerate on Mesa 26.1.8 with a 7.2.5 kernel,
which is the prerequisite for anything that wants to use them, and the correctness gate did not move.
