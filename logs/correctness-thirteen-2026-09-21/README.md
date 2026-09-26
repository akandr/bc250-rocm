# The correctness table, re-measured on the thirteen-patch build, 2026-09-21

The front page's perplexity table and its two verification gates both dated from the original Fedora 44
page, which is the three-patch build. Three of the patches added since change the arithmetic: the
prefill GEMM multiplies in fp16 where MMQ quantises the activations to eight bits, q8_0 joined it, and
so did a mixture-of-experts model's experts. Anyone following the recipe would have run the gates and
seen figures that did not match.

Context 2048, eight chunks of `wiki.test.raw`, flash attention on, default compute type, one boot, both
backends ([`scripts/correctness_table.sh`](../../scripts/correctness_table.sh), [`summary-log`](summary-log)):

| model | ROCm, thirteen patches | Vulkan | ROCm, three patches | change |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 8.0093 | 8.0485 | 8.0366 | -0.0273 |
| qwen3-8B Q8_0 | 7.3586 | 7.3948 | 7.3522 | +0.0064 |
| qwen3-14B Q4_K_M | 6.3896 | 6.4547 | 6.3986 | -0.0090 |
| deepseek-r1-14B Q4_K_M | 5.9865 | 6.0505 | 6.0013 | -0.0148 |
| qwen3.6-35B-A3B MoE IQ2_M | 5.1820 | 5.1975 | 5.1887 | -0.0067 |
| qwen3.8-27B UD-IQ3_XXS | 5.2938 | 5.3385 | 5.2899 | +0.0039 |

**Every Vulkan figure reproduces the August measurement to four decimals.** That is why the
comparison readable: the reference did not move, so the ROCm column's movement is the patches and
nothing else.

Four models improved and two are slightly worse, and the split is the one the patches predict. The four
that improved have weights narrower than eight bits, so replacing MMQ's eight-bit activations with fp16
is a net gain. The 8B is q8_0, whose weights are already the int8 operands MMQ wants, so there is no
weight-side gain to set against accumulating in fp16; it is 0.0064 worse, which is the cost recorded
when that patch landed. The 27B is 0.0039 worse. The largest movement anywhere in the table is a tenth
of one standard error, and every ROCm value remains below its Vulkan pair.

## The verification gates

| gate | thirteen patches | three patches |
|---|---|---|
| qwen2.5-1.5B, context 4096, eight chunks | 8.9274 | 8.9442 |
| qwen3-8B, context 2048, two chunks, default compute | 9.1125 | 9.1117 |
| the same with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | 9.1130 | 9.0975 |

The 1.5B gate reads exactly the 8.9274 the prefill GEMM's patch header quotes. The compute-type setting
has almost stopped mattering on the second gate, 9.1125 against 9.1130 where it used to be 9.1117
against 9.0975, because that model's prefill matmuls now go through the packed-fp16 GEMM instead of
rocBLAS and the rocBLAS setting has little left to change.

## Files

`summary-log` is the run, `hip_*.log` and `vk_*.log` the per-model output, and `gate_*.log` the three
gates.
