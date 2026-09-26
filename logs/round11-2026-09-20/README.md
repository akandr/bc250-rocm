# q8_0 in the prefill GEMM, measured properly this time, 2026-09-20

q8_0 was taken out of the packed-fp16 GEMM in [`logs/round6-2026-09-19/`](../round6-2026-09-19/) on the
strength of a version that materialised the whole weight matrix in f16 before every matmul, which is not
the kernel that shipped: [`logs/round7-2026-09-20/`](../round7-2026-09-20/) decodes tiles inside the
kernel and turned the same comparison from a loss into a 36 percent win on q4_K. So the exclusion was
made on evidence that no longer applied, and it had to be re-measured.

`GGML_RDNA1_PKF16_Q8` switches q8_0 inside one build, so all three arms below are the same binary on the
same boot, interleaved, three repetitions each ([`scripts/round11.sh`](../../scripts/round11.sh), `log`):

| qwen3-8B Q8_0 | pp512 | pp2048 |
|---|---|---|
| MMQ (`build-hip-final`) | 244 (spread 55) / 260 | 269 / 264 |
| packed fp16, q8_0 on | 220 / 210 | 228 / 243 |
| packed fp16, q8_0 off | 261 / 260 | 264 / 265 |

The exclusion stands: q8_0 is 15 to 20 percent slower through this kernel. The off arm reproducing MMQ to
within a percent is the control that says the rest of the build is not doing anything to this model.

Perplexity with it on is 9.1125 against MMQ's 9.1273, better for the same reason it is better everywhere
else: MMQ quantises the activations to eight bits and this does not. So this is a speed decision and not
a correctness one.

Why q8_0 and not the others is a fair question, and the honest answer is that the measurement is firmer
than the explanation. The plausible part: for every other supported type MMQ has to unpack a codebook or
a nibble before it can multiply, while for q8_0 its weights are already the int8 operands its dot product
wants, so MMQ does no unpacking at all; and q8_0 carries 8.5 bits per weight against q4_K's 4.5, so the
tile loader moves nearly twice the bytes per value to produce the same number of f16 entries. The
arithmetic advantage of packed fp16 is real for q8_0 as well, and it does not pay for that.

The decoder is kept in the kernel behind the switch, off by default, so the measurement can be repeated
on another RDNA1 part without rewriting it.
