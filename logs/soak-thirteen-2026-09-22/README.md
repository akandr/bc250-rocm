# Eight hours on the thirteen-patch build, 2026-09-21 to 2026-09-22

None of the kernels added on 20 and 21 September had run for hours: the packed-fp16 prefill GEMM, its
expert path, and the matrix-vector kernel with the IQ codebooks in shared memory. Between them they
handle every prefill matmul on four of the six models and most of the decode on two. The soak this
repository had behind it was 41 rounds of one model on kernel 7.1.8 with three patches
([`logs/fedora44-soak8-2026-09-16/`](../fedora44-soak8-2026-09-16/)), which says nothing about any of
them.

[`scripts/soak_thirteen.sh`](../../scripts/soak_thirteen.sh), 17:23 to 01:28, kernel 7.2.5 with the
bc250 amdgpu module, `amdgpu.gpu_recovery=0`, GPU clock policy at 1500 MHz, `HSA_ENABLE_SDMA=0`,
`ollama` stopped. Each round is one model: a `llama-bench` pp512/tg64 pair, then a six-chunk
perplexity gate checked against the value measured when the patch that touches that model landed.
Every fourth round adds a `test-backend-ops perf -o MUL_MAT` sweep, because model loads and frees are
the historically fragile part.

## Result

| | value |
|---|---|
| rounds | 152, 38 per model |
| gate mismatches | **0** |
| distinct gate values per model | **one each**: 10.2088, 9.4017, 6.2265, 6.2737 |
| allocation-churn sweeps | 38 of 38 passed |
| kernel fault lines, this boot and the one before | **0** |
| edge temperature | 62 to 76 C |

Throughput, medians over the 38 rounds of each model:

| model | pp512 | spread | tg64 | spread |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1781.63 | 0.38 % | 196.99 | 1.53 % |
| qwen3-8B Q8_0 | 414.82 | 1.51 % | 38.74 | 1.11 % |
| qwen3.6-35B-A3B MoE IQ2_M | 594.58 | 0.99 % | 71.78 | 1.07 % |
| qwen3.8-27B UD-IQ3_XXS | 104.38 | 2.28 % | 15.18 | 0.59 % |

Spread is max minus min as a percentage of the median. Nothing drifts across the eight hours: the
first and last rounds of each model sit inside those ranges, and the widest single figure, the 27B's
prefill at 2.28 percent, is the model that fills memory and carried a comparable spread in every
earlier measurement.

**Four perplexity values, 152 evaluations, one number each.** That is the part worth stating plainly.
The expert-path GEMM changes the arithmetic of a mixture-of-experts model's prefill and the shared-memory
codebook changes where the 27B's matvec reads its tables; if either were occasionally wrong, a gate
repeated 38 times is where it would show.

## Against the campaign

The four models it shares with the campaign read a little differently there: 1799.2, 405.3, 589.9 and
102.4 against 1781.6, 414.8, 594.6 and 104.4 here
([`logs/fedora44-campaign-experts-2026-09-21/`](../fedora44-campaign-experts-2026-09-21/)). The
harnesses differ in three ways and no attempt is made here to say which of them accounts for what: the
campaign measures prefill and decode in separate invocations where this measures both in one, it takes
three samples where this takes two, and it pools two runs of eighteen samples where this is a median
of 38 single rounds spread over eight hours. The campaign numbers remain the ones the front page
quotes, because they are the ones measured against Vulkan on the same boot.

## What it does not cover

No PyTorch. The earlier eight-hour soak ran a training loop every third round and returned an
identical final loss thirteen times; this one rotates four llama.cpp models instead, because the
kernels under test are llama.cpp's. Neither run exercises both.

It is also the first long run on kernel 7.2.5 with `amdgpu.gpu_recovery=0`. The rare page fault that
ended two Fedora 43 soaks after about 190 and 254 rounds did not appear in 152, which is fewer rounds
than either of those, so this is consistent with that rate, not evidence against it.

## Files

`log` is the round-by-round record. `r1_1.5b_*` and `r152_8b_*` are the first and last rounds'
benchmark and perplexity output, `churn_4.log` and `churn_148.log` an early and a late
allocation-churn sweep. The remaining 600 per-round files are on the board and not kept here.
