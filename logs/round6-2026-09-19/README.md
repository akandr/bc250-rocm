# The packed-fp16 GEMM inside llama.cpp: correct, faster per kernel, slower per model, 2026-09-19

[`logs/pk-gemm-prototype-2026-09-19/`](../pk-gemm-prototype-2026-09-19/) measured a standalone
packed-fp16 tile GEMM at 2.6 to 2.8 times MMQ's rate on the real prefill shapes. This is that kernel
written into llama.cpp ([`patches/llamacpp/rdna1-pkf16-gemm/`](../../patches/llamacpp/rdna1-pkf16-gemm/),
hooked in by [`scripts/apply_rdna1_pkf16.py`](../../scripts/apply_rdna1_pkf16.py), switchable at run time
with `GGML_RDNA1_PKF16=0`), and the answer is that it does not transfer. **It is not adopted.** The runs
are [`scripts/round6_pkf16.sh`](../../scripts/round6_pkf16.sh) and, for the coalesced store and the
narrow-matrix guard below, [`scripts/round6b.sh`](../../scripts/round6b.sh).

## Correct, and slightly more accurate than MMQ

`test-backend-ops test -o MUL_MAT` against the CPU: 682 of 682 with the kernel on, 649 of 649 with it
off, no failures. The perplexity gates *improve*: 8.9237 against MMQ's 8.9498 on the 1.5B and 9.1125
against 9.1273 on the 8B. That is expected, and worth remembering: MMQ quantises the activations to
eight bits, and multiplying f16 weights by f16 activations does not.

## Faster per kernel on the large matrices

Per op, the 1.5B's graph at 2048 tokens, both builds replayed twice, alternating, GPU otherwise idle
(`per-shape.log`; MMQ's second pass is a throttled reading, the packed-fp16 readings reproduce to 0.1
percent):

| shape | MMQ | packed fp16 | |
|---|---|---|---|
| `ffn_down` q6_K[8960,1536] | 22700 | **18592** | 1.22x |
| `ffn_gate` q4_K[1536,8960] | 19505 | **16215** | 1.20x |
| output head q6_K[1536,151936] | 345944 | **278550** | 1.24x |
| `ffn_down` q4_K[8960,1536] | 20665 | **18605** | 1.11x |
| `Qcur` q4_K[1536,1536] | 3533 | **3181** | 1.11x |
| `Kcur` q4_K[1536,256] | 757 | 1105 | 0.68x |
| `Vcur` q6_K[1536,256] | 824 | 1106 | 0.75x |

Narrow matrices lose because 256 rows is two tiles of 128, so the whole matmul is a handful of blocks.

## And slower on every model

`llama-bench`, three repetitions, two passes, interleaved (`round6b-log`):

| model | pp128 MMQ | pkf16 | pp512 MMQ | pkf16 | pp2048 MMQ | pkf16 |
|---|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 695 | 384 | 915 | 912 | 890 | 883 |
| qwen3-8B Q8_0 | 224 | 71 | 261 to 277 | 112 | 265 to 268 | 111 |
| qwen3.8-27B IQ3_XXS | 65 | 28 | 72 | 41 | 64 | 44 |

Decode is unchanged (tg64 180.4 against 183.4 on the 1.5B), as it should be: the kernel only takes
batches of 64 tokens and up.

## Why the per-op win does not become a per-model win

The kernel takes its weights in f16, so the hook dequantises the whole weight matrix into a pool buffer
before every matmul. For the 1.5B's `Qcur` that is 4.7 MB; for the 8B's `ffn_down` q8_0[12288,4096] it is
100 MB written and read back per call, against the 53 MB MMQ reads once, directly, in its quantised form.
That decides it between the two columns above: the models whose matrices are large in bytes
lose the most, and the 1.5B, whose largest weight is 27 MB, is level.

The per-op replay hides it because it runs the same op thousands of times: the pool buffer is allocated
once and stays warm, and the dequantised weights are reused across iterations, which never happens in a
model where each layer's weights are touched once per batch. That is a measurement lesson as much as a
kernel one, a per-op replay flatters any design that caches work across calls.

So the arithmetic finding of [`logs/alu-rates-2026-09-19/`](../alu-rates-2026-09-19/) stands and the
naive way of using it does not. A packed-fp16 GEMM that beats MMQ on this board has to dequantise into
shared memory inside the kernel, one tile at a time, per quantisation type, exactly as MMQ does with
int8 and as ggml-vulkan's `mul_mm.comp` does in f16. That is a much larger piece of work than this
prototype, and it is where the remaining prefill gap lives.

One smaller finding from the same session: the first version of the store loop wrote the f32 output with
scalar stores whose lanes were eight rows apart, so each instruction touched sixteen cache lines and used
four bytes of each; a coalesced `float4` store down the column is in the file now. It was worth about
one percent, not the factor the analysis suggested, which is its own lesson about guessing at bottlenecks.
