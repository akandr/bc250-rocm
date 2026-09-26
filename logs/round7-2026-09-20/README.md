# The packed-fp16 GEMM, dequantising inside the kernel: 36 percent of the 1.5B's prefill, 2026-09-20

[`logs/round6-2026-09-19/`](../round6-2026-09-19/) ended with the kernel correct, faster per matmul and
slower per model, because it took its weights already in f16 and so materialised the whole weight matrix
before every call. This is the same kernel with the tiles dequantised inside it, which is how MMQ and
ggml-vulkan's `mul_mm.comp` do it: a BM x BK block of weights is decoded straight into shared memory,
the activations are read as f32 and converted while staging, and nothing else is written to memory.
[`patches/llamacpp/rdna1-pkf16-gemm/mmf16-rdna1.cu`](../../patches/llamacpp/rdna1-pkf16-gemm/mmf16-rdna1.cu),
q4_K and q6_K, hooked in by [`scripts/apply_rdna1_pkf16.py`](../../scripts/apply_rdna1_pkf16.py) and run
by [`scripts/round7.sh`](../../scripts/round7.sh).

## Per matmul

The 1.5B's graph at 2048 tokens, both builds replayed twice, alternating, GPU otherwise idle
(`per-shape.log`; every reading reproduces to 0.1 percent):

| shape | MMQ | in-kernel packed fp16 | |
|---|---|---|---|
| `ffn_gate` q4_K[1536,8960] | 19508 | **12077** | 1.62x |
| output head q6_K[1536,151936] | 346028 | **226879** | 1.53x |
| `ffn_down` q6_K[8960,1536] | 22678 | **14887** | 1.52x |
| `ffn_down` q4_K[8960,1536] | 20660 | **14427** | 1.43x |
| `Qcur` q4_K[1536,1536] | 3532 | **2475** | 1.43x |
| `Kcur` q4_K[1536,256] | 756 | 754 | guard sends it to MMQ |

The materialising version reached 1.11 to 1.24 times on the same shapes; removing the round trip through
memory is the difference between that and 1.43 to 1.62.

## Per model

`llama-bench`, three repetitions, two passes, interleaved (`round8-log`):

| model | pp512 MMQ | pkf16 | pp2048 MMQ | pkf16 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 915 | **1245** (+36 %) | 890 | **1190** (+34 %) |
| qwen3-14B Q4_K_M | 99 | **117 to 125** (+18 to 27 %) | 96 | **117 to 126** (+22 to 32 %) |
| qwen3.6-35B MoE IQ2_M | 301 | **311** (+3.3 %) | 306 | **318** (+3.8 %) |
| qwen3-8B Q8_0 | 265 | 261 | 264 | 264 |

The MoE gains only what its q4_K and q6_K tensors are worth, four percent of its bytes; its experts are
IQ2 and IQ3, which this kernel does not handle yet. Q8_0 was measured and removed from the predicate: on
the 8B it read 112 against MMQ's 265 at pp512, because that type's int8 `vec_dot` is short and the matmul
is close to memory bound, so there is nothing for the arithmetic to win back. Decode is untouched, the
kernel only takes batches of 256 tokens and up.

## Correctness

The honest statement first: `test-backend-ops -o MUL_MAT` has exactly **one** case that this kernel can
take (n >= 64 and m >= 512 at the time of the run), so its 657 passes say almost nothing here. The
evidence is the model-level checks, which run prefill at batch 512 through q4_K and q6_K weights,
including the q6_K output head over the whole vocabulary
([`scripts/round8.sh`](../../scripts/round8.sh), which rebuilt both trees first because the gate
binary from the previous run was stale, `round8-log`):

| check | MMQ | in-kernel packed fp16 |
|---|---|---|
| qwen2.5-1.5B perplexity, 8 chunks | 8.9498 | **8.9274** |
| qwen3-14B perplexity, 2 chunks | 7.7536 | 7.7691 |
| qwen2.5-1.5B greedy text, 48 tokens | | **identical** |

The 1.5B's perplexity improves by 0.25 percent and the 14B's worsens by 0.20. Both are expected: MMQ
quantises the activations to eight bits and this does not, while f16 accumulation (promoted to f32 every
32 steps) is coarser than MMQ's exact int32 sums. The two effects have opposite signs and neither is
large.

## The batch threshold

At 128 tokens the kernel is 22 percent *behind* MMQ on the 1.5B (542 against 690): 128 columns is one
BN tile, so the whole matmul is as many blocks as the weight matrix has 128-row strips, which does not
fill 40 CUs. At 512 it is 36 percent ahead. The predicate therefore requires 256 tokens, and also 512
rows, since 256-row matrices lose for the same reason.
The 256-token point the predicate turns on was measured separately, with the campaign that adopted
the kernel ([`logs/fedora44-campaign-pkf16-2026-09-20/`](../fedora44-campaign-pkf16-2026-09-20/)).
