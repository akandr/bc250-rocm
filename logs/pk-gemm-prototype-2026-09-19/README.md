# What a packed-fp16 GEMM is worth on this board: a prototype, 2026-09-19


**Ceiling correction, 25 September.** The ceilings this page divides by, 4.74 TFLOP/s fp32 and 14.40
packed fp16, were measured below the clock cap. Re-measured with the clock verified at 1500 MHz they
are 6.52 and 13.02, and packed fp16 is twice fp32, not three times
([`logs/alu-rates-recheck-2026-09-25/`](../alu-rates-recheck-2026-09-25/)). Every percentage of peak on
this page is therefore against the wrong denominator; the measured rates themselves are unaffected.

[`logs/alu-rates-2026-09-19/`](../alu-rates-2026-09-19/) established that `v_pk_fma_f16` runs at 14.4
TFLOP/s on gfx1013 against 4.74 for fp32 and 2.08 for the emulated int8 dot product MMQ depends on, and
that ROCm's prefill (2.5 to 2.9) is at its arithmetic's limit while RADV's shaders (4.9 to 5.4) are
fp16 work at a third of the packed ceiling. The obvious question is what a packed-fp16 GEMM would
actually reach here, since the answer decides whether writing one into llama.cpp is worth it.

[`scripts/pk_gemm.cpp`](../../scripts/pk_gemm.cpp) is a plain LDS-staged tile GEMM: A and B tiles into
shared memory, each thread holding a TM x TN block of accumulators as `__half2`, one `v_pk_fma_f16` per
two outputs, no double buffering and no assembly. Weights are taken already in f16, which is what a
dequantising kernel would put in shared memory. Tile geometry was swept; the best configuration is BM
128, BN 128, BK 32, TM 8, TN 16, and the timings below are the best of three passes of thirty
repetitions each, reproducible to a percent across runs (`pk_gemm-best.log`). The comparison is against
the same shapes measured on both backends in
[`logs/op-perf-hip-vs-vulkan-2026-09-17/`](../op-perf-hip-vs-vulkan-2026-09-17/), at the same 2048-token
batch:

| shape, 2048 tokens | ROCm MMQ today | RADV | this prototype | of the 14.4 ceiling | over MMQ | over RADV |
|---|---|---|---|---|---|---|
| `Qcur` [1536,1536] | 2.74 | 5.40 | **7.09** | 49 % | 2.6x | 1.31x |
| `ffn_gate` [1536,8960] | 2.89 | 4.87 | **7.99** | 56 % | 2.8x | 1.64x |
| `ffn_down` [8960,1536] | 2.73 | 5.43 | **7.77** | 54 % | 2.8x | 1.43x |

At a 512-token batch the same kernel reads 4.80, 7.65 and 5.00 TFLOP/s, so the small-batch shapes are
where the tiling still leaves something.

Tile geometry matters a great deal and not uniformly: `BK` 32 beats 16 and 64, `TN` 16 beats 8, and the
best shape for one matmul is not the best for another. A production kernel would pick per shape, as
MMQ's tables already do.

So the finding is: **2.6 to 2.8 times MMQ's rate on the real shapes, and 1.3 to 1.6 times RADV's**, from
a prototype with no double buffering, purely by doing the multiplication in the format the chip is fast
at. That is the size of the prefill opportunity, and it is a kernel-writing job instead of a parameter
change: a real one has to dequantise q4_K, q6_K and the IQ types into f16 tiles in shared memory, which
is what MMQ already does in int8 and what ggml-vulkan's `mul_mm.comp` does in f16.

Two caveats are part of the finding. `v_pk_fma_f16` accumulates in f16, so a real kernel needs to
promote to f32 every so many K steps, and that costs some of the rate (the prototype does not, and its
worst relative error against a CPU reference is 0.0000 at K=256 with small values, which says nothing
about K=8960 with model weights). And `v_fma_mix_f32`, the instruction that would accumulate in f32 from
f16 inputs, runs at the fp32 rate, so it buys nothing over the current path: the 3x is specifically the
packed-f16-in, f16-out instruction.
