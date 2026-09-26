# Dequantise and multiply with rocBLAS instead of MMQ, 2026-09-19

`GGML_CUDA_FORCE_CUBLAS` is a compile-time option, so this is a second build of the seven-patch tree
([`scripts/round5_cublas.sh`](../../scripts/round5_cublas.sh)), interleaved with the normal one, three
repetitions, two passes (`log`).

| model | pp128 MMQ | cuBLAS | pp512 MMQ | cuBLAS | pp2048 MMQ | cuBLAS |
|---|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 695 | **765** | 914 | **958** | 890 | **933** |
| qwen3-8B Q8_0 | 226 | 84 to 146 | 263 to 279 | 153 to 155 | 265 | 156 |
| qwen3.6-35B MoE IQ2_M | 160 | 131 | 301 | 188 | 306 | 184 |
| qwen3.8-27B IQ3_XXS | 65 | 18 to 27 | 72 | 37 to 54 | 64 | 49 to 53 |

The small q4_K model gains 5 to 10 percent; everything else loses 30 to 60. Decode is unchanged on the
1.5B (182 against 185) and reads higher on the MoE, where the MMQ arm was throttled. The gates move
slightly, as an f16 multiply should: 8.9582 against 8.9498 on the 1.5B, 9.1633 against 9.1273 on the 8B.

The trace of the 1.5B's pp512 says what the build actually runs: 386 dispatches of a Tensile kernel
(`Cijk_Alik_Bljk_HB_MT64x64x8_...`), 958 ms, 90 percent of the kernel time, plus 193 `convert_unary`
dequantisations at 47 us. Against MMQ's 386 dispatches and 1060 ms for the same work. So on the one
model where it wins it wins by running a 10-percent-faster kernel, not by a different order of
magnitude, and on the larger models rocBLAS's HGEMM for these shapes is far worse than MMQ.

The useful conclusion is the negative one, and it is what sent the investigation to the arithmetic
rates ([`logs/alu-rates-2026-09-19/`](../alu-rates-2026-09-19/)): swapping emulated int8 for rocBLAS's
f16 GEMM does not close a 2x gap, because that Tensile kernel reaches about 3 TFLOP/s where the chip's
packed-fp16 ceiling is 14.4.
