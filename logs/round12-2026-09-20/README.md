# What prefill is made of now, 2026-09-20

Before the packed-fp16 GEMM, prefill on this board was 92 percent one kernel, MMQ
([`logs/round4-2026-09-19/`](../round4-2026-09-19/)). The GEMM that replaced it is 1.4 to 1.6 times
faster per matmul, so the profile has moved and the question of what to optimise next has to be asked
again. Traced with [`scripts/kerntrace.cpp`](../../scripts/kerntrace.cpp) on the eight-patch build
(`trace-*.txt`, `log`).

## qwen2.5-1.5B

| kernel | pp512 | pp2048 |
|---|---|---|
| packed-fp16 GEMM, q4_K | 37 % | 68 % |
| packed-fp16 GEMM, q6_K | 8 % | 15 % |
| MMQ, q4_K and q6_K | | 4 % |
| flash attention | 2 % | 9 % |
| everything else | | 4 % |

MMQ has not disappeared and should not: the `Kcur` and `Vcur` projections are 256 rows, below the
kernel's threshold, so they keep the faster path. Flash attention grows with the prompt, as it must.

## qwen3.8-27B

| kernel | pp512 | pp2048 |
|---|---|---|
| packed-fp16 GEMM, IQ3_S | 29 % | 58 % |
| packed-fp16 GEMM, IQ3_XXS | 13 % | 25 % |
| packed-fp16 GEMM, IQ4_XS | 6 % | 12 % |
| flash attention (D=256) | | 1.5 % |
| gated delta net | | 1.3 % |

This model's prefill is now 95 percent the one GEMM. Two things follow. The flash-attention D=256 rows
and the gated delta net, both of which had their own rounds of work
([`logs/rdna1-fattn-remainder-2026-09-19/`](../rdna1-fattn-remainder-2026-09-19/),
[`logs/rdna1-gdn-concat-2026-09-19/`](../rdna1-gdn-concat-2026-09-19/)), are together under three percent
of this model's prefill and are not worth further effort. And whatever is left of the gap to RADV is in
the GEMM itself: per matmul at 2048 tokens it is at parity to 30 percent behind, `ffn_gate` 12077 against
11573 microseconds, `ffn_down` 14427 against 10386, `Qcur` 2475 against 1790. Rounds thirteen and
fourteen tried the two obvious remedies on it, and both came back negative.
