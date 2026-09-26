# The IQ types and q5_K in the prefill GEMM, 2026-09-20

[`logs/round7-2026-09-20/`](../round7-2026-09-20/) covered q4_K and q6_K, which left the two models made
of codebook types where they were: the qwen3.8-27B is IQ3_S 53 percent, IQ3_XXS 23, IQ4_XS 13 and Q5_K 7
by bytes, and the qwen3.6-35B MoE is Q5_K 10 percent in its dense tensors. This adds tile decoders for
q5_K, IQ2_XXS, IQ3_XXS, IQ3_S and IQ4_XS to the same kernel, each thread decoding one row's 32 values
per stage straight into shared memory, the codebook grids read from the tables `vecdotq.cuh` already uses
([`scripts/round10.sh`](../../scripts/round10.sh)).

## Correctness first, because a wrong decoder would be invisible in a speed number

| check | MMQ | with the new decoders |
|---|---|---|
| qwen3.8-27B perplexity, 2 chunks | 6.2472 | **6.2482** |
| qwen3.6-35B MoE perplexity, 3 chunks | 6.7545 | 6.7629 |
| qwen2.5-1.5B perplexity, 8 chunks | 8.9498 | 8.9274 (unchanged from the q4/q6 build) |
| qwen3.8-27B greedy text, 48 tokens | | **identical** |

The 27B's perplexity moves by 0.016 percent, which is the strongest evidence available that IQ3_S,
IQ3_XXS and IQ4_XS are decoded correctly: those three are 89 percent of that model's weights and its
gate runs them at batch 512 through the whole graph.

## Speed

`llama-bench`, three repetitions, two passes, interleaved (`log`):

| model | pp512 MMQ | pkf16 | pp2048 MMQ | pkf16 |
|---|---|---|---|---|
| qwen3.8-27B IQ3_XXS | 71.5 / 71.8 | 69.9 / 72.5 | 65.0 to 65.9 | **75.1 / 75.7** (+15 %) |
| qwen3.6-35B MoE IQ2_M | 297.4 / 296.9 | **318.5 / 321.2** (+7.3 %) | 305.4 / 305.9 | **329.5 / 331.4** (+7.9 %) |
| qwen2.5-1.5B Q4_K_M | 914.2 / 914.5 | 1244.0 / 1244.6 | 890 | 1190 |

The 27B gains at 2048 tokens and is level at 512; the campaign that follows
([`logs/fedora44-campaign-pkf16-all-2026-09-20/`](../fedora44-campaign-pkf16-all-2026-09-20/)), which
measures prefill in its own invocation, puts its pp512 at 81.2 against 72.8, 11.5 percent. The MoE gains
7 to 8 percent, which is what its dense q5_K is worth.

## Why the MoE's experts are not the next 81 percent

Its experts are 81 percent of its bytes and they do not come through this path: `ggml_cuda_mul_mat_id`
sends them to `ggml_cuda_mul_mat_q` with an `ids` tensor. That looked like the largest prefill item left,
and it is not, for a reason worth writing down before anyone implements the indirection. The model has
**256 experts and uses 8 per token**, so a 512-token batch spreads 4096 (token, expert) pairs over 256
experts: sixteen tokens each, against a weight matrix of 512 rows. A 512 x 16 matmul is matvec-shaped,
not GEMM-shaped; this kernel needs 256 columns to beat MMQ and would run a 128-column tile eleven
twelfths empty. Even at 2048 tokens it is 64 columns per expert. MMQ's `ids` path, which packs the
per-expert token lists and uses small tiles, is the right kernel for that shape, and a packed-fp16
version of it would be a different kernel from this one, not an extension of it.
