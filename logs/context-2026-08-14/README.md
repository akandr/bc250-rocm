# Context ladder, 2026-08-14

`ctx.log` is the run that first established how far context can be pushed on
this board. It holds two sessions: the ladder itself (8192 through 65536 on the
1.5B, plus the 14B at 8192 and its failure at 16384), and a later dedicated
attempt at 131072 with a three-hour budget, run twice, once with HIP graph
capture on and once off.

Two things in it are worth knowing before reading the numbers.

The 131072 rows generate 8 tokens where the shorter rows generate 32. Decode
rate depends on that count, so the two are not directly comparable; the
per-model ceilings in `../context-ceilings-2026-08-17/` re-measure everything at
8 tokens for that reason.

The 14B failure at 16384 recorded here is not a memory limit. It was later
attributed to HIP graph instantiation failing at a primed depth of 12000, and the
model is reported to run a 16384-token context correctly with
`GGML_CUDA_DISABLE_GRAPHS=1`.

Two things about that, noted 26 August. The failing run above is deepseek-r1-14B,
not qwen3-14B, which matters because other pages attribute the same limit to the
other model. And the success it is contrasted with is a perplexity run at that
context rather than the primed-depth decode that failed here, so the pair is not
a before and after of the same measurement. No run of this configuration with the
flag set survives anywhere. The attribution may well be right; it is not
demonstrated by what is kept.

Collected by repeated `llama-bench` invocations at increasing `-d` depth, the same measurement the context-ceiling table uses; see `scripts/decode_variance.sh` for the same call shape.
