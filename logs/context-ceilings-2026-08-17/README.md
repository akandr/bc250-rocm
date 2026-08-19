# Context ceilings per model, 2026-08-17

Each model pushed until it fails, on ROCm under the working configuration.
Measurement is `llama-bench -p 0 -n 8 -d <depth>`: prime the cache to a depth,
then decode 8 tokens. That exercises the KV allocation and long-context decode,
which is what actually runs out, rather than prefill throughput.

The 14B rows were run with `GGML_CUDA_DISABLE_GRAPHS=1`, since HIP graph
instantiation fails on that model at a primed depth of 12000 for reasons
unrelated to context size (described in the top-level README).

## Results

| model | 8192 | 16384 | 24576 | 32768 | 131072 | 262144 |
|---|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 84.8 | 73.6 | | 62.7 | 27.6 | fails |
| qwen3-8B Q8_0 | 22.6 to 23.9 | 12.9 to 18.3 | fails | fails | | |
| qwen3-14B Q4_K_M | 12.6 | 7.3 | | fails | | |
| qwen3.8-27B UD-IQ3_XXS | 7.0 | fails | | fails | | |

Tokens per second. Blank means not run at that size.

The 1.5B row differs from the published context ladder (96.3, 85.5, 67.7)
because that ladder generated 32 tokens per point and this one generates 8.
Decode rate depends on the count, so the two are not interchangeable; these
were re-measured at 8 tokens in `consistent-tg8/` so the whole table is one
measurement.

The 27B fails at 16384 here while processing a 16384-token *prompt* works at
41.4 t/s (`../qwen38-2026-08-17/`). Generation needs the whole cache resident
alongside 11.09 GiB of weights; prompt processing does not hold as much live at
once. A column about decode has to use the decode measurement.

## The two failure modes, both memory

`failed to create context`: the allocation is refused before any kernel runs.
The 14B at 32768 fails this way, as does the 27B (in
`../qwen38-2026-08-17/`).

`ROCm error` at `ggml-cuda.cu:109`, with dmesg showing `amdgpu: SVM mapping
failed, exceeds resident system memory limit` (captured in
`dmesg-svm-limit.txt`): the run gets further and then aborts in the backend.
The 8B at 24576 and the 1.5B at 262144 fail this way. The 1.5B reached this
point only after three and a half hours of priming.

Neither is one of the defects documented in the top-level README, and the
campaign recorded zero GPU faults in dmesg.

## Two notes on reading these numbers

The 8B at 16384 is a range because two runs on different boots returned 12.9
and 16.6. That spread is wider than anything else measured here.

`inv52.log` contains a reading of 2.13 t/s for the 8B at 8192, which is wrong
and is kept only because it is what the run recorded. It executed immediately
after the 262144 attempt, a three-and-a-half hour job that ended by exhausting
memory. Re-measured on an idle board, the same configuration returns 22.8 and
22.6, and those are the numbers in the table. Anything measured directly after
a memory-exhausting run on this board should be treated the same way.

## The 8B repeats

`8b-repeats/` holds two further runs of the 8B at both depths, taken on an idle
board after the first pass. Collected across boots, its figures are 22.6, 22.8
and 23.9 at 8192, and 12.9, 16.6 and 18.3 at 16384.

That is a much wider spread than any other model here, roughly 40 percent at
16384, and it is the reason this row is given as a range. An earlier version of
the table quoted single values that happened to come from one boot each.
