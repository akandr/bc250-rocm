# The deterministic baseline is only deterministic at one chunk, 2026-08-23

Generated with `BC250_TEMP` counting zeroed GEMM outputs, `GGML_CUDA_NO_POOL=1`,
context 2048, ubatch 512. The prediction was written before the run and is in
`prediction.txt`; raw counts in `runs.txt`.

## What was asked

At one chunk there are four batches and the zeros fall on the first call of
batches two, three and four. Two readings fit that and differ at a chunk
boundary: either every batch after the very first in the process is affected, or
something resets between chunks. Two chunks means eight batches, and call 145,
the first batch of the second chunk, separates them.

## What happened instead

The premise did not survive. Six runs at two chunks:

| run | zeros | positions |
|---|---|---|
| 1 | 4 | 37, 73, 109, 181 |
| 2 | 32 | 37, 73, 109, 181, then a contiguous block 253 to 280 |
| 3 | 9 | 37, 73, 109, 181, 253, 254, 255, 256, ... |
| 4 | 40 | as run 3, with a longer block |
| 5 | 5 | 37, 73, 109, 181, 253 |
| 6 | 4 | 37, 73, 109, 181 |

`GGML_CUDA_NO_POOL=1` does not make this deterministic at two chunks. It does at
one: three further runs at a single chunk give three zeros at 37, 73 and 109 and
a perplexity of 14.1344, which with the five from the previous cycle is eight
identical runs.

## The correction this forces

The previous cycle concluded that the memory pool causes the defect's
non-determinism. That is too general. What the evidence supports is narrower:
with the pool disabled, the defect is deterministic **at one chunk**. At two
chunks it is not, pool or no pool, so the pool cannot be the whole explanation
for the variability.

## What is stable, and what is not

Four positions appear in every one of the six runs: 37, 73, 109 and 181. Beyond
those, runs diverge, and when they do it is as a contiguous cascade starting at
253 rather than as scattered extra zeros. Two runs stop at the stable four.

Two batch starts are consistently clean: 145 and 217. That 145 is clean is what
a per-chunk reset would predict, since it is the first batch of the second
chunk, exactly as call 1 is the first batch of the first. But 217 is also clean
and no per-chunk reading predicts that, so the tidy model does not survive
either. The honest position is that the simple one-zero-per-batch rule holds at
one chunk and breaks at two, in a way not yet described.

## Why this matters more than it looks

The exactness at one chunk was the basis for treating the batch-boundary model as
established. It is established for that workload only. Anything measured at one
chunk and generalised to inference at large is on weaker ground than the
five-identical-runs figure suggested, and the cascade at two chunks says
something the model does not currently account for.
