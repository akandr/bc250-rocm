# The batch-boundary pattern is not one model's pattern, 2026-08-23

Generated with `BC250_TEMP` on the exact baseline, `GGML_CUDA_NO_POOL=1`, context
2048, ubatch 512, fp16 compute. Prediction in `prediction.txt`, counts in
`runs.txt`.

## Why

Everything established about this defect came from qwen3-8B: the batch-boundary
model, the determinism under `NO_POOL`, the single-batch workaround. A pattern
seen on one model is one model's pattern until shown otherwise, and the write-up
had begun generalising.

qwen3-14B is the other model the defect is documented to affect. Its layer count
differs, so the calls per batch should differ, but the structure should not.

## Result: it replicates exactly

| | qwen3-8B | qwen3-14B |
|---|---|---|
| GEMM calls at one chunk | 144 | 160 |
| calls per batch | 36 | 40 |
| zeros | 3, at 37, 73, 109 | 3, at 41, 81, 121 |
| runs identical under `NO_POOL` | 8 of 8 | 3 of 3 |
| single batch, ubatch 2048 | 0 zeros, 7.2768 | 0 zeros, 6.3539 |
| f32 reference at ubatch 2048 | 7.2672 | 6.3454 (see below) |
| residual fp16 error at one batch | 0.13 percent | 0.13 percent |

On the 14B, 160 calls over four batches is 40 per batch, so batch starts fall at
calls 1, 41, 81 and 121. The zeros are at 41, 81 and 121: the first call of every
batch after the first, exactly as on the 8B, with the spacing following the
model's layer count rather than any fixed number.

The determinism holds too, three identical runs, and so does the single-batch
workaround, with a residual error against f32 of 0.13 percent on both models.

## The 1.5B is clean for a different reason than "clean" suggests

qwen2.5-1.5B Q4_K_M records **zero GEMM calls** through this path. Its perplexity
is fine, 8.3567, but not because the model resists the defect: it never reaches
the fp16 cuBLAS path at all, which this repository documents separately for
Q4_K models. The instrument saw nothing because there was nothing to see.

That distinction was written into the prediction beforehand, because reading "the
1.5B is clean" as evidence about the defect would be the same mistake this work
has made before: treating an instrument's silence as a measurement.

## What this adds

The batch-boundary model is a property of the defect rather than of one model.
Anything built on it now rests on two models with different shapes, and the two
agree on structure, determinism, workaround and residual error.

One input to the residual had no captured artifact, and now does. `runs.txt` holds
the fp16 runs and the single-batch value 6.3539, and the first model's f32
reference 7.2672 is in
[`../fp16-pool-2026-08-19/f32ref.log`](../fp16-pool-2026-08-19/f32ref.log), but the
14B's f32 reference 6.3454 appeared in no captured file anywhere: it was measured
at the time and the output was not kept.

Re-run on 25 August with the recorded method, `-c 2048 -ub 2048`, one chunk,
`GGML_CUDA_NO_POOL=1`, it comes back at exactly 6.3454, and the fp16 arm beside it
comes back at exactly 6.3539
(`f32ref-remeasured-2026-08-25.log`). Both figures were right; only the evidence
was missing. The fp16 value reproducing as well is the check that the conditions
were the same ones, since that value was already captured.

The residual stands at 0.134 percent, quoted as 0.13.
