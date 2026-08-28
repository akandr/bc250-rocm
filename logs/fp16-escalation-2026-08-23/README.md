# Escalation is not data corruption, 2026-08-23

Generated with `BC250_TEMP` and `BC250_DEVCHECK` together, `GGML_CUDA_NO_POOL=1`,
qwen3-8B at context 2048, ubatch 512, three chunks. Prediction in
`prediction.txt`, raw lines in `calls.txt`.

## What was asked

At an isolated batch-start zero the operands are intact: good input, empty
output. Once a batch escalates and every call in it returns zero, that question
had not been asked. Either the operands stay intact, making escalation the same
output-side failure applied to more calls, or they go bad, making the first
zeroed call corrupt something later calls read and the cascade a propagation.
Those are different shapes of defect.

## Result: the operands stay intact throughout

| call | state | GEMM output | weight sum | activation sum |
|---|---|---|---|---|
| 1 | clean | 9361.13 | 84296 | 16499.4 |
| 37 | isolated zero | 0 | 84296.1 | 17028.6 |
| 38 | clean | 23911 | 88083.7 | 32439.6 |
| 253 | cascade start | 0 | 84296 | 17343 |
| 258 | inside cascade | 0 | 91141.5 | 95397.8 |
| 259 | clean, cascade ended | 67809.6 | 92217.1 | 122251 |
| 330 | inside cascade | 0 | 91141.4 | 95939.5 |
| 340 | inside cascade | 0 | 93913.7 | 272061 |

Deep inside a cascade, at calls 330 and 340, both operands are healthy and of the
magnitude their position warrants, and the output is still exactly zero. The
weight sums are internally consistent too: calls 258 and 330 are both `Vcur-5`
and report 91141.5 and 91141.4 for the same weight, while their activations
differ as different tokens should.

So escalation is the same failure applied to every call in the batch, not a
propagation of bad data. The defect is wholly on the output side, at every scale
it has been observed.

The scale of this particular run, which `calls.txt` records and this page had not
quoted: 109 of 432 calls returned zero, and the perplexity came out at 189.2381.
That is the three-chunk configuration, so it sits far above the one-chunk runs
that return three zeros of 144 and a perplexity near 14, and it is the clearest
single measurement of how far the defect escalates with workload size.

## The instrument could have shown otherwise

`BC250_DEVCHECK` reports non-zero sums at clean calls, at isolated zeroed calls
and inside cascades, so it is reporting rather than failing. Had the activations
collapsed during a cascade, this is exactly the measurement that would have
shown it, and instead they rise through the run as the context fills.

## Recovery is visible

Call 258 is zero and call 259, the very next, is correct at 67809.6 with healthy
operands. Whatever state produces a cascade is released between one call and the
next, without any batch boundary in between, which constrains how persistent the
underlying condition can be.
