# The fp16 defect still reproduces, and still never the same way twice, 2026-08-25

`verify.out` is the run: two rounds on the 8B, alternating
`GGML_CUDA_CUBLAS_COMPUTE_TYPE` between `f32` and `f16`, context 2048 over two
wikitext chunks, which is the method
[`../defects-recheck-2026-08-22/`](../defects-recheck-2026-08-22/) used.

| round | `f32` | `f16` |
|---|---|---|
| 1 | 9.0975 | 18.0833 |
| 2 | 9.0975 | 19.3402 |

Both halves of the defect entry hold. The workaround arm is bit-identical to the
figure recorded three days earlier and on every earlier occasion, 9.0975. The
defect arm is wrong both times and wrong by a different amount each time.

## What this adds

Five `f16` values now exist for this configuration across two dates: 17.1068,
14.7398 and 21.4429 on 22 August, and 18.0833 and 19.3402 today. All five differ,
spanning 14.7398 to 21.4429, against an `f32` arm that has never moved off
9.0975. The non-determinism is not a small perturbation of a wrong answer; the
spread between wrong runs is larger than the gap between the correct answer and
the nearest wrong one.

That matters for anyone reading the defect table. A single `f16` measurement would
look like a fixed accuracy penalty, and it is not one. There is no fp16 number to
quote for this path, only a distribution of wrong ones.

## Why run it again

The week put about a dozen instrumented amdgpu builds on this board and took them
off again. The last restoration was by hand. A defect that reproduced before that
churn is not evidence that it reproduces after it, and this is the only open
correctness defect the repository still carries.
