# The defect escalates from one call to whole batches, 2026-08-23

Generated with `BC250_TEMP`, `GGML_CUDA_NO_POOL=1`, qwen3-8B at context 2048,
ubatch 512, three chunks: twelve batches of 36 GEMM calls, 432 in total. The
prediction is in `prediction.txt` and the raw positions in `runs.txt`.

## What was asked

At two chunks the zeros split into a stable core at batch starts and a
contiguous cascade beginning at call 253, whose length varied wildly. Call 253
is the start of the last batch at two chunks, so the cascade might be anchored to
the last batch or to a fixed call number. Three chunks separates those: the last
batch then starts at 397, not 253.

## Neither. It is anchored to batch starts, any of them

| run | isolated zeros | contiguous cascades | perplexity |
|---|---|---|---|
| 1 | 37, 73, 109, 181, 217 | 253 to 288, 325 to 432 | 569.7033 |
| 2 | 37, 73, 109, 181 | 325 to 432 | 224.6106 |
| 3 | 37, 73, 109, 181 | 325 to 326, 361 to 406 | 55.7436 |

The cascades begin at 253, 325 and 361 across runs, never at a fixed call, and
every one of those is a batch start: 1 + 36k for k of 7, 9 and 10.

## The shape of it

Two behaviours, not one.

Early batches lose only their first call. That part is the familiar pattern and
it is stable: 37, 73, 109 and 181 are zero in every run at both two and three
chunks.

Later batches lose everything. Cascade lengths are 36, 108, 108, 46 and 2 calls.
Thirty-six is exactly one batch; a hundred and eight is exactly three. In run 1
the cascade at 253 covers batch eight entirely, batch nine then runs clean, and
from batch ten to the end everything is zero. So a batch is often either
"clean except its first call" or "entirely zero", and once the second state is
reached late in a run it usually persists to the end.

The two exceptions, lengths 2 and 46, do not fit whole batches, so the rule is a
tendency rather than a law.

## What it changes

The model was "one zeroed GEMM per batch after the first". That is right for
short runs and wrong for long ones. The defect escalates: the same failure that
costs one call early costs an entire batch later, and eventually every remaining
call. The damage to perplexity follows, 14.13 at one chunk, 13.04 to 42.07 at
two, 55.74 to 569.70 at three.

That also explains why perplexity was such a poor instrument for this. It
compresses a structured, escalating failure into one number that moves by orders
of magnitude for reasons the number itself cannot show.

## What is still unknown

What determines when a batch escalates from losing one call to losing all of
them, and why the escalation is not reliably permanent, since run 3 recovers
after two calls at 325 and again after 406.

Corrected on 25 August. The two-chunk upper bound previously read 39.35, which is
the second-highest of the six two-chunk runs in
[`../fp16-boundary-scope-2026-08-23/runs.txt`](../fp16-boundary-scope-2026-08-23/runs.txt);
the highest is 42.0691. The three-chunk bounds and the one-chunk value were
already right. The error understated the defect rather than overstating it.
